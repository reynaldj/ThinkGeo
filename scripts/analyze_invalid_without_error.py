#!/usr/bin/env python3
"""
Quantitative analysis of "invalid without explicit error" tool calls in ThinkGeo predictions.

Usage:
  python scripts/analyze_invalid_without_error.py \
      --pred-files /path/to/ThinkGeo_bench_0.json /path/to/ThinkGeo_bench_1.json

The script reports counts of:
- explicit_error: prediction step has an error field
- missing_tool: gold expects a tool call, prediction has none
- tool_mismatch: prediction uses a tool but the gold either expects an answer or a different tool
- arg_mismatch: same tool as gold but arguments don't match ground truth

These categories approximate the silent invalid cases that still hurt inst_align, tool_acc, arg_acc.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class Counters:
    explicit_error: int = 0
    missing_tool: int = 0
    tool_mismatch: int = 0
    arg_mismatch: int = 0
    tool_in_answer_step: int = 0  # pred calls tool when gold expects answer (without error)
    tool_in_answer_step_with_error: int = 0  # pred calls tool when gold expects answer (with error)
    tool_correct: int = 0  # correct tool + correct args + no error
    explicit_error_missing_tool: int = 0  # gold expects tool, pred has error and no tool
    explicit_error_tool_mismatch: int = 0  # gold expects tool, pred has error and wrong tool
    explicit_error_arg_mismatch: int = 0  # gold expects tool, pred has error, tool matches but args differ
    classifier_rejection: int = 0  # errors with type CLASSIFIER_REJECTION
    total_steps: int = 0
    
    # Detailed breakdowns
    tool_mismatch_pairs: Counter = field(default_factory=Counter)  # (gold_tool, pred_tool) -> count
    arg_mismatch_details: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))  # tool -> {error_type: count}
    missing_tool_by_expected: Counter = field(default_factory=Counter)  # gold_tool -> count
    tool_in_answer_step_by_tool: Counter = field(default_factory=Counter)  # tool -> count
    tool_in_answer_step_with_error_by_tool: Counter = field(default_factory=Counter)  # tool -> count
    tool_correct_by_tool: Counter = field(default_factory=Counter)  # tool -> count
    explicit_error_tool_mismatch_pairs: Counter = field(default_factory=Counter)  # (gold_tool, pred_tool) -> count
    explicit_error_arg_mismatch_details: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))  # tool -> {error_type: count}

    def add(self, other: "Counters") -> None:
        self.explicit_error += other.explicit_error
        self.missing_tool += other.missing_tool
        self.tool_mismatch += other.tool_mismatch
        self.arg_mismatch += other.arg_mismatch
        self.tool_in_answer_step += other.tool_in_answer_step
        self.tool_in_answer_step_with_error += other.tool_in_answer_step_with_error
        self.tool_correct += other.tool_correct
        self.explicit_error_missing_tool += other.explicit_error_missing_tool
        self.explicit_error_tool_mismatch += other.explicit_error_tool_mismatch
        self.explicit_error_arg_mismatch += other.explicit_error_arg_mismatch
        self.classifier_rejection += other.classifier_rejection
        self.total_steps += other.total_steps
        
        # Merge detailed counters
        self.tool_mismatch_pairs.update(other.tool_mismatch_pairs)
        for tool, errors in other.arg_mismatch_details.items():
            self.arg_mismatch_details[tool].update(errors)
        self.missing_tool_by_expected.update(other.missing_tool_by_expected)
        self.tool_in_answer_step_by_tool.update(other.tool_in_answer_step_by_tool)
        self.tool_in_answer_step_with_error_by_tool.update(other.tool_in_answer_step_with_error_by_tool)
        self.tool_correct_by_tool.update(other.tool_correct_by_tool)
        self.explicit_error_tool_mismatch_pairs.update(other.explicit_error_tool_mismatch_pairs)
        for tool, errors in other.explicit_error_arg_mismatch_details.items():
            self.explicit_error_arg_mismatch_details[tool].update(errors)

    def summary(self) -> Dict[str, float]:
        if self.total_steps == 0:
            return {"total_steps": 0}
        return {
            "total_steps": self.total_steps,
            "explicit_error": self.explicit_error,
            "classifier_rejection": self.classifier_rejection,
            "missing_tool": self.missing_tool,
            "tool_mismatch": self.tool_mismatch,
            "arg_mismatch": self.arg_mismatch,
            "tool_in_answer_step": self.tool_in_answer_step,
            "tool_in_answer_step_with_error": self.tool_in_answer_step_with_error,
            "tool_correct": self.tool_correct,
            "explicit_error_missing_tool": self.explicit_error_missing_tool,
            "explicit_error_tool_mismatch": self.explicit_error_tool_mismatch,
            "explicit_error_arg_mismatch": self.explicit_error_arg_mismatch,
            "invalid_without_error": self.missing_tool
            + self.tool_mismatch
            + self.arg_mismatch,
            "invalid_without_error_rate": (
                (self.missing_tool
                 + self.tool_mismatch
                 + self.arg_mismatch)
                / self.total_steps
            ),
        }


_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def _is_math(expr: str) -> bool:
    return bool(re.fullmatch(r"[0-9+\-*/(). %]+", expr.strip()))


def _bbox_ok(bbox: Any) -> bool:
    if isinstance(bbox, str):
        nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", bbox)
        return len(nums) >= 4
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return all(isinstance(x, (int, float)) for x in bbox[:4])
    return False


def _arg_check(tool: str, args: Dict[str, Any], gold_args: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Check if args match ground truth arguments.
    Returns (is_valid, list_of_error_reasons)
    
    Strategy: First check if arguments are exactly equal. If not, provide specific error reasons.
    This catches both format errors AND semantic mismatches (wrong values).
    """
    errors = []
    
    # First, check exact equality (strictest check) - this catches ALL argument mismatches
    if args == gold_args:
        return True, []
    
    # If not exactly equal, analyze why for better diagnostics
    if tool == "TextToBbox":
        # Check each argument individually
        if args.get("image") != gold_args.get("image"):
            errors.append("image_mismatch")
        
        if args.get("text") != gold_args.get("text"):
            errors.append("text_mismatch")
        
        if args.get("top1") != gold_args.get("top1"):
            errors.append("top1_mismatch")
        
        return False, errors

    if tool == "Calculator":
        gold_expr = gold_args.get("expression", "")
        pred_expr = args.get("expression", "")
        
        if pred_expr != gold_expr:
            # Check if it's a format issue or semantic issue
            if not isinstance(pred_expr, str):
                errors.append("expression_not_string")
            elif not _is_math(pred_expr):
                errors.append("expression_not_math")
            else:
                # Valid format but different value
                errors.append("expression_mismatch")
        
        return False, errors

    if tool == "DrawBox":
        if args.get("bbox") != gold_args.get("bbox"):
            if not _bbox_ok(args.get("bbox")):
                errors.append("invalid_bbox_format")
            else:
                # Valid format but wrong coordinates
                errors.append("bbox_mismatch")
        
        return False, errors

    if tool == "CountGivenObject":
        if args.get("image") != gold_args.get("image"):
            errors.append("image_mismatch")
        
        if args.get("name") != gold_args.get("name"):
            errors.append("name_mismatch")
        
        return False, errors

    if tool == "RegionAttributeDescription":
        if args.get("image") != gold_args.get("image"):
            errors.append("image_mismatch")
        
        if args.get("region") != gold_args.get("region"):
            errors.append("region_mismatch")
        
        return False, errors

    if tool == "ImageDescription":
        if args.get("image") != gold_args.get("image"):
            errors.append("image_mismatch")
        
        return False, errors
    
    if tool == "GetObjects":
        if args.get("image") != gold_args.get("image"):
            errors.append("image_mismatch")
        
        return False, errors
    
    if tool == "ObjectsDistance":
        if args.get("object1") != gold_args.get("object1"):
            errors.append("object1_mismatch")
        
        if args.get("object2") != gold_args.get("object2"):
            errors.append("object2_mismatch")
        
        return False, errors
    
    if tool == "GetImageURL":
        if args.get("image") != gold_args.get("image"):
            errors.append("image_mismatch")
        
        return False, errors
    
    if tool == "ChangeDetection":
        if args.get("image1") != gold_args.get("image1"):
            errors.append("image1_mismatch")
        
        if args.get("image2") != gold_args.get("image2"):
            errors.append("image2_mismatch")
        
        return False, errors
    
    if tool == "SegmentObjectPixels":
        if args.get("image") != gold_args.get("image"):
            errors.append("image_mismatch")
        
        if args.get("object_name") != gold_args.get("object_name"):
            errors.append("object_name_mismatch")
        
        return False, errors
    
    if tool == "Solver":
        if args.get("equation") != gold_args.get("equation"):
            errors.append("equation_mismatch")
        
        return False, errors
    
    if tool in {"Plot", "OCR", "AddText", "GoogleSearch"}:
        # For these tools, compare all arguments and report mismatches
        for key in set(list(gold_args.keys()) + list(args.keys())):
            if args.get(key) != gold_args.get(key):
                errors.append(f"{key}_mismatch")
        
        return False, errors
    
    # Fallback: compare all arguments field by field
    for key in set(list(gold_args.keys()) + list(args.keys())):
        if args.get(key) != gold_args.get(key):
            errors.append(f"{key}_mismatch")
    
    return False, errors


def analyze_file(path: Path) -> Counters:
    data = json.loads(path.read_text())
    counters = Counters()

    for task in data.values():
        pred_steps: List[Dict[str, Any]] = task.get("prediction", [])
        gold_steps: List[Dict[str, Any]] = task.get("gold", [])

        for idx, pred_step in enumerate(pred_steps):
            counters.total_steps += 1

            gold_step = gold_steps[idx] if idx < len(gold_steps) else {}
            gold_tools = gold_step.get("tool_calls", []) or []
            pred_tools = pred_step.get("tool_calls", []) or []

            g_has_tool = len(gold_tools) > 0
            g_is_answer = gold_step.get("role") == "assistant" and not g_has_tool
            p_has_tool = len(pred_tools) > 0
            has_error = "error" in pred_step

            # Check for tool in answer step with error BEFORE checking explicit_error
            if g_is_answer and p_has_tool and has_error:
                counters.explicit_error += 1
                counters.tool_in_answer_step_with_error += 1
                p_tool = pred_tools[0]["function"]["name"]
                counters.tool_in_answer_step_with_error_by_tool[p_tool] += 1
                # Check if it's a CLASSIFIER_REJECTION
                error_obj = pred_step.get("error", {})
                if isinstance(error_obj, dict) and error_obj.get("type") == "CLASSIFIER_REJECTION":
                    counters.classifier_rejection += 1
                continue

            if has_error:
                # Check if it's a CLASSIFIER_REJECTION
                error_obj = pred_step.get("error", {})
                if isinstance(error_obj, dict) and error_obj.get("type") == "CLASSIFIER_REJECTION":
                    counters.classifier_rejection += 1
                
                # Attribute error cases to tool categories when possible
                if g_has_tool:
                    if not p_has_tool:
                        counters.explicit_error_missing_tool += 1
                    else:
                        g_tool = gold_tools[0]["function"]["name"]
                        p_tool = pred_tools[0]["function"]["name"]
                        if g_tool != p_tool:
                            counters.explicit_error_tool_mismatch += 1
                            counters.explicit_error_tool_mismatch_pairs[(g_tool, p_tool)] += 1
                        else:
                            p_args = pred_tools[0]["function"].get("arguments", {}) or {}
                            g_args = gold_tools[0]["function"].get("arguments", {}) or {}
                            if not isinstance(p_args, dict) or not isinstance(g_args, dict):
                                counters.explicit_error_arg_mismatch += 1
                                counters.explicit_error_arg_mismatch_details[p_tool]["invalid_arguments_type"] += 1
                            else:
                                is_valid, error_reasons = _arg_check(p_tool, p_args, g_args)
                                if not is_valid:
                                    counters.explicit_error_arg_mismatch += 1
                                    for reason in error_reasons:
                                        counters.explicit_error_arg_mismatch_details[p_tool][reason] += 1
                counters.explicit_error += 1
                continue

            if g_has_tool and not p_has_tool:
                counters.missing_tool += 1
                g_tool = gold_tools[0]["function"]["name"]
                counters.missing_tool_by_expected[g_tool] += 1
                continue

            if g_is_answer and p_has_tool:
                counters.tool_mismatch += 1
                p_tool = pred_tools[0]["function"]["name"]
                counters.tool_mismatch_pairs[("ANSWER", p_tool)] += 1
                counters.tool_in_answer_step += 1
                counters.tool_in_answer_step_by_tool[p_tool] += 1
                continue

            if not g_has_tool and not g_is_answer and p_has_tool:
                counters.tool_mismatch += 1
                p_tool = pred_tools[0]["function"]["name"]
                counters.tool_mismatch_pairs[("ANSWER", p_tool)] += 1
                counters.tool_in_answer_step += 1
                counters.tool_in_answer_step_by_tool[p_tool] += 1
                continue

            if g_has_tool and p_has_tool:
                g_tool = gold_tools[0]["function"]["name"]
                p_tool = pred_tools[0]["function"]["name"]
                if g_tool != p_tool:
                    counters.tool_mismatch += 1
                    counters.tool_mismatch_pairs[(g_tool, p_tool)] += 1
                    continue

                p_args = pred_tools[0]["function"].get("arguments", {}) or {}
                g_args = gold_tools[0]["function"].get("arguments", {}) or {}
                if not isinstance(p_args, dict) or not isinstance(g_args, dict):
                    counters.arg_mismatch += 1
                    counters.arg_mismatch_details[p_tool]["invalid_arguments_type"] += 1
                else:
                    is_valid, error_reasons = _arg_check(p_tool, p_args, g_args)
                    if not is_valid:
                        counters.arg_mismatch += 1
                        for reason in error_reasons:
                            counters.arg_mismatch_details[p_tool][reason] += 1
                    else:
                        # Correct tool with correct args and no error
                        counters.tool_correct += 1
                        counters.tool_correct_by_tool[p_tool] += 1

    return counters


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze invalid-without-error tool calls.")
    parser.add_argument("--pred-files", nargs="+", required=True, help="Prediction JSON files to analyze")
    parser.add_argument("--detailed", action="store_true", help="Show detailed breakdowns")
    parser.add_argument("--output", "-o", type=str, help="Output file path (optional, prints to stdout if not provided)")
    args = parser.parse_args()
    
    # Setup output stream
    import sys
    if args.output:
        output_file = open(args.output, 'w')
        original_stdout = sys.stdout
        sys.stdout = output_file
    else:
        output_file = None

    total = Counters()
    for file_str in args.pred_files:
        path = Path(file_str)
        if not path.exists():
            print(f"[warn] file not found: {path}")
            continue
        stats = analyze_file(path)
        total.add(stats)
        print(f"=== {path.name} ===")
        for k, v in stats.summary().items():
            if isinstance(v, float):
                print(f"{k:25s}: {v:.4f}")
            else:
                print(f"{k:25s}: {v}")
        print()

    print("=" * 80)
    print("=== AGGREGATED ===")
    print("=" * 80)
    for k, v in total.summary().items():
        if isinstance(v, float):
            print(f"{k:25s}: {v:.4f}")
        else:
            print(f"{k:25s}: {v}")
    
    if args.detailed:
        print("\n" + "=" * 80)
        print("=== TOOL MISMATCH BREAKDOWN ===")
        print("=" * 80)
        if total.tool_mismatch_pairs:
            print(f"{'Expected Tool':<30} {'Predicted Tool':<30} {'Count':<10}")
            print("-" * 80)
            for (gold_tool, pred_tool), count in total.tool_mismatch_pairs.most_common(20):
                print(f"{gold_tool:<30} {pred_tool:<30} {count:<10}")
        else:
            print("No tool mismatches found.")
        
        print("\n" + "=" * 80)
        print("=== MISSING TOOL BREAKDOWN (by expected tool) ===")
        print("=" * 80)
        if total.missing_tool_by_expected:
            print(f"{'Expected Tool':<40} {'Count':<10}")
            print("-" * 80)
            for tool, count in total.missing_tool_by_expected.most_common():
                print(f"{tool:<40} {count:<10}")
        else:
            print("No missing tools found.")
        
        print("\n" + "=" * 80)
        print("=== TOOL CORRECT BREAKDOWN (correct tool + correct args + no error) ===")
        print("=" * 80)
        print(f"Total correct tool calls: {total.tool_correct}")
        if total.tool_correct_by_tool:
            print(f"{'Tool':<40} {'Count':<10}")
            print("-" * 80)
            for tool, count in total.tool_correct_by_tool.most_common():
                print(f"{tool:<40} {count:<10}")
        else:
            print("No correct tool calls found.")

        print("\n" + "=" * 80)
        print("=== EXPLICIT ERROR TOOL BREAKDOWN (gold expects tool, pred has error) ===")
        print("=" * 80)
        print(f"Missing tool (with error): {total.explicit_error_missing_tool}")
        print(f"Tool mismatch (with error): {total.explicit_error_tool_mismatch}")
        print(f"Arg mismatch (with error): {total.explicit_error_arg_mismatch}")
        if total.explicit_error_tool_mismatch_pairs:
            print(f"{'Expected Tool':<30} {'Predicted Tool':<30} {'Count':<10}")
            print("-" * 80)
            for (g_tool, p_tool), count in total.explicit_error_tool_mismatch_pairs.most_common():
                print(f"{g_tool:<30} {p_tool:<30} {count:<10}")
        else:
            print("No explicit-error tool mismatches found.")

        if total.explicit_error_arg_mismatch_details:
            print("\nArg mismatch (with error) details:")
            for tool in sorted(total.explicit_error_arg_mismatch_details.keys()):
                errors = total.explicit_error_arg_mismatch_details[tool]
                print(f"  {tool}:")
                for error_type, count in errors.most_common():
                    print(f"    {error_type:<30} {count}")
        
        print("\n" + "=" * 80)
        print("=== TOOL IN ANSWER STEP BREAKDOWN (pred calls tool, gold expects answer) ===")
        print("=" * 80)
        print(f"Without error: {total.tool_in_answer_step}")
        print(f"With error: {total.tool_in_answer_step_with_error}")
        print(f"Total: {total.tool_in_answer_step + total.tool_in_answer_step_with_error}")
        print()
        
        if total.tool_in_answer_step_by_tool or total.tool_in_answer_step_with_error_by_tool:
            print(f"{'Predicted Tool':<40} {'Without Error':<15} {'With Error':<15} {'Total':<10}")
            print("-" * 80)
            all_tools = set(total.tool_in_answer_step_by_tool.keys()) | set(total.tool_in_answer_step_with_error_by_tool.keys())
            for tool in sorted(all_tools):
                no_err = total.tool_in_answer_step_by_tool.get(tool, 0)
                with_err = total.tool_in_answer_step_with_error_by_tool.get(tool, 0)
                total_count = no_err + with_err
                print(f"{tool:<40} {no_err:<15} {with_err:<15} {total_count:<10}")
        else:
            print("No tool calls in answer steps found.")
        
        print("\n" + "=" * 80)
        print("=== ARGUMENT MISMATCH BREAKDOWN (by tool and error type) ===")
        print("=" * 80)
        if total.arg_mismatch_details:
            for tool in sorted(total.arg_mismatch_details.keys()):
                errors = total.arg_mismatch_details[tool]
                print(f"\n{tool}:")
                print(f"  {'Error Type':<40} {'Count':<10}")
                print("  " + "-" * 60)
                for error_type, count in errors.most_common():
                    print(f"  {error_type:<40} {count:<10}")
        else:
            print("No argument mismatches found.")
    
    # Close output file if opened
    if output_file:
        sys.stdout = original_stdout
        output_file.close()
        print(f"Analysis written to: {args.output}")


if __name__ == "__main__":
    main()
