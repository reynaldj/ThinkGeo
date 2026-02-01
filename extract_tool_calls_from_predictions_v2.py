"""
Extract Tool Call Training Data from Prediction Files - V2
Ensures all tools are represented in train/val, with random sampling for test

This script extracts individual tool calls from LLM prediction files (like ThinkGeo_bench_0.json)
and creates training data for the tool choice classifier.

Strategy:
1. Collect ALL positive examples and group by tool
2. For each tool, randomly select min_samples_per_tool examples for train
3. Remaining examples go to test
4. Track test sample indices in separate JSON file
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict, Counter


def normalize_tool_name(name: str) -> str:
    """Normalize tool name for comparison."""
    return name.strip().lower().replace(" ", "").replace("-", "").replace("_", "")


def extract_tool_calls_from_dialog(dialogs: List[Dict]) -> List[Dict]:
    """
    Extract all tool calls from a dialog sequence.
    Handles both:
    - tool_calls format: messages with 'tool_calls' list containing function calls
    - content format: messages with 'content' list containing tool_use items
    """
    if dialogs and isinstance(dialogs[0], list):
        flat_dialogs = []
        for sublist in dialogs:
            if isinstance(sublist, list):
                flat_dialogs.extend(sublist)
        dialogs = flat_dialogs
    
    tool_calls = []
    context = []
    
    for msg in dialogs:
        if not isinstance(msg, dict):
            continue
        
        context.append(msg)
        
        if msg.get("role") == "assistant":
            # Handle tool_calls format (new format from predictions)
            if msg.get("tool_calls"):
                for tool_call in msg["tool_calls"]:
                    func = tool_call.get("function", {})
                    tool_calls.append({
                        "context": context.copy(),
                        "thought": msg.get("thought", ""),
                        "tool_name": func.get("name", ""),
                        "tool_arguments": func.get("arguments", {})
                    })
            
            # Handle content format (alternative format)
            elif msg.get("content"):
                content = msg["content"]
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "tool_use":
                            tool_calls.append({
                                "context": context.copy(),
                                "thought": msg.get("thought", ""),
                                "tool_name": item.get("name", ""),
                                "tool_arguments": item.get("input", {})
                            })
    
    return tool_calls


def compare_tool_calls(tool1: str, tools2: List[str]) -> bool:
    """Compare tool names after normalization."""
    norm1 = normalize_tool_name(tool1)
    for t2 in tools2:
        if normalize_tool_name(t2) == norm1:
            return True
    return False


def extract_from_all_model_runs(
    outputs_base_dir: str,
    toolmeta_path: str,
    output_dir: str,
    ensure_all_tools: bool = True,
    min_samples_per_tool: int = 10
):
    """
    Extract training data ensuring all tools are represented in train set.
    
    Strategy:
    - Collect ALL positive examples grouped by tool
    - For each tool, randomly select min_samples_per_tool for train/val
    - Remaining examples go to test
    - Track and save test sample indices
    """
    print("="*80)
    print("Discovering All Model Run Prediction Files")
    print("="*80)
    
    outputs_path = Path(outputs_base_dir) / "default"
    
    if not outputs_path.exists():
        print(f"Error: Output base directory not found: {outputs_path}")
        return
    
    # Discover all prediction directories
    pred_dirs = []
    for model_run_dir in sorted(outputs_path.iterdir()):
        if not model_run_dir.is_dir():
            continue
        
        pred_path = model_run_dir / "predictions"
        if pred_path.exists():
            for model_subdir in pred_path.iterdir():
                if model_subdir.is_dir():
                    pred_dirs.append(model_subdir)
    
    print(f"\nFound {len(pred_dirs)} model prediction directories")
    for pred_dir in pred_dirs:
        print(f"  - {pred_dir.relative_to(outputs_path.parent)}")
    
    # PASS 1: Collect ALL positive examples grouped by tool
    print(f"\n{'='*80}")
    print(f"[PASS 1] Collecting all positive examples grouped by tool")
    print(f"{'='*80}\n")
    
    all_positive_examples_by_tool = defaultdict(list)
    all_sample_indices = set()
    example_to_metadata = {}  # Map example to (tool, qid, model_run, file, global_idx)
    
    for pred_dir in pred_dirs:
        pred_files = sorted(pred_dir.glob("ThinkGeo_bench_*.json"))
        
        if not pred_files:
            continue
        
        model_run_name = pred_dir.parent.parent.name
        print(f"Processing {len(pred_files)} files from: {model_run_name}")
        
        for pred_file in pred_files:
            with open(pred_file, 'r') as f:
                predictions = json.load(f)
            
            with open(toolmeta_path, 'r') as f:
                toolmeta = json.load(f)
            
            for qid in sorted(predictions.keys(), key=lambda x: int(x) if x.isdigit() else float('inf')):
                qdata = predictions.get(qid)
                if not isinstance(qdata, dict):
                    continue
                
                gold_dialogs = qdata.get("gold", [])
                origin_prompts = qdata.get("origin_prompt", [])
                
                if not gold_dialogs:
                    continue
                
                # Calculate global sample index
                file_num = int(pred_file.stem.split('_')[-1])
                qid_int = int(qid)
                global_sample_idx = file_num * 49 + qid_int
                all_sample_indices.add(global_sample_idx)
                
                gold_tool_calls = extract_tool_calls_from_dialog(gold_dialogs)
                
                for step_idx, gold_call in enumerate(gold_tool_calls):
                    full_history = []
                    if step_idx < len(origin_prompts):
                        history_msgs = origin_prompts[step_idx]
                        if isinstance(history_msgs, list):
                            full_history = history_msgs
                    
                    if not full_history:
                        full_history = gold_call["context"][-10:]
                    
                    tool_name = gold_call["tool_name"]
                    tool_desc = toolmeta.get(tool_name, {}).get(
                        "description",
                        f"Tool: {tool_name}"
                    )
                    
                    example = {
                        "question_id": qid,
                        "step": step_idx,
                        "full_history": full_history,
                        "thought": gold_call["thought"],
                        "tool_name": tool_name,
                        "tool_description": tool_desc,
                        "tool_arguments": gold_call["tool_arguments"],
                        "gold_tool": tool_name,
                        "label": 1,
                        "source_file": pred_file.name,
                        "model_run": model_run_name,
                        "global_sample_idx": global_sample_idx
                    }
                    
                    example_id = id(example)
                    all_positive_examples_by_tool[tool_name].append(example)
                    example_to_metadata[example_id] = {
                        "tool": tool_name,
                        "qid": qid,
                        "model_run": model_run_name,
                        "file": pred_file.name,
                        "global_sample_idx": global_sample_idx
                    }
    
    print(f"\nCollected positive examples by tool:")
    for tool, examples in sorted(all_positive_examples_by_tool.items()):
        print(f"  {tool}: {len(examples)} examples")
    
    print(f"\nTotal unique sample indices: {len(all_sample_indices)}")
    print(f"Sample index range: {min(all_sample_indices)} to {max(all_sample_indices)}")
    
    # PASS 2: Select SAMPLES (not individual examples) ensuring all tools in both sets
    print(f"\n{'='*80}")
    print(f"[PASS 2] Selecting samples ensuring all tools in both train/val and test")
    print(f"{'='*80}\n")
    
    # Group examples by sample index and track tools per sample
    examples_by_sample = defaultdict(list)
    tools_per_sample = defaultdict(set)  # Which tools appear in each sample
    samples_by_tool = defaultdict(set)   # Which samples have which tools
    
    for tool, examples in all_positive_examples_by_tool.items():
        for ex in examples:
            sample_idx = ex["global_sample_idx"]
            examples_by_sample[sample_idx].append(ex)
            tools_per_sample[sample_idx].add(tool)
            samples_by_tool[tool].add(sample_idx)
    
    # Strategy: For each tool, ensure representation in both sets
    # Start with completely separate sets
    train_val_sample_indices = set()
    test_sample_indices = set()
    all_samples = set(examples_by_sample.keys())
    
    # For each tool, allocate samples to train/val and test
    tool_assignments = {tool: {'train_val': set(), 'test': set()} for tool in samples_by_tool.keys()}
    
    print("Allocating samples to ensure all tools in both sets:")
    for tool in sorted(samples_by_tool.keys()):
        available_samples = list(samples_by_tool[tool])
        total_available = len(available_samples)
        
        # Calculate split
        if total_available <= min_samples_per_tool:
            num_train_val = max(1, total_available - 1)  # Keep at least 1 for test if possible
            num_test = max(1, total_available - num_train_val)
        else:
            num_train_val = max(min_samples_per_tool, int(total_available * 0.7))
            num_test = total_available - num_train_val
        
        # Shuffle for randomness
        random.shuffle(available_samples)
        
        # Assign samples
        tool_train_val = set(available_samples[:num_train_val])
        tool_test = set(available_samples[num_train_val:num_train_val + num_test])
        
        tool_assignments[tool]['train_val'] = tool_train_val
        tool_assignments[tool]['test'] = tool_test
        
        print(f"  {tool}: {len(tool_train_val)} samples for train/val, {len(tool_test)} for test (from {total_available})")
    
    # Now create final assignment ensuring no overlap
    # Priority: if a sample is needed for test (especially for rare tools), put it in test
    assigned_to_test = set()
    assigned_to_train_val = set()
    
    # First pass: assign samples that are ONLY in test for any tool
    for tool, assignments in tool_assignments.items():
        exclusive_test = assignments['test'] - assignments['train_val']
        assigned_to_test.update(exclusive_test)
    
    # Second pass: assign samples that are ONLY in train/val for any tool
    for tool, assignments in tool_assignments.items():
        exclusive_train_val = assignments['train_val'] - assignments['test']
        assigned_to_train_val.update(exclusive_train_val - assigned_to_test)
    
    # Third pass: handle samples that appear in both (multi-tool samples)
    # Prefer test for rare tools, train/val for common tools
    unassigned = all_samples - assigned_to_test - assigned_to_train_val
    
    for sample in unassigned:
        sample_tools = tools_per_sample[sample]
        # Count how many times each set needs this sample
        test_needs = sum(1 for t in sample_tools if sample in tool_assignments[t]['test'])
        train_val_needs = sum(1 for t in sample_tools if sample in tool_assignments[t]['train_val'])
        
        # Check if any rare tool needs it for test
        rare_tool_needs_test = any(
            len(samples_by_tool[t]) <= min_samples_per_tool and sample in tool_assignments[t]['test']
            for t in sample_tools
        )
        
        if rare_tool_needs_test or test_needs > train_val_needs:
            assigned_to_test.add(sample)
        else:
            assigned_to_train_val.add(sample)
    
    train_val_sample_indices = assigned_to_train_val
    test_sample_indices = assigned_to_test
    
    # Verify no overlap
    overlap = train_val_sample_indices & test_sample_indices
    print(f"\nFinal assignment:")
    print(f"  Train/Val samples: {len(train_val_sample_indices)}")
    print(f"  Test samples: {len(test_sample_indices)}")
    print(f"  Overlap: {len(overlap)}")
    print(f"  Total unique: {len(train_val_sample_indices | test_sample_indices)}")
    
    # Verify all tools are represented in both sets
    train_val_tools = set()
    test_tools = set()
    for sample in train_val_sample_indices:
        train_val_tools.update(tools_per_sample[sample])
    for sample in test_sample_indices:
        test_tools.update(tools_per_sample[sample])
    
    print(f"\nTool coverage:")
    print(f"  Tools in train/val: {len(train_val_tools)}/{len(samples_by_tool)}")
    print(f"  Tools in test: {len(test_tools)}/{len(samples_by_tool)}")
    missing_from_test = set(samples_by_tool.keys()) - test_tools
    if missing_from_test:
        print(f"  WARNING - Missing from test: {missing_from_test}")
    
    # Now extract examples based on sample assignment
    train_val_positive_examples = []
    test_positive_examples = []
    
    for sample_idx, examples in examples_by_sample.items():
        if sample_idx in train_val_sample_indices:
            train_val_positive_examples.extend(examples)
        elif sample_idx in test_sample_indices:
            test_positive_examples.extend(examples)
    
    print(f"\nExtracted examples:")
    print(f"  Train/Val positive: {len(train_val_positive_examples)}")
    print(f"  Test positive: {len(test_positive_examples)}")
    
    # PASS 3: Extract negative examples for train/val
    print(f"\n{'='*80}")
    print(f"[PASS 3] Extracting negative examples")
    print(f"{'='*80}\n")
    
    train_val_negative_examples = []
    test_negative_examples = []
    train_val_target_negatives = len(train_val_positive_examples)
    
    # Build set of (file, qid, model_run) that are in train/val
    train_val_allowed_qids = set()
    for ex in train_val_positive_examples:
        train_val_allowed_qids.add((ex["source_file"], ex["question_id"], ex["model_run"]))
    
    print(f"Targeting {train_val_target_negatives} negative examples for train/val...")
    
    for pred_dir in pred_dirs:
        pred_files = sorted(pred_dir.glob("ThinkGeo_bench_*.json"))
        
        for pred_file in pred_files:
            if len(train_val_negative_examples) >= train_val_target_negatives:
                break
            
            with open(pred_file, 'r') as f:
                predictions = json.load(f)
            
            with open(toolmeta_path, 'r') as f:
                toolmeta = json.load(f)
            
            for qid in sorted(predictions.keys(), key=lambda x: int(x) if x.isdigit() else float('inf')):
                if (pred_file.name, qid, pred_dir.parent.parent.name) not in train_val_allowed_qids:
                    continue
                
                qdata = predictions.get(qid)
                if not isinstance(qdata, dict):
                    continue
                
                gold_dialogs = qdata.get("gold", [])
                pred_dialogs = qdata.get("prediction", [])
                origin_prompts = qdata.get("origin_prompt", [])
                
                if not gold_dialogs or not pred_dialogs:
                    continue
                
                if len(train_val_negative_examples) >= train_val_target_negatives:
                    break
                
                file_num = int(pred_file.stem.split('_')[-1])
                qid_int = int(qid)
                global_sample_idx = file_num * 49 + qid_int
                
                gold_tool_calls = extract_tool_calls_from_dialog(gold_dialogs)
                pred_tool_calls = extract_tool_calls_from_dialog(pred_dialogs)
                
                gold_tools_by_position = {}
                for i, call in enumerate(gold_tool_calls):
                    gold_tools_by_position[i] = call["tool_name"]
                
                for pred_step_idx, pred_call in enumerate(pred_tool_calls):
                    if len(train_val_negative_examples) >= train_val_target_negatives:
                        break
                    
                    gold_tool = gold_tools_by_position.get(pred_step_idx)
                    pred_tool = pred_call["tool_name"]
                    
                    if not (gold_tool and compare_tool_calls(pred_tool, [gold_tool])):
                        full_history = []
                        if pred_step_idx < len(origin_prompts):
                            history_msgs = origin_prompts[pred_step_idx]
                            if isinstance(history_msgs, list):
                                full_history = history_msgs
                        
                        if not full_history:
                            full_history = pred_call["context"][-10:]
                        
                        tool_desc = toolmeta.get(pred_tool, {}).get(
                            "description",
                            f"Tool: {pred_tool}"
                        )
                        
                        example = {
                            "question_id": qid,
                            "step": pred_step_idx,
                            "full_history": full_history,
                            "thought": pred_call["thought"],
                            "tool_name": pred_tool,
                            "tool_description": tool_desc,
                            "tool_arguments": pred_call["tool_arguments"],
                            "gold_tool": gold_tool if gold_tool else "N/A",
                            "label": 0,
                            "source_file": pred_file.name,
                            "model_run": pred_dir.parent.parent.name,
                            "global_sample_idx": global_sample_idx
                        }
                        train_val_negative_examples.append(example)
        
        if len(train_val_negative_examples) >= train_val_target_negatives:
            break
    
    # PASS 4: Extract negative examples for test
    print(f"\n{'='*80}")
    print(f"[PASS 4] Extracting test set negative examples")
    print(f"{'='*80}\n")
    
    test_negative_examples = []
    test_target_negatives = len(test_positive_examples)
    print(f"Test positive examples: {len(test_positive_examples)}")
    print(f"Targeting {test_target_negatives} negative examples for test...")
    
    # Build set of (file, qid, model_run) that are in test positive
    test_allowed_qids = set()
    for ex in test_positive_examples:
        test_allowed_qids.add((ex["source_file"], ex["question_id"], ex["model_run"]))
    
    for pred_dir in pred_dirs:
        pred_files = sorted(pred_dir.glob("ThinkGeo_bench_*.json"))
        
        for pred_file in pred_files:
            if len(test_negative_examples) >= test_target_negatives:
                break
            
            with open(pred_file, 'r') as f:
                predictions = json.load(f)
            
            with open(toolmeta_path, 'r') as f:
                toolmeta = json.load(f)
            
            for qid in sorted(predictions.keys(), key=lambda x: int(x) if x.isdigit() else float('inf')):
                # Only process samples in test set
                if (pred_file.name, qid, pred_dir.parent.parent.name) not in test_allowed_qids:
                    continue
                
                qdata = predictions.get(qid)
                if not isinstance(qdata, dict):
                    continue
                
                gold_dialogs = qdata.get("gold", [])
                pred_dialogs = qdata.get("prediction", [])
                origin_prompts = qdata.get("origin_prompt", [])
                
                if not gold_dialogs or not pred_dialogs:
                    continue
                
                if len(test_negative_examples) >= test_target_negatives:
                    break
                
                file_num = int(pred_file.stem.split('_')[-1])
                qid_int = int(qid)
                global_sample_idx = file_num * 49 + qid_int
                
                gold_tool_calls = extract_tool_calls_from_dialog(gold_dialogs)
                pred_tool_calls = extract_tool_calls_from_dialog(pred_dialogs)
                
                gold_tools_by_position = {}
                for i, call in enumerate(gold_tool_calls):
                    gold_tools_by_position[i] = call["tool_name"]
                
                for pred_step_idx, pred_call in enumerate(pred_tool_calls):
                    if len(test_negative_examples) >= test_target_negatives:
                        break
                    
                    gold_tool = gold_tools_by_position.get(pred_step_idx)
                    pred_tool = pred_call["tool_name"]
                    
                    if not (gold_tool and compare_tool_calls(pred_tool, [gold_tool])):
                        full_history = []
                        if pred_step_idx < len(origin_prompts):
                            history_msgs = origin_prompts[pred_step_idx]
                            if isinstance(history_msgs, list):
                                full_history = history_msgs
                        
                        if not full_history:
                            full_history = pred_call["context"][-10:]
                        
                        tool_desc = toolmeta.get(pred_tool, {}).get(
                            "description",
                            f"Tool: {pred_tool}"
                        )
                        
                        example = {
                            "question_id": qid,
                            "step": pred_step_idx,
                            "full_history": full_history,
                            "thought": pred_call["thought"],
                            "tool_name": pred_tool,
                            "tool_description": tool_desc,
                            "tool_arguments": pred_call["tool_arguments"],
                            "gold_tool": gold_tool if gold_tool else "N/A",
                            "label": 0,
                            "source_file": pred_file.name,
                            "model_run": pred_dir.parent.parent.name,
                            "global_sample_idx": global_sample_idx
                        }
                        test_negative_examples.append(example)
        
        if len(test_negative_examples) >= test_target_negatives:
            break
    
    # Print statistics
    print(f"\n{'='*80}")
    print("Combined Extraction Statistics")
    print(f"{'='*80}")
    
    print(f"\nTRAIN/VAL SET:")
    print(f"  Positive examples: {len(train_val_positive_examples)}")
    print(f"  Negative examples: {len(train_val_negative_examples)}")
    print(f"  Total: {len(train_val_positive_examples) + len(train_val_negative_examples)}")
    
    print(f"\nTEST SET:")
    print(f"  Positive examples: {len(test_positive_examples)}")
    print(f"  Negative examples: {len(test_negative_examples)}")
    print(f"  Total: {len(test_positive_examples) + len(test_negative_examples)}")
    
    # Tool distribution
    train_val_pos_tools = Counter([ex["tool_name"] for ex in train_val_positive_examples])
    train_val_neg_tools = Counter([ex["tool_name"] for ex in train_val_negative_examples])
    test_pos_tools = Counter([ex["tool_name"] for ex in test_positive_examples])
    test_neg_tools = Counter([ex["tool_name"] for ex in test_negative_examples])
    
    print(f"\nTools in TRAIN/VAL POSITIVE examples:")
    for tool, count in train_val_pos_tools.most_common():
        print(f"  {tool}: {count}")
    
    print(f"\nTools in TEST POSITIVE examples:")
    for tool, count in test_pos_tools.most_common():
        print(f"  {tool}: {count}")
    
    missing_in_train_val = set(test_pos_tools.keys()) - set(train_val_pos_tools.keys())
    if missing_in_train_val:
        print(f"\nWARNING - Tools still missing in train/val:")
        for tool in missing_in_train_val:
            print(f"  {tool}: {test_pos_tools[tool]} in test")
    
    # Combine and shuffle train/val data
    train_val_data = train_val_positive_examples + train_val_negative_examples
    random.shuffle(train_val_data)
    
    # Split train/val into 70% train, 30% val
    train_val_total = len(train_val_data)
    train_size = int(train_val_total * 0.7)
    
    train_data = train_val_data[:train_size]
    val_data = train_val_data[train_size:]
    
    # Combine and shuffle test data
    test_data = test_positive_examples + test_negative_examples
    random.shuffle(test_data)
    
    print(f"\n{'='*80}")
    print("Dataset Splits")
    print(f"{'='*80}")
    print(f"Train/Val Total: {train_val_total} examples")
    print(f"  Train: {len(train_data)} examples (70% of train/val)")
    print(f"    Positive: {len([x for x in train_data if x['label']==1])}")
    print(f"    Negative: {len([x for x in train_data if x['label']==0])}")
    print(f"  Val: {len(val_data)} examples (30% of train/val)")
    print(f"    Positive: {len([x for x in val_data if x['label']==1])}")
    print(f"    Negative: {len([x for x in val_data if x['label']==0])}")
    print(f"\nTest Total: {len(test_data)} examples")
    print(f"  Positive: {len([x for x in test_data if x['label']==1])}")
    print(f"  Negative: {len([x for x in test_data if x['label']==0])}")
    
    # Save to files
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "train.json", 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"\nSaved train data to: {output_path / 'train.json'}")
    
    with open(output_path / "val.json", 'w') as f:
        json.dump(val_data, f, indent=2)
    print(f"Saved val data to: {output_path / 'val.json'}")
    
    with open(output_path / "test.json", 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"Saved test data to: {output_path / 'test.json'}")
    
    # Save test sample indices
    test_indices_data = {
        "test_sample_indices": sorted(list(test_sample_indices)),
        "num_test_samples": len(test_sample_indices),
        "total_samples": len(all_sample_indices),
        "train_val_sample_indices": sorted(list(train_val_sample_indices)),
        "num_train_val_samples": len(train_val_sample_indices)
    }
    
    with open(output_path / "test_sample_indices.json", 'w') as f:
        json.dump(test_indices_data, f, indent=2)
    print(f"Saved test sample indices to: {output_path / 'test_sample_indices.json'}")
    
    # Save statistics
    stats = {
        "strategy": "Ensure all tools in train/val, random sampling for test",
        "min_samples_per_tool": min_samples_per_tool,
        "train_val_positive_examples": len(train_val_positive_examples),
        "train_val_negative_examples": len(train_val_negative_examples),
        "test_positive_examples": len(test_positive_examples),
        "test_negative_examples": len(test_negative_examples),
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "train_val_positive_tool_distribution": dict(train_val_pos_tools),
        "train_val_negative_tool_distribution": dict(train_val_neg_tools),
        "test_positive_tool_distribution": dict(test_pos_tools),
        "test_negative_tool_distribution": dict(test_neg_tools),
        "model_run_distribution": dict(Counter([ex["model_run"] for ex in train_val_positive_examples])),
        "num_model_runs": len(pred_dirs),
        "num_prediction_files": sum(len(list(d.glob("ThinkGeo_bench_*.json"))) for d in pred_dirs)
    }
    
    with open(output_path / "dataset_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to: {output_path / 'dataset_stats.json'}")
    
    print(f"\n{'='*80}")
    print("Extraction Complete!")
    print(f"{'='*80}\n")
    
    return train_data, val_data, test_data


if __name__ == "__main__":
    # Configuration
    OUTPUTS_BASE_DIR = "/home/james/ThinkGeo/opencompass/outputs"
    TOOLMETA_PATH = "/home/james/ThinkGeo/opencompass/data/ThinkGeo_dataset/toolmeta.json"
    OUTPUT_DIR = "/home/james/ThinkGeo/tool_choice_data_from_predictions"
    
    # Extract with new strategy
    extract_from_all_model_runs(
        outputs_base_dir=OUTPUTS_BASE_DIR,
        toolmeta_path=TOOLMETA_PATH,
        output_dir=OUTPUT_DIR,
        ensure_all_tools=True,
        min_samples_per_tool=10
    )
