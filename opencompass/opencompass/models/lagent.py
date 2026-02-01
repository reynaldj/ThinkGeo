from copy import deepcopy
import types
import json
import re
import ast
import sys
from typing import List, Tuple, Any, Dict
import os

from mmengine.registry import Registry

REGISTRY = Registry('helper')

try:
    import lagent
    import agentlego
except ImportError:
    lagent = None
    agentlego = None

# Import tool classifier
try:
    import torch
    from transformers import AutoTokenizer
    # Import from parent directory - adjust path if needed
    sys.path.insert(0, '/home/james/ThinkGeo')
    from simple_tool_classifier import ToolChoiceValidator
    CLASSIFIER_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Tool classifier not available: {e}", file=sys.stderr)
    CLASSIFIER_AVAILABLE = False
    ToolChoiceValidator = None

class DummyTool(agentlego.tools.BaseTool):

    def __init__(self, toolmeta):
        # Convert dict to ToolMeta object
        if isinstance(toolmeta, dict):
            # Convert inputs/outputs from dicts to Parameter objects
            def convert_params(param_dicts):
                """Convert list of param dicts to Parameter objects"""
                if not param_dicts:
                    return []
                params = []
                for p in param_dicts:
                    if isinstance(p, dict):
                        # Normalize common schema type strings to Python types
                        try:
                            type_map = {
                                'text': str,
                                'string': str,
                                'image': str,
                                'path': str,
                                'int': int,
                                'integer': int,
                                'float': float,
                                'double': float,
                                'bool': bool,
                                'boolean': bool,
                            }
                            if 'type' in p and isinstance(p['type'], str):
                                mapped = type_map.get(p['type'].lower())
                                if mapped is not None:
                                    p['type'] = mapped
                        except Exception:
                            # if normalization fails, continue without change
                            pass
                        # Try to create Parameter object
                        try:
                            # Check if Parameter class exists
                            if hasattr(agentlego.schema, 'Parameter'):
                                param = agentlego.schema.Parameter(**p)
                            elif hasattr(agentlego.schema, 'InputParameter'):
                                # Use InputParameter for inputs
                                param = agentlego.schema.InputParameter(**p)
                            else:
                                # Fallback: create object with attributes
                                from types import SimpleNamespace
                                param = SimpleNamespace(**p)
                            params.append(param)
                        except (TypeError, AttributeError):
                            # If Parameter creation fails, use SimpleNamespace
                            from types import SimpleNamespace
                            params.append(SimpleNamespace(**p))
                    else:
                        params.append(p)
                return params
            
            inputs = convert_params(toolmeta.get('inputs', []))
            outputs = convert_params(toolmeta.get('outputs', []))
            
            # Try to create ToolMeta
            try:
                # Method 1: Direct instantiation
                self.toolmeta = agentlego.schema.ToolMeta(
                    name=toolmeta.get('name', ''),
                    description=toolmeta.get('description', ''),
                    inputs=inputs,
                    outputs=outputs
                )
            except (TypeError, AttributeError):
                try:
                    # Method 2: Try with **toolmeta (but with converted inputs/outputs)
                    meta_dict = toolmeta.copy()
                    meta_dict['inputs'] = inputs
                    meta_dict['outputs'] = outputs
                    self.toolmeta = agentlego.schema.ToolMeta(**meta_dict)
                except (TypeError, AttributeError):
                    try:
                        # Method 3: Try from_dict
                        self.toolmeta = agentlego.schema.ToolMeta.from_dict(toolmeta)
                    except AttributeError:
                        # Method 4: Create SimpleNamespace as fallback
                        from types import SimpleNamespace
                        self.toolmeta = SimpleNamespace(
                            name=toolmeta.get('name', ''),
                            description=toolmeta.get('description', ''),
                            inputs=inputs,
                            outputs=outputs
                        )
        else:
            self.toolmeta = toolmeta
        self.set_parser(agentlego.parsers.DefaultParser)
        self._is_setup = False

    def apply(self, *args, **kwargs):
        return 'Dummy Result'

def dummy_action_executor(tools):
    return lagent.ActionExecutor(
        actions=[DummyTool(cfg).to_lagent() for cfg in tools])


def model_adapter(model):
    """Modify the generate method to accept and return single item."""
    if getattr(model, '_generate_is_wrapped', False):
        # Avoid wrap twice.
        return model

    from opencompass.utils import PromptList

    def chat(self, inputs, *args, **kwargs):
        prompt = PromptList()
        for item in inputs:
            msg = {'prompt': item['content']}
            if item['role'] == 'user':
                msg['role'] = 'HUMAN'
            elif item['role'] == 'assistant':
                msg['role'] = 'BOT'
            elif item['role'] == 'system':
                msg['role'] = 'SYSTEM'
            prompt.append(msg)
        return self.generate([prompt], *args, **kwargs)[0]

    model.chat = types.MethodType(chat, model)
    setattr(model, '_generate_is_wrapped', True)
    return model


def react_style_history(history, files, protocol) -> List[dict]:
    from lagent.schema import ActionReturn
    inner_steps = []
    if files:
        prompt = 'The related files are at ' + ', '.join(f'`{file["path"]}`'
                                                         for file in files)
        inner_steps.append(dict(role='system', content=prompt))
    for step in history:
        if step['role'] == 'user':
            inner_steps.append(dict(role='user', content=step['content']))
        elif step['role'] == 'assistant' and step.get('tool_calls'):
            name = step['tool_calls'][0]['function']['name']
            args = step['tool_calls'][0]['function']['arguments']
            response = "{action}{name}\n{action_input}{args}".format(
                action=protocol.action['begin'],
                name=name,
                action_input=protocol.action_input['begin'],
                args=json.dumps(args),
            )
            inner_steps.append(dict(role='assistant', content=response))
        elif step['role'] == 'tool' and step.get('content'):
            # action= ActionReturn(result=[dict(type='text', content=step['content'])])
            action= ActionReturn(result=[step['content']])
            inner_steps.append(dict(role='system', content=action.format_result()))
        elif step['role'] == 'assistant' and step.get('content'):
            inner_steps.append(dict(role='assistant', content=step['content']))
    return inner_steps


def _build_examples_from_tools(tools: List[Any], files: List[dict] | None = None) -> str:
    """Build concrete examples (Action + fenced JSON Action Input) for tools.

    Args:
        tools: list of tool objects (lagent tool instances) exposing `toolmeta`.
        files: optional list of file resources (dicts with `path`) to use
               as realistic image path examples when available.

    Returns:
        A string containing example calls appended to the prompt's tool
        description to teach the model exact field names and types.
    """
    examples = []
    # If dataset files provided, use the first file path as a realistic image example
    sample_path = None
    try:
        if files and isinstance(files, list) and len(files) > 0:
            sample_path = files[0].get('path')
    except Exception:
        sample_path = None

    for t in tools:
        try:
            name = getattr(t, 'name', None) or (
                getattr(t, 'toolmeta', None) and getattr(t.toolmeta, 'name', None)
            ) or str(t)
            meta = getattr(t, 'toolmeta', None)
            # Robustly extract parameter list from several possible shapes:
            # - meta.inputs (list)
            # - meta.parameters (list)
            # - meta may itself be a dict with 'inputs'/'parameters'
            inputs = []
            param_list = None
            try:
                # attribute access (objects)
                param_list = getattr(meta, 'inputs', None) or getattr(meta, 'parameters', None)
            except Exception:
                param_list = None
            # dict-like meta
            if param_list is None and isinstance(meta, dict):
                param_list = meta.get('inputs') or meta.get('parameters')

            if param_list is None:
                param_list = []

            # Debug: surface param_list shape so we can see why examples may be empty
            try:
                pl_len = len(param_list) if hasattr(param_list, '__len__') else 'N/A'
                print(f"DEBUG: tool '{name}' param_list type={type(param_list)} len={pl_len}")
            except Exception:
                pass

            for p in param_list or []:
                # p may be object or dict
                if isinstance(p, dict):
                    p_name = p.get('name')
                    p_type = p.get('type')
                    p_default = p.get('default')
                else:
                    p_name = getattr(p, 'name', None)
                    p_type = getattr(p, 'type', None)
                    p_default = getattr(p, 'default', None)

                # choose a realistic dummy value based on type/name
                val = 'example'
                ttype = (str(p_type) if p_type is not None else '').lower()
                pname = (str(p_name) if p_name is not None else '').lower()

                # prefer explicit default when available
                if p_default is not None:
                    val = p_default
                elif 'image' in ttype or 'image' in pname:
                    val = sample_path or '/home/james/ThinkGeo/opencompass/data/ThinkGeo_dataset/image/example.jpg'
                elif 'path' in ttype or 'path' in pname:
                    val = sample_path or '/home/james/ThinkGeo/opencompass/data/ThinkGeo_dataset/image/example.jpg'
                elif 'int' in ttype or 'integer' in ttype or pname.endswith('_idx'):
                    val = 1
                elif 'float' in ttype or 'double' in ttype:
                    val = 1.0
                elif 'bool' in ttype or 'boolean' in ttype or pname.startswith('is_') or pname.startswith('has_') or pname in ('top1', 'flag'):
                    val = True
                elif pname == 'bbox':
                    # keep bbox as string since many tools expect text bbox
                    val = '(50, 30, 200, 400)'
                elif pname == 'position':
                    val = '(50, 60)'
                elif pname == 'command' and ('plot' in name.lower() or 'solver' in name.lower()):
                    # include a small code block string for plot/solver
                    if 'plot' in name.lower():
                        val = "```python\nimport matplotlib.pyplot as plt\ndef solution():\n    fig = plt.figure()\n    plt.plot([1,2,3],[4,5,6])\n    return fig\n```"
                    else:
                        val = "```python\nfrom sympy import symbols, Eq, solve\ndef solution():\n    x = symbols('x')\n    return str(solve(Eq(x**2 - 4, 0)))\n```"
                else:
                    # default: short example string
                    val = 'example'

                if p_name:
                    inputs.append((p_name, val))
        except Exception:
            continue

        if not inputs:
            # No parameters inferred for this tool — still provide a minimal
            # empty-JSON example so the protocol receives at least one
            # example block. This helps the model follow the exact Action
            # / Action Input formatting even when tools take no args.
            try:
                examples.append(
                    f"Action: {name}\nAction Input:\n```json\n{{}}\n```\n"
                )
            except Exception:
                # Fallback: ignore if formatting fails
                pass
            continue

        # build JSON dict and pretty-print it as compact JSON
        obj = {k: v for k, v in inputs}
        examples.append(
            f"Action: {name}\nAction Input:\n```json\n{json.dumps(obj, ensure_ascii=False)}\n```\n"
        )
    return "\n".join(examples)


# def _build_demonstrations_from_bench(files: List[dict] | None,
#                                      allowed_tools: List[Any] | None,
#                                      max_examples: int = 3) -> str:
#     """Build concrete demonstrations from pre-extracted demonstrations.json.

#     - Loads demonstrations.json (pre-built by scripts/build_demonstrations.py).
#     - Filters to only include tools that are in `allowed_tools` (by name).
#     - Renders up to `max_examples` per tool using the standard
#       "Action:" + fenced JSON "Action Input:" format expected by the protocol.
#     """
#     # Locate the demonstrations file
#     demo_path = None
#     try:
#         if files:
#             for f in files:
#                 p = f.get('path')
#                 if p and os.path.basename(p) == 'demonstrations.json' and os.path.exists(p):
#                     demo_path = p
#                     break
#     except Exception:
#         demo_path = None
    
#     if demo_path is None:
#         # Fallback to default dataset location
#         candidate = '/home/james/ThinkGeo/opencompass/data/ThinkGeo_dataset/demonstrations.json'
#         demo_path = candidate if os.path.exists(candidate) else None

#     if demo_path is None:
#         return ""  # No demonstrations available

#     # Build allowed tool name set
#     allowed_names = set()
#     try:
#         if allowed_tools:
#             for t in allowed_tools:
#                 name = getattr(t, 'name', None) or (
#                     getattr(t, 'toolmeta', None) and getattr(t.toolmeta, 'name', None)
#                 ) or str(t)
#                 if name:
#                     allowed_names.add(name)
#     except Exception:
#         allowed_names = allowed_names  # keep as is

#     demonstrations: List[str] = []
#     try:
#         with open(demo_path, 'r', encoding='utf-8') as f:
#             demo_data = json.load(f)
        
#         if not isinstance(demo_data, dict):
#             return ""
        
#         # Iterate through tools in demonstrations.json
#         for tool_name, examples in demo_data.items():
#             if not isinstance(examples, list):
#                 continue
            
#             # Filter by allowed tools if provided
#             if allowed_names and tool_name not in allowed_names:
#                 continue
            
#             # Add up to max_examples for this tool
#             for example in examples[:max_examples]:
#                 if not isinstance(example, dict):
#                     continue
                
#                 name = example.get('name')
#                 args = example.get('arguments')
                
#                 if not isinstance(name, str) or not isinstance(args, dict):
#                     continue
                
#                 try:
#                     demonstrations.append(
#                         f"Action: {name}\nAction Input:\n```json\n{json.dumps(args, ensure_ascii=False)}\n```\n"
#                     )
#                 except Exception:
#                     # Skip if arguments cannot be serialized
#                     continue
#     except Exception:
#         return ""

#     if not demonstrations:
#         return ""

#     header = "Demonstrations (from ThinkGeoBench):\n"
#     return header + "\n".join(demonstrations)


def _expected_keys_for(tool_name: str) -> set:
    """Return expected/allowed keys for a given tool (best-effort)."""
    t = (tool_name or "").strip()
    mapping = {
        "TextToBbox": {"image", "text", "top1"},
        "CountGivenObject": {"image", "text", "bbox"},
        "RegionAttributeDescription": {"image", "bbox", "attribute"},
        "DrawBox": {"image", "bbox", "annotation"},
        "Plot": {"command", "timeout"},
        "Solver": {"command", "timeout"},
        "GoogleSearch": {"query", "k"},
        "Calculator": {"expression"},
        "ChangeDetection": {"pre_image", "post_image", "text"},
        "SegmentObjectPixels": {"image", "text", "flag"},
    }
    return mapping.get(t, set())


def _parse_and_sanitize_args(raw_args, expected_keys: set | None = None):
    """Try to convert model-emitted argument strings into Python objects.

    - Strips common Markdown code fences.
    - Extracts JSON-like substrings and chooses the one matching `expected_keys`.
    - Falls back to `ast.literal_eval` for Python literal parsing.
    Returns the original object when parsing isn't possible.
    """
    if not isinstance(raw_args, str):
        return raw_args

    # Strip fenced codeblocks ```json ... ``` or ```...```
    m = re.search(r"```\s*json\n(.*?)```", raw_args, re.S | re.I)
    if not m:
        m = re.search(r"```(?:\w*\n)?(.*?)```", raw_args, re.S)
    raw = m.group(1).strip() if m else raw_args.strip()

    # Collect all JSON-like candidates (non-greedy)
    candidates = re.findall(r"(\{.*?\}|\[.*?\])", raw, re.S)
    # If none found, use raw as single candidate
    if not candidates:
        candidates = [raw]

    def try_parse(text: str):
        # Try JSON
        try:
            return json.loads(text)
        except Exception:
            pass
        # Try Python literal eval (handles single quotes, True/False)
        try:
            return ast.literal_eval(text)
        except Exception:
            pass
        # As a last resort, try to coerce common Python tokens to JSON tokens
        try:
            norm = text.replace("None", "null").replace("True", "true").replace("False", "false")
            # convert single quotes to double quotes when safe
            if "'" in norm and '"' not in norm:
                norm2 = norm.replace("'", '"')
            else:
                norm2 = norm
            return json.loads(norm2)
        except Exception:
            return None

    # Prefer candidate that best matches expected_keys (if provided)
    best_obj = None
    best_score = -1
    for cand in candidates:
        obj = try_parse(cand)
        if obj is None:
            continue
        if isinstance(obj, dict) and expected_keys:
            score = len(set(obj.keys()) & set(expected_keys))
        elif isinstance(obj, list) and obj and isinstance(obj[0], dict) and expected_keys:
            # list of dicts: score based on union of first element keys
            score = len(set(obj[0].keys()) & set(expected_keys))
        else:
            # Without expected keys, keep first successful parse
            score = 0
        if score >= best_score:
            best_score = score
            best_obj = obj

    if best_obj is not None:
        return best_obj

    # Nothing worked — return the original (string) so downstream can handle it
    return raw_args


# --- Constraint helpers (Rule-based validator) ---

_MATH_EXPR_RE = re.compile(r"^[0-9+\-*/(). %]+$")


def _is_math_expr(expr: str) -> bool:
    if not isinstance(expr, str):
        return False
    return bool(_MATH_EXPR_RE.fullmatch(expr.strip()))


def _bbox_like_ok(bbox: Any) -> bool:
    """Check if bbox is in a valid format and optionally normalize it."""
    try:
        # Accept string containing at least four numbers (handles parens, brackets, etc.)
        if isinstance(bbox, str):
            # Strip common delimiters and extract numbers
            nums = re.findall(r"[-+]?\d*\.?\d+", bbox)
            return len(nums) >= 4
        # Accept list/tuple with 4 numeric entries
        if isinstance(bbox, (list, tuple)):
            if len(bbox) >= 4 and all(isinstance(x, (int, float)) for x in bbox[:4]):
                return True
            # list of boxes (each box is a list/tuple of 4+ numbers)
            if len(bbox) > 0 and all(
                isinstance(b, (list, tuple)) and len(b) >= 4 and all(isinstance(x, (int, float)) for x in b[:4])
                for b in bbox
            ):
                return True
        return False
    except Exception:
        return False


def _normalize_image_path(image_path: Any) -> Any:
    """Return image path unchanged.

    Note: We previously coerced absolute dataset paths to a relative
    `image/<file>` form. This normalization is now disabled so that
    predicted arguments can exactly match gold references which use
    absolute paths.
    """
    return image_path


def _normalize_bbox(bbox: Any) -> str:
    """Normalize bbox to string tuple format expected by evaluation.
    
    Converts various bbox formats to the standard string tuple format:
      "(x1, y1, x2, y2)"
    
    This matches the evaluation's expected format for exact comparison.
    """
    try:
        # If string with parens/brackets like "(50, 30, 200, 400)", extract numbers and reformat
        if isinstance(bbox, str):
            nums = [float(x) if '.' in x else int(x) for x in re.findall(r"[-+]?\d*\.?\d+", bbox)]
            if len(nums) >= 4:
                # Return string tuple format for single bbox
                return f"({nums[0]}, {nums[1]}, {nums[2]}, {nums[3]})"
        
        # If list/tuple with 4+ numbers, convert to string tuple
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            # Convert first 4 elements to string tuple
            return f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})"
        
        # Return as-is if format not recognized
        return bbox
    except Exception:
        return bbox


def _auto_correct_args(tool_name: str, args: Any, reason: str) -> tuple[bool, Any, str]:
    """Attempt lightweight, safe corrections so we can retry execution.

    Returns (changed, corrected_args, note). If changed=False, we failed to
    repair and caller should surface the ARGS_ERROR to the model.
    """
    note_parts = []

    # Re-parse strings that look like JSON, prefer schema-matching block
    if isinstance(args, str):
        parsed = _parse_and_sanitize_args(args, _expected_keys_for(tool_name))
        if parsed is not args:
            args = parsed
            note_parts.append("parsed string to object")

    # Unwrap singleton list/tuple containing a dict
    if isinstance(args, (list, tuple)) and len(args) == 1 and isinstance(args[0], dict):
        args = args[0]
        note_parts.append("unwrapped singleton list/tuple")

    if not isinstance(args, dict):
        return False, args, ""

    t = tool_name or ""

    # Drop unknown extras for tools with well-known schemas
    def drop_extras(allowed: set[str]):
        nonlocal args
        extras = set(args.keys()) - allowed
        if extras:
            args = {k: v for k, v in args.items() if k in allowed}
            note_parts.append(f"dropped extras {sorted(extras)}")

    if t == "TextToBbox":
        drop_extras({"image", "text", "top1"})
        # Default top1 to True if missing
        if "top1" not in args:
            args["top1"] = True
            note_parts.append("defaulted top1=True")
        # Coerce string bools
        v = args.get("top1")
        if isinstance(v, str) and v.lower() in {"true", "false"}:
            args["top1"] = v.lower() == "true"
            note_parts.append("coerced top1 bool")
        # If has bbox but no text, can't fix (DrawBox mistake)
        if "bbox" in args and "text" not in args:
            note_parts.append("has bbox but missing text; likely DrawBox args")

    elif t == "CountGivenObject":
        # Real schema: image (required), text (required), bbox (optional)
        # No mapping needed - 'text' is the correct field name, not 'name'
        drop_extras({"image", "text", "bbox"})
        # Remove bbox if it's null, empty string, or placeholder value (optional field)
        if "bbox" in args:
            bbox_val = args["bbox"]
            if bbox_val is None or bbox_val == "" or (isinstance(bbox_val, str) and bbox_val.lower() in ("null", "none", "n/a", "unknown", "not_given")):
                del args["bbox"]
                note_parts.append(f"removed null/empty bbox (optional field)")
            else:
                # Normalize bbox format
                args["bbox"] = _normalize_bbox(bbox_val)
                note_parts.append("normalized bbox to string tuple")

    elif t == "Plot":
        drop_extras({"command", "timeout"})
        # If command is nested under 'code'
        if "command" not in args and "code" in args and isinstance(args["code"], str):
            args["command"] = args["code"]
            note_parts.append("mapped code->command")

    elif t == "Solver":
        drop_extras({"command", "timeout"})
        if "command" not in args and "code" in args and isinstance(args["code"], str):
            args["command"] = args["code"]
            note_parts.append("mapped code->command")

    elif t == "GoogleSearch":
        drop_extras({"query", "k"})  # k is optional
        # If prompt provided under other key
        if "query" not in args and "text" in args and isinstance(args["text"], str):
            args["query"] = args["text"]
            note_parts.append("mapped text->query")

    elif t == "RegionAttributeDescription":
        # Real schema: image, bbox, attribute (all required)
        # normalize bbox if present; reject placeholder strings
        if "bbox" in args:
            bbox_val = args["bbox"]
            if isinstance(bbox_val, str):
                # reject placeholder strings like "not_given_yet", "unknown", etc.
                if bbox_val.lower() in ("not_given_yet", "unknown", "none", "null", "n/a", "tbd", "todo"):
                    note_parts.append(f"rejected placeholder bbox '{bbox_val}'")
                    # Can't fix; let validator reject it
                else:
                    args["bbox"] = _normalize_bbox(bbox_val)
                    note_parts.append("normalized bbox to string tuple")
            else:
                args["bbox"] = _normalize_bbox(bbox_val)
                note_parts.append("normalized bbox to string tuple")

    elif t == "DrawBox":
        # If tool got TextToBbox args, try to recover bbox if present
        if "bbox" not in args and "text" in args:
            # Can't auto-generate bbox from text; just drop the wrong fields
            note_parts.append("detected TextToBbox args for DrawBox; cannot auto-recover")
        if "bbox" in args:
            args["bbox"] = _normalize_bbox(args["bbox"])
            note_parts.append("normalized bbox to string tuple")

    elif t == "Calculator":
        # If expression hidden inside other key, try to extract
        if "expression" not in args:
            for k in ("expr", "equation", "calc", "formula", "math"):
                if k in args:
                    args["expression"] = args[k]
                    note_parts.append(f"mapped {k}->expression")
                    break
        # Drop extra fields like variables, units, type, etc.
        if "expression" in args:
            allowed = {"expression"}
            extras = set(args.keys()) - allowed
            if extras:
                args = {"expression": args["expression"]}
                note_parts.append(f"dropped Calculator extras {sorted(extras)}")

    changed = bool(note_parts)
    return changed, args, "; ".join(note_parts)


def _detect_tool_announcement_mismatch(thought_text: str, actual_tool_name: str) -> tuple[bool, str]:
    """Detect if LLM announced one tool in thought but called a different one.
    
    Returns (mismatch_detected, announced_tool_name)
    """
    if not thought_text:
        return False, ""
    
    # Extract tool names mentioned in the thought
    tool_names = {
        'TextToBbox', 'RegionAttributeDescription', 'CountGivenObject',
        'DrawBox', 'Plot', 'Solver', 'GoogleSearch', 'Calculator',
        'ChangeDetection', 'SegmentObjectPixels', 'OCR', 'ImageDescription',
        'MathOCR', 'AddText', 'TextToImage', 'ImageStylization',
        'SemanticSegmentation', 'ObjectDetection'
    }
    
    # Find tools mentioned (case-insensitive)
    mentioned = set()
    for tool in tool_names:
        if tool.lower() in thought_text.lower():
            mentioned.add(tool)
    
    # Check if actual tool differs from announced tools
    if mentioned and actual_tool_name not in mentioned:
        # Return first announced tool for reference
        announced = sorted(mentioned)[0] if mentioned else ""
        return True, announced
    
    return False, ""


def _validate_tool_choice_with_classifier(
    context: str,
    history: str = "",
    tool_name: str = "",
    tool_description: str = "",
    argument_schema: Dict = None,
    classifier: 'ToolChoiceValidator' = None,
    confidence_threshold: float = 0.5
) -> tuple[bool, str, float]:
    """Validate tool choice using the trained classifier.
    
    Args:
        context: User's original query/context
        history: String with tool calls from history (formatted as training script expects)
        tool_name: Name of tool being called
        tool_description: Tool description
        argument_schema: Dict of argument names to masked type values
        classifier: Initialized ToolChoiceValidator instance
        confidence_threshold: Confidence threshold for accepting prediction (default 0.5)
    
    Returns:
        (is_valid, reason, confidence) where:
        - is_valid: True if classifier predicts correct tool
        - reason: Explanation message
        - confidence: Model confidence score [0, 1]
    """
    if not CLASSIFIER_AVAILABLE or classifier is None:
        return True, "Classifier not available", 1.0
    
    try:
        # Debug: Print classifier input with full details
        print(f"\n[CLASSIFIER_INPUT]", file=sys.stderr)
        print(f"  Context: {context[:100]}{'...' if len(context) > 100 else ''}", file=sys.stderr)
        print(f"  History: {history if history else '(empty)'}", file=sys.stderr)
        print(f"  Tool: {tool_name}", file=sys.stderr)
        print(f"  Schema: {argument_schema}", file=sys.stderr)
        
        is_correct, confidence = classifier.validate(
            context=context,
            tool_name=tool_name,
            tool_description=tool_description,
            history=history,
            argument_schema=argument_schema,
            return_confidence=True
        )
        
        if is_correct:
            msg = f"Tool choice '{tool_name}' validated (confidence: {confidence:.3f})"
            return True, msg, confidence
        else:
            msg = f"Tool choice '{tool_name}' rejected by classifier (confidence: {confidence:.3f}). Consider using a different tool."
            return False, msg, confidence
    
    except Exception as e:
        print(f"[CLASSIFIER_ERROR] Tool choice validation failed: {str(e)}", file=sys.stderr)
        return True, f"Classifier error (proceeding): {str(e)}", 1.0


def _extract_tool_mentions_ordered(thought_text: str) -> list[str]:
    """Extract tool mentions in order of first appearance in the thought text."""
    if not thought_text:
        return []
    tool_names = [
        'TextToBbox', 'RegionAttributeDescription', 'CountGivenObject',
        'DrawBox', 'Plot', 'Solver', 'GoogleSearch', 'Calculator',
        'ChangeDetection', 'SegmentObjectPixels', 'OCR', 'ImageDescription',
        'MathOCR', 'AddText', 'TextToImage', 'ImageStylization',
        'SemanticSegmentation', 'ObjectDetection'
    ]
    text_l = thought_text.lower()
    found = []
    for tool in tool_names:
        idx = text_l.find(tool.lower())
        if idx >= 0:
            found.append((idx, tool))
    found.sort(key=lambda x: x[0])
    # dedupe preserving order
    ordered = []
    seen = set()
    for _, t in found:
        if t not in seen:
            ordered.append(t)
            seen.add(t)
    return ordered


def _validate_constraints(tool_name: str, args: Any) -> tuple[bool, str, Any]:
    """Validate and minimally coerce arguments for common tools.

    Returns (ok, reason, coerced_args). If ok=False, reason explains the
    violation. If ok=True, coerced_args is the (possibly sanitized) args.
    """
    # Block NoAction early with a clear directive to choose tool or finish.
    if tool_name == "NoAction":
        message = (
            "NoAction is not allowed. Use one of these formats:\n"
            "- Tool call: Action: <ToolName> and Action Input: <JSON args>\n"
            "- Finish: Final Answer: <your answer>"
        )
        return False, message, args

    # Ensure arguments are a dict (tools expect a mapping of fields)
    if isinstance(args, str):
        args = _parse_and_sanitize_args(args, _expected_keys_for(tool_name))
    if not isinstance(args, dict):
        return False, "arguments must be a JSON object (dict)", args

    # Note: Do not normalize image paths here; keep them as emitted to match gold exactly.

    # Per-tool minimal schemas (only strict where we observed frequent issues)
    t = tool_name or ""

    if t == "Calculator":
        expr = args.get("expression")
        if expr is None:
            return False, "missing required field 'expression'", args
        # try to coerce nested structures to string
        if not isinstance(expr, str):
            try:
                expr = str(expr)
                args["expression"] = expr
            except Exception:
                return False, "field 'expression' must be a string", args
        
        # Aggressive cleaning: strip common noise patterns
        # 1. Remove code fences
        expr = re.sub(r"```(?:python|json|math)?\s*", "", expr)
        # 2. Remove "Response:" prefix and everything before it
        if "Response:" in expr:
            expr = expr.split("Response:", 1)[1]
        # 3. Remove comments (Python # and inline explanations)
        expr = re.sub(r"#.*", "", expr)
        # 4. Remove "Thought:", "Action:", etc.
        expr = re.sub(r"(?i)(thought|action|action input|response):\s*", "", expr)
        # 5. Extract first valid math expression if mixed with text
        # Look for patterns like "distance = ...", "calculate: ...", etc.
        expr = expr.strip()
        
        # If it contains variable assignment, try to extract RHS
        if "=" in expr and not expr.startswith("=="):
            parts = expr.split("=", 1)
            if len(parts) == 2:
                expr = parts[1].strip()
        
        # If still contains non-math tokens, extract the math-like portion
        if not _is_math_expr(expr):
            # Try to find a continuous math expression
            m = re.search(r"([0-9+\-*/(). %]+)", expr)
            if m:
                expr = m.group(1)
            else:
                # No valid math found - reject
                return False, f"field 'expression' contains no valid math expression (got: {expr[:100]})", args
        
        args["expression"] = expr.strip()

    elif t == "TextToBbox":
        for key in ("image", "text", "top1"):
            if key not in args:
                return False, f"missing required field '{key}'", args
        if not isinstance(args.get("image"), str):
            return False, "field 'image' must be a string path", args
        if not isinstance(args.get("text"), str):
            return False, "field 'text' must be a string", args
        if not isinstance(args.get("top1"), bool):
            # try to coerce common values
            v = args.get("top1")
            if isinstance(v, str) and v.lower() in {"true", "false"}:
                args["top1"] = v.lower() == "true"
            else:
                return False, "field 'top1' must be a boolean", args
        # Reject unexpected extras that downstream tool won't accept
        allowed = {"image", "text", "top1"}
        extras = set(args.keys()) - allowed
        if extras:
            return False, f"unknown arguments: {sorted(extras)}", args

    elif t == "DrawBox":
        for key in ("image", "bbox"):
            if key not in args:
                return False, f"missing required field '{key}'", args
        if not isinstance(args.get("image"), str):
            return False, "field 'image' must be a string path", args
        bbox_val = args.get("bbox")
        if not _bbox_like_ok(bbox_val):
            return False, "field 'bbox' must be a box or list of boxes (4 numbers)", args
        # Normalize bbox format
        args["bbox"] = _normalize_bbox(bbox_val)

    elif t == "CountGivenObject":
        # Real schema: image (required), text (required), bbox (optional)
        for key in ("image", "text"):
            if key not in args:
                return False, f"missing required field '{key}'", args
        if not isinstance(args.get("image"), str):
            return False, "field 'image' must be a string path", args
        if not isinstance(args.get("text"), str):
            return False, "field 'text' must be a string", args
        # Validate optional bbox if present
        if "bbox" in args:
            bbox_val = args.get("bbox")
            if not _bbox_like_ok(bbox_val):
                return False, "field 'bbox' is not a valid box format", args
            args["bbox"] = _normalize_bbox(bbox_val)
        # Drop unknown extras
        allowed = {"image", "text", "bbox"}
        extras = set(args.keys()) - allowed
        if extras:
            return False, f"unknown arguments: {sorted(extras)}", args

    elif t == "RegionAttributeDescription":
        # Real schema: image, bbox, attribute (all required)
        for key in ("image", "bbox", "attribute"):
            if key not in args:
                return False, f"missing required field '{key}'", args
        if not isinstance(args.get("image"), str):
            return False, "field 'image' must be a string path", args
        if not isinstance(args.get("attribute"), str):
            return False, "field 'attribute' must be a string", args
        bbox_val = args.get("bbox")
        if not _bbox_like_ok(bbox_val):
            return False, "field 'bbox' is not a valid box format", args
        # Normalize bbox format
        args["bbox"] = _normalize_bbox(bbox_val)

    elif t == "Plot":
        # Expect code string under 'command'; reject wrong shapes/keys.
        if "command" not in args:
            return False, "missing required field 'command'", args
        if not isinstance(args.get("command"), str):
            return False, "field 'command' must be a string", args
        allowed = {"command", "timeout"}
        extras = set(args.keys()) - allowed
        if extras:
            return False, f"unknown arguments: {sorted(extras)}", args

    elif t == "Solver":
        if "command" not in args:
            return False, "missing required field 'command'", args
        if not isinstance(args.get("command"), str):
            return False, "field 'command' must be a string", args
        allowed = {"command", "timeout"}
        extras = set(args.keys()) - allowed
        if extras:
            return False, f"unknown arguments: {sorted(extras)}", args

    elif t == "GoogleSearch":
        if "query" not in args:
            return False, "missing required field 'query'", args
        if not isinstance(args.get("query"), str):
            return False, "field 'query' must be a string", args
        allowed = {"query", "k"}  # k is optional
        extras = set(args.keys()) - allowed
        if extras:
            return False, f"unknown arguments: {sorted(extras)}", args

    # Other tools: best effort, require dict already satisfied
    return True, "", args


class LagentAgent:
    """Agent wrapper for Lagent.

    https://github.com/InternLM/lagent.
    """
    is_api = True

    def __init__(self,
                 agent_type,
                 llm,
                 actions=None,
                 tool_server=None,
                 tool_meta=None,
                 protocol=None,
                 **kwargs):
        llm = model_adapter(REGISTRY.build(llm))
        agent_cfg = {'type': agent_type, 'llm': llm, **kwargs}
        tools = {}
        if actions is not None:
            for action in actions:
                action = REGISTRY.build(action)
                if isinstance(action, agentlego.tools.BaseTool):
                    action = action.to_lagent()
                tools[action.name] = action
        if tool_server is not None:
            from agentlego.tools.remote import RemoteTool
            for tool in RemoteTool.from_server(tool_server):
                tools[tool.name] = tool.to_lagent()
        if tool_meta is not None:
            # Safely open tool_meta JSON file to avoid ResourceWarning
            with open(tool_meta, 'r', encoding='utf-8') as f:
                metas = json.load(f)
            for meta in metas.values():
                tool = DummyTool(meta).to_lagent()
                tools.setdefault(tool.name, tool)

        self.tools = tools

        if protocol is not None:
            protocol = REGISTRY.build(protocol)
            # mmengine's config building can serialize constants as dotted
            # import paths (e.g. 'opencompass.models.lagent.FEWSHOT_INSTRUCTION'),
            # which then appear verbatim in templates. If that happened,
            # try to resolve those dotted references back to the actual
            # Python objects (usually strings) so the protocol.format()
            # emits the real instruction text.
            try:
                def _resolve_dotted_ref(val):
                    if not isinstance(val, str):
                        return val
                    # Heuristic: looks like a dotted path with an attribute
                    parts = val.split('.')
                    if len(parts) < 2:
                        return val
                    mod_path = '.'.join(parts[:-1])
                    attr = parts[-1]
                    # Skip obvious non-dotted strings (contain whitespace or newlines)
                    if '\n' in val or ' ' in val:
                        return val
                    try:
                        mod = __import__(mod_path, fromlist=[attr])
                        return getattr(mod, attr)
                    except Exception:
                        return val

                if hasattr(protocol, 'call_protocol'):
                    protocol.call_protocol = _resolve_dotted_ref(
                        protocol.call_protocol)
                if hasattr(protocol, 'force_stop'):
                    protocol.force_stop = _resolve_dotted_ref(
                        protocol.force_stop)
            except Exception:
                # If resolution fails, fall back to whatever was built.
                pass

            agent_cfg['protocol'] = protocol

        from lagent import BaseAgent, ActionExecutor
        agent_cfg['action_executor'] = ActionExecutor(tools.values())
        self.agent: BaseAgent = REGISTRY.build(agent_cfg)
        
        # Initialize tool classifier if available
        self.tool_classifier = None
        # Lowered from 0.5 to 0.45 to accept first-step tool calls with lower confidence
        # (model tends to have lower confidence on first-step calls due to training data distribution)
        self.classifier_confidence_threshold = kwargs.get('classifier_confidence_threshold', 0.45)
        self.classifier_enabled = kwargs.get('classifier_enabled', False)  # DISABLED by default
        
        if CLASSIFIER_AVAILABLE and ToolChoiceValidator is not None and self.classifier_enabled:
            try:
                classifier_model_path = kwargs.get(
                    'classifier_model_path',
                    '/home/james/ThinkGeo/checkpoints_augmented/best_model.pth'
                )
                if os.path.exists(classifier_model_path):
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    self.tool_classifier = ToolChoiceValidator(
                        model_path=classifier_model_path,
                        device=device
                    )
                    print(f"[CLASSIFIER] Loaded tool choice classifier from {classifier_model_path}", file=sys.stderr)
                else:
                    print(f"[CLASSIFIER] Model not found at {classifier_model_path}", file=sys.stderr)
            except Exception as e:
                print(f"[CLASSIFIER] Failed to initialize: {str(e)}", file=sys.stderr)
                self.tool_classifier = None

    def reset(self):
        pass

    def gt_response(self, prompt):
        if 'CIReAct' in str(self.agent.__class__):
            gold = prompt
            prompt = f"""{self.agent._protocol.action['begin']} IPythonInterpreter
{self.agent._protocol.action_input['begin']} ```python\n{gold}\n```\n"""  # noqa
            action_input = dict(
                command=f"""```python\n{gold}\n```\n""",
                timeout=120,
            )
            response = self.agent._action_executor('IPythonInterpreter',
                                                   action_input)
            gt_response = dict(role='assistant', content=prompt)
            system_response = dict(
                role='system',
                content=self.agent._protocol.format_response(response))
            return [gt_response, system_response]
        else:
            gt_response = dict(role='assistant', content=prompt)
            return [gt_response]

    @property
    def template_parser(self):
        return self.agent._llm.template_parser

    @template_parser.setter
    def template_parser(self, value):
        self.agent._llm.template_parser = value

    def next_step(self, history, resources=None, stop=False):
        from lagent.schema import ActionReturn
        import sys
        tools = []
        files = []
        if resources is not None:
            tools = [
                self.tools[item['name']] for item in resources
                if item['type'] == 'tool'
            ]
            files = [item for item in resources if item['type'] == 'file']

        action_executor = lagent.ActionExecutor(actions=tools)
        if stop:
            history = history + [{'role': 'user', 'content': 'Please provide your final answer now. Use this EXACT format:\nFinal Answer: <your answer here>\n\nDo not call any tools.'}]
        
        # Extract user context BEFORE converting to React style (for classifier)
        original_user_context = ""
        for item in history:
            if isinstance(item, dict) and item.get('role') == 'user':
                original_user_context = item.get('content', '')
                break
        
        # Build history string from actual history for classifier (BEFORE react conversion)
        # Format: [TOOL_CALL] name args -> result
        original_history_parts = []
        i = 0
        while i < len(history):
            item = history[i]
            if isinstance(item, dict):
                role = item.get('role')
                # Look for assistant tool calls
                if role == 'assistant' and item.get('tool_calls'):
                    tool_calls = item.get('tool_calls', [])
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict) and tool_call.get('type') == 'function':
                            func = tool_call.get('function', {})
                            name = func.get('name', '')
                            args = func.get('arguments', {})
                            try:
                                args_str = json.dumps(args, separators=(",", ":")) if isinstance(args, (dict, list)) else str(args)
                            except Exception:
                                args_str = str(args) if args else '{}'
                            
                            # Look for corresponding tool response in next item
                            result = ''
                            if i + 1 < len(history):
                                next_item = history[i + 1]
                                if isinstance(next_item, dict) and next_item.get('role') in ('tool', 'function'):
                                    result = next_item.get('result', '') or next_item.get('output', '') or next_item.get('content', '')
                                    # If result is a dict with 'content' key, extract it
                                    if isinstance(result, dict) and 'content' in result:
                                        result = result['content']
                                    # Convert to string and truncate if too long
                                    result = str(result) if result else ''
                                    if len(result) > 200:
                                        result = result[:200] + '...'
                            
                            original_history_parts.append(f"[TOOL_CALL] {name} {args_str} -> {result}")
            i += 1
        
        original_history_str = " ".join(original_history_parts)
        # Truncate if too long (but use a generous limit - tokenizer max_length is 512)
        if len(original_history_str) > 2000:
            original_history_str = original_history_str[:2000] + "..."
        
        # Token counting helper
        def estimate_tokens(text: str) -> int:
            """Estimate token count. ~4 chars = 1 token for English/code."""
            return max(1, len(text) // 4)
        
        # HISTORY STEP EXPLANATION:
        # A "step" in history is one item with a role (user/assistant/tool):
        # - User step: User's question/instruction
        # - Assistant step: LLM's thought + tool call OR final answer
        # - Tool step: Tool result/response
        # So a typical turn = user step + assistant step + tool step = 3 steps
        
        # CONTEXT MANAGEMENT STRATEGY:
        # System prompt + template overhead: ~1250 tokens (fixed)
        # Generation budget: 2048 tokens (max_tokens in request)
        # Available for history: 32768 - 1250 - 2048 - 512 = 28958 tokens
        # Each history step averages: ~300-400 tokens
        # This gives us: 28958 / 350 = ~82+ steps we can safely fit
        
        SESSION_TOKENS = 32768  # LMDeploy session limit (extended)
        RESERVED_GEN_TOKENS = 2048  # For response generation
        RESERVED_BUFFER = 512  # Safety margin
        SYSTEM_OVERHEAD = 1250  # System prompt + tool descriptions (measured)
        MAX_PROMPT_TOKENS = SESSION_TOKENS - RESERVED_GEN_TOKENS - RESERVED_BUFFER - SYSTEM_OVERHEAD  # ~28958 tokens
        
        # Start with a generous step limit - can handle ~10 full turns + retries
        MAX_HISTORY_STEPS = 35  # Allow up to 35 steps (~11 full turns with retries)
        
        # Truncate by step count first
        initial_history_len = len(history)
        if len(history) > MAX_HISTORY_STEPS:
            history = [history[0]] + history[-(MAX_HISTORY_STEPS-1):]
            print(f"[HISTORY_LIMIT] Truncated from {initial_history_len} to {MAX_HISTORY_STEPS} steps (step limit)", file=sys.stderr)
        
        # Convert to react style (this adds the system prompt + formatting)
        history = react_style_history(history, files, self.agent._protocol)
        
        max_retry_attempts = 5
        retry_count = 0
        retry_history = list(history)
        action: ActionReturn | None = None
        last_raw_response = None
        while retry_count < max_retry_attempts:
            prompt = self.agent._protocol.format(chat_history=[],
                                                inner_step=history,
                                                action_executor=action_executor)
            
            # Analyze prompt size
            prompt_str = prompt if isinstance(prompt, str) else str(prompt)
            prompt_chars = len(prompt_str)
            prompt_tokens = estimate_tokens(prompt_str)
            
            # print(f"[PROMPT_INFO] Chars: {prompt_chars}, Tokens: {prompt_tokens}/{MAX_PROMPT_TOKENS}, History: {len(history)} items")
            
            # Check if approaching token limit, rarely the case
            if prompt_tokens > MAX_PROMPT_TOKENS:
                print(f"[TOKEN_LIMIT] Prompt exceeds limit ({prompt_tokens} > {MAX_PROMPT_TOKENS}). Pruning history...", file=sys.stderr)
                if len(history) > 4:
                    keep_count = max(4, len(history) // 2)
                    history = [history[0]] + history[-keep_count:]
                    print(f"[TOKEN_REDUCE] Moderate prune: kept first + last {keep_count} items (~50% retained)", file=sys.stderr)
                    prompt = self.agent._protocol.format(chat_history=[],
                                                        inner_step=history,
                                                        action_executor=action_executor)
                    prompt_str = prompt if isinstance(prompt, str) else str(prompt)
                    prompt_tokens = estimate_tokens(prompt_str)  # Pass string, not int
                    print(f"[PROMPT_INFO] After prune: Tokens: {prompt_tokens}/{MAX_PROMPT_TOKENS}", file=sys.stderr)
                else:
                    print(f"[TOKEN_REDUCE] Cannot prune further (only {len(history)} items remain)", file=sys.stderr)
            
            # # Log LLM generation parameters if available
            # try:
            #     llm_config = getattr(self.agent._llm, 'model_kwargs', {})
            #     max_tokens = llm_config.get('max_new_tokens') or llm_config.get('max_length')
            #     print(f"[GEN_PARAMS] max_new_tokens={max_tokens}, top_p={llm_config.get('top_p')}, temperature={llm_config.get('temperature')}", file=sys.stderr)
            # except Exception:
            #     pass
            
            response = self.agent._llm.chat(prompt)
            # print(f"[RESPONSE_INFO] Got {len(response)} chars, Empty: {response == ''}", file=sys.stderr)
            
            if response == '' and retry_count > 0:
                attempt = retry_count  # capture current retry index before mutating
                print(f"[EMPTY_AFTER_RETRY] Backend returned empty after retry {attempt}. Simplifying prompt...", file=sys.stderr)
                
                # Strategy: progressively simplify the prompt to break the refusal
                if attempt == 1:
                    retry_count += 1  # advance so the next empty goes to the terminal branch
                    print(f"[SIMPLIFY_STRATEGY] Retry 1: Pruning history, removing errors, simplifying instructions", file=sys.stderr)
                    
                    cleaned_history = [
                        step for step in retry_history 
                        if not (step.get('role') == 'system' and 'ERROR' in str(step.get('content', '')))
                    ]
                    
                    if len(cleaned_history) > 2:
                        cleaned_history = [cleaned_history[0]] + cleaned_history[-1:]
                    
                    simple_instruction = """You must respond in one of these formats:
1) Tool call: Action: <ToolName>\nAction Input: <JSON>
2) Final answer: Final Answer: <answer>

Choose the appropriate tool or provide your final answer now."""
                    
                    original_protocol = self.agent._protocol.call_protocol
                    self.agent._protocol.call_protocol = simple_instruction
                    
                    history = react_style_history(cleaned_history, files, self.agent._protocol)
                    retry_history = cleaned_history  # Update retry_history to cleaned version
                    
                    # Restore original protocol after formatting
                    self.agent._protocol.call_protocol = original_protocol
                    
                    continue  # Retry with simplified prompt
                
                elif attempt >= 2:
                    print(f"[EMPTY_AFTER_RETRY_{attempt}] Backend returned empty after {attempt} retries. Giving up.", file=sys.stderr)
                    msg = {'role': 'assistant'}
                    msg['error'] = dict(
                        type='EMPTY_RESPONSE',
                        msg='Model returned empty response after retry attempts with progressive simplification'
                    )
                    msg['tool_calls'] = [dict(type='function', function=dict(name='NoAction', arguments={'text': ""}))]  # noqa
                    msg['raw_response'] = last_raw_response or ""
                    return msg
            last_raw_response = response
            import sys
            thought, action_name, action_input = self.agent._protocol.parse(
                response, action_executor)
            print(f"[THOUGHT_EXTRACTED] Length: {len(thought or '')}, Full: {repr((thought or '')[:300])}", file=sys.stderr)
            if action_name == "FinishAction":
                ok = True
                coerced = action_input
            else:    
                mismatch, announced = _detect_tool_announcement_mismatch(thought, action_name)
                if mismatch:
                    print(f"[TOOL_MISMATCH] Announced '{announced}' but called '{action_name}'", file=sys.stderr)
                
                # Parse and sanitize arguments first
                try:
                    sanitized_args = _parse_and_sanitize_args(action_input, _expected_keys_for(action_name))
                except Exception as _e:
                    ok, reason, coerced = False, f"Argument parsing error: {str(_e)}", action_input
                    sanitized_args = action_input
                
                # STEP 1: Validate tool choice with classifier (before constraint validation)
                # This checks if the tool choice is semantically correct for the task
                # Skip classifier for FinishAction
                if self.tool_classifier:
                    # Get tool description
                    tool_desc = ""
                    if action_name in action_executor.actions:
                        tool_obj = action_executor.actions[action_name]
                        tool_desc = getattr(tool_obj, 'description', None)
                        if not tool_desc and hasattr(tool_obj, 'toolmeta'):
                            toolmeta = tool_obj.toolmeta
                            if isinstance(toolmeta, dict):
                                tool_desc = toolmeta.get('description', '')
                            else:
                                tool_desc = getattr(toolmeta, 'description', '')
                    
                    # Ensure tool_desc is a string
                    if isinstance(tool_desc, dict):
                        tool_desc = tool_desc.get('description', str(tool_desc))
                    tool_desc = str(tool_desc) if tool_desc else action_name
                    
                    # Extract argument schema from sanitized_args (parameter names with masked values)
                    argument_schema = None
                    if isinstance(sanitized_args, dict):
                        argument_schema = {k: '<MASKED>' for k in sanitized_args.keys()}
                    
                    # Validate with classifier (using pre-built history from original history)
                    classifier_valid, classifier_reason, classifier_conf = _validate_tool_choice_with_classifier(
                        context=original_user_context,
                        history=original_history_str,
                        tool_name=action_name,
                        tool_description=tool_desc,
                        argument_schema=argument_schema,
                        classifier=self.tool_classifier,
                        confidence_threshold=self.classifier_confidence_threshold
                    )
                    
                    print(f"[CLASSIFIER] {classifier_reason} (confidence: {classifier_conf:.3f})", file=sys.stderr)
                    
                    if not classifier_valid:
                        retry_count += 1
                        if retry_count >= max_retry_attempts:
                            msg = {'role': 'assistant'}
                            if thought:
                                msg['thought'] = thought
                            msg['error'] = dict(
                                type='CLASSIFIER_REJECTION',
                                msg=f"Tool choice '{action_name}' rejected by validator after {max_retry_attempts} attempts"
                            )
                            msg['tool_calls'] = [dict(type='function', function=dict(name='NoAction', arguments={}))]
                            msg['raw_response'] = last_raw_response
                            return msg
                        
                        error_msg = {
                            'role': 'system',
                            'content': f"ERROR: {classifier_reason}\n\nPlease reconsider your tool choice and try a different tool that better matches the task requirements."
                        }
                        retry_history.append(error_msg)
                        history = react_style_history(retry_history, files, self.agent._protocol)
                        continue
                
                # STEP 2: Validate argument constraints (format, required fields, etc.)
                try:
                    ok, reason, coerced = _validate_constraints(action_name, sanitized_args)
                except Exception as _e:
                    ok, reason, coerced = False, f"Validation pipeline error: {str(_e)}", sanitized_args

            if not ok:
                changed, corrected, note = _auto_correct_args(action_name, sanitized_args, reason)
                if changed:
                    ok2, reason2, coerced2 = _validate_constraints(action_name, corrected)
                    if ok2:
                        coerced = coerced2
                        ok = True
                    else:
                        reason = reason2
                        coerced = corrected

            if ok:
                action = action_executor(action_name, coerced)
                if action.type == action_executor.finish_action.name:
                    return dict(role='assistant', content=action.format_result())
                if action.errmsg is not None:
                    retry_count += 1
                    if retry_count >= max_retry_attempts:
                        msg = {'role': 'assistant'}
                        if thought:
                            msg['thought'] = thought
                        msg = {'role': 'assistant'}
                        msg['error'] = dict(
                            type='EMPTY_RESPONSE',
                            msg='Model returned empty response after retry attempts with progressive simplification'
                        )
                        msg['tool_calls'] = [dict(type='function', function=dict(name='NoAction', arguments={'text': ""}))]  # noqa
                        msg['raw_response'] = last_raw_response or ""
                        return msg
                    error_msg = {
                        'role': 'system',
                        'content': f"ERROR: Tool {action_name} returned error: {action.errmsg}\nPlease fix the arguments and try again."
                    }
                    retry_history.append(error_msg)
                    history = react_style_history(retry_history, files, self.agent._protocol)
                    continue

                msg = {'role': 'assistant'}
                if thought:
                    msg['thought'] = thought
                args = action.args
                function = dict(name=action.type, arguments=args)
                msg['tool_calls'] = [dict(type='function', function=function)]
                return msg

            # Validation failed path
            retry_count += 1
            if retry_count >= max_retry_attempts:
                msg = {'role': 'assistant'}
                if thought:
                    msg['thought'] = thought
                error_msg_text = f"Constraint violation for {action_name} after {max_retry_attempts} attempts: {reason}"
                msg['content'] = error_msg_text
                function = dict(name='NoAction', arguments={})
                msg['tool_calls'] = [dict(type='function', function=function)]
                msg['raw_response'] = last_raw_response
                return msg

            # Re-prompt with validation feedback
            print(f"[RETRY] Attempt {retry_count}/{max_retry_attempts} for {action_name}: {reason}", file=sys.stderr)
            guidance = {}
            if action_name == "CountGivenObject":
                guidance['bbox_help'] = "Field 'bbox' is OPTIONAL. If not needed, OMIT IT ENTIRELY from the JSON object—do NOT include it with null, empty string, or placeholder values. Correct: {\"image\": \"...\", \"text\": \"...\"}. Wrong: {\"image\": \"...\", \"text\": \"...\", \"bbox\": null} or {\"bbox\": \"\"}. If you do provide bbox, use '(x1, y1, x2, y2)' format with 4 numbers."
            elif action_name == "RegionAttributeDescription":
                guidance['bbox_help'] = "Field 'bbox' is REQUIRED. Use format '(x1, y1, x2, y2)' or [x1, y1, x2, y2] with 4 numbers (integers or floats)."
            elif action_name == "DrawBox":
                guidance['bbox_help'] = "Field 'bbox' is REQUIRED. Use format '(x1, y1, x2, y2)' or [x1, y1, x2, y2] with 4 numbers."
            bbox_guidance = guidance.get('bbox_help', "- For bbox fields, use format '(x1, y1, x2, y2)' or [x1, y1, x2, y2] with 4 numbers. If bbox is optional and not needed, omit the field entirely.")
            error_msg = {
                'role': 'system',
                'content': f"ERROR: The tool call to {action_name} failed validation: {reason}\n\nPlease correct the arguments and try again. Remember:\n- Use the exact field names from the tool schema\n- Provide all required fields; omit optional fields if not applicable\n- Ensure correct data types (strings, numbers, booleans)\n- {bbox_guidance}\n- Final answer must be concise (<=2 sentences) and include the requested number(s) with units or an explicit yes/no. Do not recap history."
            }
            retry_history.append(error_msg)
            history = react_style_history(retry_history, files, self.agent._protocol)

        # Safety fallback
        print(f"[WARNING] action is None at final check; returning finish response", file=sys.stderr)
        return dict(role='assistant', content="I cannot determine the answer from the given information.")

    def chat(self, query, memory=None, resources=None):
        tools = []
        files = []
        if resources is not None:
            tools = {
                item['name']: self.tools[item['name']]
                for item in resources if item['type'] == 'tool'
            }
            files = [item for item in resources if item['type'] == 'file']

        action_executor = self.agent._action_executor
        action_executor.actions = tools
        if memory is None:
            memory = []
            if files:
                prompt = 'The related files are at ' + ', '.join(
                    f'`{file["path"]}`' for file in files)
                memory.append(dict(role='user', content=prompt))
        memory.append(dict(role='user', content=query))

        agent_return = self.agent.chat(memory)

        steps = []
        for action in agent_return.actions:
            if action.type == action_executor.finish_action.name:
                step = dict(role='assistant', content=action.format_result())
                steps.append(step)
                print(steps, memory)
            else:
                step = {'role': 'assistant'}
                args = action.args

                if action.errmsg is not None:
                    step['error'] = dict(type=action.state.name,
                                         msg=action.errmsg)
                    # Handle fallback args
                    args = args.get('inputs', args)
                function = dict(name=action.type, arguments=args)
                step['tool_calls'] = [dict(type='function', function=function)]
                steps.append(step)
                steps.append(dict(role='tool', content=action.result))
        return steps, memory

FORCE_STOP_PROMPT_EN = (
    """You should directly give results based on history information."""  # noqa
)

FEWSHOT_INSTRUCTION = """\
You are a assistant who can utilize external tools.
{tool_description}
To use a tool, please response with the following format:
```
{thought} Think what you need to solve, do you need to use tools?
{action} The tool name, should be one of [{action_names}].
{action_input} The input to the tool that you want to use.
```
The tool will give you response after your response using the following format:
```
{response} the results after call the tool.
```
Therefore DO NOT generate tool response by yourself.

You must respond in exactly one of these two formats:
1) Tool call:
    Action: <ToolName>
    Action Input: <JSON args>
    (Tool will then return Response: …; do NOT fabricate Response yourself.)
2) Finish:
    Final Answer: <your answer>

Also please follow the guidelines:
1. Always use code interpreter to solve the problem.
2. The generated codes should always in a markdown code block format.
3. The generated codes will be executed in an ipython manner and the results will be cached.
4. Your responded code should always be simple and only solves the problem in current step.
5. Single-step discipline: In each step, think and decide on ONE tool only. Do not plan or mention using subsequent tools in the same Thought; focus on the current step.
6. Consistency: The tool name in {action} MUST exactly match the one you mentioned in {thought} for the current step.
7. Clean inputs: {action_input} must contain ONLY the JSON arguments for the CURRENT tool. Do not include any previous "Thought:", "Action:", or "Response:" blocks, nor results from earlier tools.
8. No chaining inside arguments: Never embed the output of another tool directly inside {action_input}. Use separate steps to call subsequent tools.
9. To finish and provide a final answer (not call a tool), you MUST use the Finish action explicitly. You cannot just stop—always choose either a tool call or the Finish action.
10. Keep the final answer concise to <= 2 sentences; do NOT restate tool outputs or recap history. Final answer must be a proper sentence. Do not respond with only a bare number.
11. CRITICAL: For optional fields (like 'bbox' in CountGivenObject), you MUST OMIT them entirely from the JSON if not needed. Do NOT include them with null, empty string "", or placeholder values like "unknown". Omitting means: {{"field1": "value1", "field2": "value2"}} without the optional field. An omitted field is different from null or empty.
12. MANDATORY: You must always respond with either a tool call OR the Finish action. NEVER call or mention NoAction.
13. CRITICAL: If you cannot call a tool (missing required arguments, uncertain about tool selection), use the Finish action with an explanation rather than responding with plain text or NoAction.

Begin!
"""  # noqa

PYTHON_INTERPRETER_DESCRIPTION = """\
It can run a Python code. The code must be a valid code that contains only python method, and the method' name must be 'solution' and returns a dict, which key is variable name. The libraries I recommend are sympy and scipy. the format is:
```python
# import packages
import xxx
def solution():
    # initialize some variables
    variable_names_with_real_meaning = xxx
    # middle steps
    mid_variable = func(mid_variable)
    # final answer
    final_answer = func(mid_variable)
    return final_answer
```"""  # noqa


class CodeAgent(LagentAgent):
    """Code Agent wrapper for Lagent."""

    def __init__(self, llm, **kwargs):
        from lagent import PythonInterpreter, ReAct
        from lagent.agents.react import ReActProtocol

        agent_type = kwargs.pop('agent_type', ReAct)
        max_turn = kwargs.pop('max_turn', 4)
        actions = kwargs.pop(
            'actions',
            [
                # dict(type=PythonInterpreter,
                #      description=PYTHON_INTERPRETER_DESCRIPTION),
            ],
        )
        protocol = kwargs.pop(
            'protocol',
            dict(
                type=ReActProtocol,
                call_protocol=FEWSHOT_INSTRUCTION,
                force_stop=FORCE_STOP_PROMPT_EN,
                finish=dict(role='FINISH', begin='Final Answer:', end='\n'),
            ),
        )
        super().__init__(agent_type=agent_type,
                         llm=llm,
                         actions=actions,
                         protocol=protocol,
                         max_turn=max_turn,
                         **kwargs)
