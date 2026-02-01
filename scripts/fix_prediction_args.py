#!/usr/bin/env python3
"""
Fix common malformed tool argument formats in OpenCompass prediction JSONs.

Usage:
  python scripts/fix_prediction_args.py --pred-dir PATH [--inplace]

This script scans all .json files in the given predictions directory, attempts
to parse/normalize any `tool_calls` -> `function` -> `arguments` fields that
are strings or malformed, and writes a fixed copy with suffix `_fixed.json`
unless `--inplace` is provided.
"""
import argparse
import json
import os
import re
import ast
from typing import Any


def strip_code_fences(s: str) -> str:
    s = re.sub(r'```[a-zA-Z0-9]*', '', s)
    s = s.replace('```', '')
    # Remove common prefixes
    s = re.sub(r'^(Response|Action Input|Action|Input):\s*', '', s, flags=re.IGNORECASE)
    return s.strip()


def try_parse_json_from_string(s: str) -> Any:
    s0 = strip_code_fences(s)
    # Try to extract a JSON object substring
    m = re.search(r'\{[\s\S]*\}', s0)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            # try removing comments and using ast.literal_eval
            try:
                return ast.literal_eval(candidate)
            except Exception:
                pass

    # Try entire string as JSON
    try:
        return json.loads(s0)
    except Exception:
        pass

    # Try ast literal eval on the whole string
    try:
        return ast.literal_eval(s0)
    except Exception:
        pass

    # Fallback: extract numbers (bbox-like)
    nums = re.findall(r"-?\d+\.?\d*", s0)
    if len(nums) >= 4:
        # if multiple of 4, return list of bboxes
        vals = [float(x) for x in nums]
        if len(vals) % 4 == 0 and len(vals) > 4:
            bboxes = [vals[i:i+4] for i in range(0, len(vals), 4)]
            return {'bbox': bboxes}
        else:
            return {'bbox': vals[:4]}

    return None


def normalize_arguments(args: Any) -> Any:
    # If already a mapping, try to normalize bbox strings inside
    if isinstance(args, dict):
        out = {}
        for k, v in args.items():
            if isinstance(v, str) and k.lower() == 'bbox':
                parsed = try_parse_json_from_string(v)
                out[k] = parsed if parsed is not None else v
            else:
                out[k] = v
        return out

    if isinstance(args, str):
        parsed = try_parse_json_from_string(args)
        return parsed if parsed is not None else args

    return args


def process_file(path: str, inplace: bool = False) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    fixed = 0
    total = 0
    # data is mapping of example id -> {prediction: [...], ...}
    for ex_id, ex in list(data.items()):
        preds = ex.get('prediction', [])
        for pred in preds:
            if not isinstance(pred, dict):
                continue
            tool_calls = pred.get('tool_calls') or pred.get('tool_call') or []
            # some dumps use 'tool_calls' as list of dicts
            if isinstance(tool_calls, dict):
                tool_calls = [tool_calls]
            for tc in tool_calls:
                func = tc.get('function') if isinstance(tc, dict) else None
                if not func:
                    continue
                total += 1
                args = func.get('arguments')
                if args is None:
                    continue
                if isinstance(args, (dict, list)):
                    # if dict but bbox inside is string, normalize
                    new_args = normalize_arguments(args)
                    if new_args != args:
                        func['arguments'] = new_args
                        fixed += 1
                else:
                    # args is probably a string
                    new_args = normalize_arguments(args)
                    if new_args is not None and new_args != args:
                        func['arguments'] = new_args
                        fixed += 1

    out_path = path if inplace else path.replace('.json', '_fixed.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return {'file': path, 'total_tool_calls_seen': total, 'fixed_args': fixed, 'out_path': out_path}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-dir', required=True, help='Predictions directory with .json files')
    parser.add_argument('--inplace', action='store_true', help='Overwrite original files')
    args = parser.parse_args()

    results = []
    for fname in os.listdir(args.pred_dir):
        if not fname.endswith('.json'):
            continue
        path = os.path.join(args.pred_dir, fname)
        print('Processing', path)
        res = process_file(path, inplace=args.inplace)
        print('  fixed', res['fixed_args'], 'of', res['total_tool_calls_seen'], 'tool-call args ->', res['out_path'])
        results.append(res)

    # summary
    total_fixed = sum(r['fixed_args'] for r in results)
    total_seen = sum(r['total_tool_calls_seen'] for r in results)
    print('\nSummary: fixed', total_fixed, 'of', total_seen, 'tool-call argument entries')


if __name__ == '__main__':
    main()
