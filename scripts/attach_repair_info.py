#!/usr/bin/env python3
"""
Attach repair information to ARGS_ERROR entries in prediction JSONs.

For any prediction entry that has an `error` with `type` == 'ARGS_ERROR',
if the corresponding `tool_calls` -> `function` -> `arguments` has been
normalized (i.e., is a dict/list rather than a raw string), this script will
replace the `error` object with a new object containing:

- `original_type`: previous error type
- `original_msg`: previous error message (trimmed)
- `type`: 'ARGS_REPAIRED'
- `msg`: short note
- `repaired_arguments`: the normalized arguments (for traceability)

The script edits files in-place and prints a summary of updates.
"""
import json
import os
import argparse
from typing import Any


def process_file(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    updated = 0
    total_errors = 0
    # iterate over examples
    for ex_id, ex in list(data.items()):
        preds = ex.get('prediction', [])
        for pred in preds:
            if not isinstance(pred, dict):
                continue
            err = pred.get('error')
            if not err or err.get('type') != 'ARGS_ERROR':
                continue
            total_errors += 1

            # Find first tool_call with parsed arguments
            tool_calls = pred.get('tool_calls') or pred.get('tool_call') or []
            if isinstance(tool_calls, dict):
                tool_calls = [tool_calls]

            repaired_args = None
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                func = tc.get('function')
                if not isinstance(func, dict):
                    continue
                args = func.get('arguments')
                # If args is not a plain string, we likely repaired it
                if args is not None and not isinstance(args, str):
                    repaired_args = args
                    break

            if repaired_args is None:
                # No repaired args found; skip
                continue

            # Create new error object preserving original message
            orig_msg = err.get('msg')
            # trim very long messages
            if isinstance(orig_msg, str) and len(orig_msg) > 2000:
                orig_msg_short = orig_msg[:2000] + '...[truncated]'
            else:
                orig_msg_short = orig_msg

            new_err = {
                'original_type': err.get('type'),
                'original_msg': orig_msg_short,
                'type': 'ARGS_REPAIRED',
                'msg': 'arguments parsed and attached by post-processing canonicalizer',
                'repaired_arguments': repaired_args,
            }

            pred['error'] = new_err
            updated += 1

    if updated > 0:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    return {'file': path, 'total_errors_seen': total_errors, 'updated_errors': updated}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-dir', required=True, help='Predictions directory to process')
    args = parser.parse_args()

    results = []
    for root, _, files in os.walk(args.pred_dir):
        for fname in files:
            if not fname.endswith('.json'):
                continue
            path = os.path.join(root, fname)
            print('Processing', path)
            res = process_file(path)
            print('  updated', res['updated_errors'], 'of', res['total_errors_seen'], 'ARGS_ERROR entries')
            results.append(res)

    total_updated = sum(r['updated_errors'] for r in results)
    total_seen = sum(r['total_errors_seen'] for r in results)
    print('\nSummary: updated', total_updated, 'of', total_seen, 'ARGS_ERROR entries')


if __name__ == '__main__':
    main()
