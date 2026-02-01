#!/usr/bin/env python3
"""
Remove ARGS_REPAIRED and ARGS_ERROR `error` objects from prediction entries.

This preserves other error types (e.g., API_ERROR). The script edits files
in-place and prints a summary of removals per file.
"""
import os
import json
import argparse


def process_file(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    removed = 0
    total = 0
    for ex_id, ex in list(data.items()):
        preds = ex.get('prediction', [])
        for pred in preds:
            if not isinstance(pred, dict):
                continue
            err = pred.get('error')
            if not err:
                continue
            total += 1
            etype = err.get('type')
            if etype in ('ARGS_REPAIRED', 'ARGS_ERROR'):
                # Remove the error object entirely
                pred.pop('error', None)
                removed += 1

    if removed > 0:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    return {'file': path, 'found_errors': total, 'removed_errors': removed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-dir', required=True, help='Predictions directory')
    args = parser.parse_args()

    results = []
    for root, _, files in os.walk(args.pred_dir):
        for fname in files:
            if not fname.endswith('.json'):
                continue
            path = os.path.join(root, fname)
            print('Processing', path)
            res = process_file(path)
            print('  removed', res['removed_errors'], 'of', res['found_errors'], 'error objects')
            results.append(res)

    total_removed = sum(r['removed_errors'] for r in results)
    total_found = sum(r['found_errors'] for r in results)
    print('\nSummary: removed', total_removed, 'of', total_found, 'ARGS_REPAIRED/ARGS_ERROR entries')


if __name__ == '__main__':
    main()
