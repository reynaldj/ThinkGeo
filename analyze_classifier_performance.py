#!/usr/bin/env python3
"""Analyze classifier performance: correct rejections, false positives/negatives."""

import json
import os
from collections import defaultdict
from pathlib import Path

def analyze_predictions(predictions_dir):
    """Analyze tool call correctness vs classifier decisions."""
    
    results = {
        'correct_rejections': [],      # Gold != Pred + Classifier rejected
        'false_positives': [],         # Gold != Pred but Classifier allowed
        'false_negatives': [],         # Gold == Pred but Classifier rejected
        'correct_allowances': [],      # Gold == Pred + Classifier allowed
        'tool_stats': defaultdict(lambda: {
            'correct_rejections': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'correct_allowances': 0
        })
    }
    
    pred_files = sorted(Path(predictions_dir).glob('*.json'))
    total_samples = 0
    
    for pred_file in pred_files:
        with open(pred_file, 'r') as f:
            data = json.load(f)
        
        for sample_id, sample_data in data.items():
            try:
                gold = sample_data.get('gold', [])
                prediction = sample_data.get('prediction', [])
                
                if not gold or not prediction:
                    continue
                
                total_samples += 1
                
                # Extract gold tool calls (assistant with tool_calls)
                gold_tools = []
                for step in gold:
                    if step.get('role') == 'assistant' and step.get('tool_calls'):
                        for tc in step['tool_calls']:
                            if tc.get('type') == 'function':
                                gold_tools.append({
                                    'name': tc['function'].get('name'),
                                    'arguments': tc['function'].get('arguments', {})
                                })
                
                # Extract prediction tool calls and their classifier status
                pred_tools = []
                for step in prediction:
                    if step.get('role') == 'assistant':
                        # Check if step has error (classifier rejection)
                        has_error = 'error' in step
                        is_classifier_rejection = (
                            has_error and 
                            step.get('error', {}).get('type') == 'CLASSIFIER_REJECTION'
                        )
                        
                        # Extract actual tool call attempted
                        if step.get('tool_calls'):
                            for tc in step['tool_calls']:
                                if tc.get('type') == 'function':
                                    tool_name = tc['function'].get('name')
                                    pred_tools.append({
                                        'name': tool_name,
                                        'arguments': tc['function'].get('arguments', {}),
                                        'rejected': is_classifier_rejection
                                    })
                
                # Compare: match by position in sequence
                for idx, (gold_tool, pred_tool) in enumerate(zip(gold_tools, pred_tools)):
                    tool_name = pred_tool['name']
                    is_correct_tool = (gold_tool['name'] == pred_tool['name'] and 
                                     gold_tool['arguments'] == pred_tool['arguments'])
                    is_rejected = pred_tool['rejected']
                    
                    # Categorize
                    if is_correct_tool:
                        if is_rejected:
                            # Classifier rejected a correct tool - FALSE NEGATIVE
                            results['false_negatives'].append({
                                'sample_id': sample_id,
                                'step': idx,
                                'tool': tool_name
                            })
                            results['tool_stats'][tool_name]['false_negatives'] += 1
                        else:
                            # Classifier allowed a correct tool - CORRECT ALLOWANCE
                            results['correct_allowances'].append({
                                'sample_id': sample_id,
                                'step': idx,
                                'tool': tool_name
                            })
                            results['tool_stats'][tool_name]['correct_allowances'] += 1
                    else:
                        if is_rejected:
                            # Classifier rejected an incorrect tool - CORRECT REJECTION
                            results['correct_rejections'].append({
                                'sample_id': sample_id,
                                'step': idx,
                                'tool': tool_name,
                                'expected': gold_tool['name']
                            })
                            results['tool_stats'][tool_name]['correct_rejections'] += 1
                        else:
                            # Classifier allowed an incorrect tool - FALSE POSITIVE
                            results['false_positives'].append({
                                'sample_id': sample_id,
                                'step': idx,
                                'tool': tool_name,
                                'expected': gold_tool['name']
                            })
                            results['tool_stats'][tool_name]['false_positives'] += 1
            
            except Exception as e:
                print(f"Error processing sample {sample_id}: {e}")
                continue
    
    return results, total_samples


def print_analysis(results, total_samples):
    """Print formatted analysis."""
    
    cr = len(results['correct_rejections'])
    fp = len(results['false_positives'])
    fn = len(results['false_negatives'])
    ca = len(results['correct_allowances'])
    total_decisions = cr + fp + fn + ca
    
    print("=" * 80)
    print("CLASSIFIER PERFORMANCE ANALYSIS")
    print("=" * 80)
    print(f"\nTotal samples analyzed: {total_samples}")
    print(f"Total tool call decisions: {total_decisions}\n")
    
    print("DECISION CATEGORIES:")
    print(f"  ✓ Correct Rejections (Gold≠Pred, Rejected):  {cr:3d} ({100*cr/total_decisions:.1f}%)")
    print(f"  ✗ False Positives (Gold≠Pred, Allowed):     {fp:3d} ({100*fp/total_decisions:.1f}%)")
    print(f"  ✗ False Negatives (Gold=Pred, Rejected):    {fn:3d} ({100*fn/total_decisions:.1f}%)")
    print(f"  ✓ Correct Allowances (Gold=Pred, Allowed):  {ca:3d} ({100*ca/total_decisions:.1f}%)")
    print(f"  {'─' * 50}")
    print(f"  Total Correct Decisions:                    {cr+ca:3d} ({100*(cr+ca)/total_decisions:.1f}%)")
    print(f"  Total Incorrect Decisions:                  {fp+fn:3d} ({100*(fp+fn)/total_decisions:.1f}%)")
    
    print("\n" + "=" * 80)
    print("PER-TOOL BREAKDOWN")
    print("=" * 80)
    print(f"{'Tool':<25} {'Correct':>10} {'False Pos':>10} {'False Neg':>10} {'Correct Allow':>13}")
    print("─" * 70)
    
    for tool_name in sorted(results['tool_stats'].keys()):
        stats = results['tool_stats'][tool_name]
        cr_tool = stats['correct_rejections']
        fp_tool = stats['false_positives']
        fn_tool = stats['false_negatives']
        ca_tool = stats['correct_allowances']
        
        print(f"{tool_name:<25} {cr_tool:>10} {fp_tool:>10} {fn_tool:>10} {ca_tool:>13}")
    
    print("\n" + "=" * 80)
    print("FALSE POSITIVES (Incorrect tools allowed)")
    print("=" * 80)
    if results['false_positives']:
        tool_fp = defaultdict(int)
        for fp in results['false_positives']:
            tool_fp[fp['tool']] += 1
        
        print(f"{'Tool Called':<20} {'Expected':<20} {'Count':>10}")
        print("─" * 50)
        for fp in results['false_positives'][:20]:  # Show first 20
            print(f"{fp['tool']:<20} {fp['expected']:<20} (sample: {fp['sample_id']})")
    else:
        print("None!")
    
    print("\n" + "=" * 80)
    print("FALSE NEGATIVES (Correct tools rejected)")
    print("=" * 80)
    if results['false_negatives']:
        print(f"{'Tool Rejected':<25} {'Sample ID':<15} {'Reason':>20}")
        print("─" * 60)
        for fn in results['false_negatives'][:20]:  # Show first 20
            print(f"{fn['tool']:<25} {fn['sample_id']:<15} (step {fn['step']})")
    else:
        print("None!")


if __name__ == '__main__':
    import sys
    
    pred_dir = sys.argv[1] if len(sys.argv) > 1 else '/home/james/ThinkGeo/opencompass/outputs/default/20260203_133155/predictions/qwen2.5-7b-instruct'
    
    print(f"Analyzing predictions from: {pred_dir}\n")
    results, total = analyze_predictions(pred_dir)
    print_analysis(results, total)
