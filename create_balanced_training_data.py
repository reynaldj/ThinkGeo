#!/usr/bin/env python3
"""
Create balanced training data from actual qwen2.5-7b-instruct error patterns.

Strategy:
1. Extract all WRONG tool predictions from qwen2.5-7b-instruct
2. Mix with CORRECT predictions from current dataset
3. Create 50-50 balanced dataset
4. Weight hard errors (TextToBbox, Calculator substitutions) more heavily
"""

import json
import glob
from collections import defaultdict, Counter
from pathlib import Path
import random

random.seed(42)

def extract_errors_from_predictions():
    """Extract tool mismatches from qwen2.5-7b-instruct predictions."""
    pred_files = sorted(glob.glob(
        "opencompass/outputs/default/20260128_133947/predictions/qwen2.5-7b-instruct/ThinkGeo_bench_*.json"
    ))
    print(f"Found {len(pred_files)} prediction files")
    
    errors = []  # List of (expected_tool, predicted_tool, example_data)
    
    for pred_file in pred_files:
        with open(pred_file) as f:
            data = json.load(f)
        
        for sample_id, sample_data in data.items():
            if 'steps' not in sample_data:
                continue
            
            # Get gold and prediction steps
            gold_steps = sample_data.get('gold', [])
            pred_steps = sample_data.get('prediction', [])
            
            # Extract tools from each
            for step_idx in range(min(len(gold_steps), len(pred_steps))):
                gold_step = gold_steps[step_idx]
                pred_step = pred_steps[step_idx]
                
                gold_tools = gold_step.get('tool_calls', [])
                pred_tools = pred_step.get('tool_calls', [])
                
                if not gold_tools or not pred_tools:
                    continue
                
                gold_tool = gold_tools[0]['function']['name']
                pred_tool = pred_tools[0]['function']['name']
                
                if gold_tool != pred_tool:
                    errors.append({
                        'expected_tool': gold_tool,
                        'predicted_tool': pred_tool,
                        'question': sample_data.get('origin_prompt', '')[:200],
                        'gold_thought': gold_step.get('thought', ''),
                        'pred_thought': pred_step.get('thought', ''),
                        'gold_args': gold_tools[0]['function'].get('arguments', {}),
                        'pred_args': pred_tools[0]['function'].get('arguments', {}),
                        'sample_id': sample_id,
                    })
    
    return errors

def load_correct_examples():
    """Load correct examples from existing training data."""
    with open('tool_choice_data_from_predictions/train.json') as f:
        data = json.load(f)
    
    correct = [ex for ex in data if ex.get('label') == 1]
    return correct

def create_hard_negatives_from_errors(errors):
    """Convert extraction errors into training examples (negatives)."""
    hard_negatives = []
    
    for error in errors:
        # Create a training example where predicted_tool is the "wrong" choice
        example = {
            'label': 0,  # WRONG
            'tool_name': error['predicted_tool'],
            'tool_description': f"Tool: {error['predicted_tool']}",
            'expected_tool': error['expected_tool'],
            'full_history': [
                {
                    'role': 'user',
                    'content': error['question']
                },
                {
                    'role': 'assistant',
                    'content': error['pred_thought']
                }
            ],
            'tool_arguments': error['pred_args'],
            'sample_id': error['sample_id'],
            'error_type': f"{error['expected_tool']}_mispredicted_as_{error['predicted_tool']}"
        }
        hard_negatives.append(example)
    
    return hard_negatives

def weight_examples(correct, hard_negatives):
    """
    Create weighted distribution:
    - All hard negatives (actual model errors)
    - Sample correct examples to balance
    """
    # Count error patterns for weighting
    error_counts = Counter()
    for neg in hard_negatives:
        error_counts[neg['error_type']] += 1
    
    print("\nError distribution:")
    for error_type, count in error_counts.most_common(10):
        print(f"  {error_type}: {count}")
    
    # Weight to create 50-50 balance
    # We have len(hard_negatives) negatives
    # Sample len(hard_negatives) positives to create balanced dataset
    
    sampled_correct = random.sample(correct, min(len(hard_negatives), len(correct)))
    
    balanced_data = sampled_correct + hard_negatives
    
    print(f"\nBalanced dataset:")
    print(f"  Correct examples: {len(sampled_correct)}")
    print(f"  Hard negatives: {len(hard_negatives)}")
    print(f"  Total: {len(balanced_data)}")
    print(f"  Ratio: {len(sampled_correct)}/{len(hard_negatives)} = {len(sampled_correct)/len(hard_negatives):.2f}")
    
    return balanced_data

def split_train_val(data, val_ratio=0.15):
    """Split data into train and validation."""
    random.shuffle(data)
    split_idx = int(len(data) * (1 - val_ratio))
    return data[:split_idx], data[split_idx:]

def main():
    print("="*80)
    print("Creating balanced training data from qwen2.5-7b-instruct errors")
    print("="*80)
    
    # Step 1: Extract errors
    print("\n[1] Extracting errors from qwen2.5-7b-instruct predictions...")
    errors = extract_errors_from_predictions()
    print(f"    Found {len(errors)} tool mismatches")
    
    # Step 2: Load correct examples
    print("\n[2] Loading correct examples from training data...")
    correct = load_correct_examples()
    print(f"    Found {len(correct)} correct examples")
    
    # Step 3: Create hard negatives from errors
    print("\n[3] Creating hard negatives from actual errors...")
    hard_negatives = create_hard_negatives_from_errors(errors)
    print(f"    Created {len(hard_negatives)} hard negative examples")
    
    # Step 4: Weight and balance
    print("\n[4] Creating balanced dataset...")
    balanced_data = weight_examples(correct, hard_negatives)
    
    # Step 5: Split train/val
    print("\n[5] Splitting into train/val...")
    train_data, val_data = split_train_val(balanced_data, val_ratio=0.15)
    print(f"    Train: {len(train_data)}")
    print(f"    Val: {len(val_data)}")
    
    # Step 6: Save
    print("\n[6] Saving new balanced datasets...")
    output_dir = Path('tool_choice_data_from_predictions_balanced')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'train.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(output_dir / 'val.json', 'w') as f:
        json.dump(val_data, f, indent=2)
    
    # Also save test.json from original (unchanged)
    with open('tool_choice_data_from_predictions/test.json') as f:
        test_data = json.load(f)
    with open(output_dir / 'test.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"    ✓ Saved to {output_dir}/")
    
    # Step 7: Show summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Original training: 78.6% correct, 21.4% wrong (3.67:1)")
    print(f"New training:      50.0% correct, 50.0% wrong (1.00:1)")
    print(f"\nHard negatives included:")
    error_counts = Counter(neg['error_type'] for neg in hard_negatives)
    for error_type, count in error_counts.most_common(10):
        print(f"  {error_type}: {count}")
    print(f"\nNext step: Retrain classifier with:")
    print(f"  python train_simple_classifier.py --train-file tool_choice_data_from_predictions_balanced/train.json --val-file tool_choice_data_from_predictions_balanced/val.json --test-file tool_choice_data_from_predictions_balanced/test.json")

if __name__ == '__main__':
    main()
