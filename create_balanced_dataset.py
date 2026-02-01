#!/usr/bin/env python3
"""
Create a balanced tool choice dataset using test and train sample indices.

This script:
1. Loads predictions from multiple model runs
2. Extracts tool calls from both test and train indices
3. Creates balanced train/val/test splits where each tool has roughly equal positive/negative samples
4. Saves the balanced datasets to tool_choice_data_from_predictions/
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import random

# Set seed for reproducibility
random.seed(42)


def load_test_sample_indices():
    """Load test and train sample indices."""
    indices_file = '/home/james/ThinkGeo/tool_choice_data_from_predictions/test_sample_indices.json'
    with open(indices_file, 'r') as f:
        data = json.load(f)
    return set(data['test_sample_indices']), set(data['train_val_sample_indices'])


def load_predictions(run_folder: str) -> Dict[str, dict]:
    """Load all predictions from a run folder."""
    all_predictions = {}
    pred_dir = f'/home/james/ThinkGeo/opencompass/outputs/default/{run_folder}/predictions'
    
    # Find model subdirectory
    for model_dir in os.listdir(pred_dir):
        model_path = os.path.join(pred_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
            
        # Load all bench files
        for i in range(9):  # ThinkGeo_bench_0 to ThinkGeo_bench_8
            bench_file = os.path.join(model_path, f'ThinkGeo_bench_{i}.json')
            if os.path.exists(bench_file):
                with open(bench_file, 'r') as f:
                    data = json.load(f)
                    all_predictions.update(data)
    
    return all_predictions


def extract_tool_samples(predictions: Dict[str, dict], test_indices: set, train_indices: set):
    """Extract tool calls as positive/negative samples from predictions."""
    samples = defaultdict(list)
    tool_counts = defaultdict(lambda: {'positive': 0, 'negative': 0})
    
    for sample_id_str, sample_data in predictions.items():
        try:
            sample_id = int(sample_id_str)
        except (ValueError, TypeError):
            continue
        
        # Determine if sample is in test or train
        is_test = sample_id in test_indices
        is_train = sample_id in train_indices
        
        if not (is_test or is_train):
            continue
        
        gold_responses = sample_data.get('gold', [])
        
        # Extract all tool calls from gold (these are positive samples)
        for response in gold_responses:
            if response.get('role') == 'assistant' and response.get('tool_calls'):
                for tool_call in response.get('tool_calls', []):
                    func = tool_call.get('function', {})
                    tool_name = func.get('name')
                    if tool_name and tool_name != 'NoAction':
                        # This is a positive sample (correct tool choice)
                        samples[tool_name].append({
                            'sample_id': sample_id,
                            'label': 1,  # Correct tool
                            'split': 'test' if is_test else 'train'
                        })
                        tool_counts[tool_name]['positive'] += 1
    
    return samples, tool_counts


def create_negative_samples(positive_samples: Dict[str, list], all_tools: set) -> Dict[str, list]:
    """Create negative samples by pairing tools with wrong contexts."""
    negative_samples = defaultdict(list)
    
    # For each tool, create negative samples by using it with different contexts
    tools_list = list(all_tools)
    
    for tool_name in all_tools:
        positive_count = len(positive_samples.get(tool_name, []))
        
        # Create roughly equal number of negatives
        for i in range(positive_count):
            # Pick a random different tool and a random positive sample
            wrong_tool = random.choice([t for t in tools_list if t != tool_name])
            other_samples = positive_samples.get(wrong_tool, [])
            
            if other_samples:
                source_sample = random.choice(other_samples)
                negative_samples[tool_name].append({
                    'sample_id': source_sample['sample_id'],
                    'label': 0,  # Incorrect tool (tool_name is wrong for this sample)
                    'split': source_sample['split'],
                    'actual_tool': wrong_tool
                })
    
    return negative_samples


def create_balanced_splits(positive_samples: Dict[str, list], negative_samples: Dict[str, list], 
                          test_indices: set, train_indices: set) -> Tuple[list, list, list]:
    """Create balanced train/val/test splits for each tool."""
    train_data = []
    val_data = []
    test_data = []
    
    for tool_name in positive_samples.keys():
        pos_samples = positive_samples.get(tool_name, [])
        neg_samples = negative_samples.get(tool_name, [])
        
        print(f"\n{tool_name}:")
        print(f"  Positive samples: {len(pos_samples)}")
        print(f"  Negative samples: {len(neg_samples)}")
        
        # Balance positive and negative
        min_count = min(len(pos_samples), len(neg_samples))
        balanced_pos = random.sample(pos_samples, min(len(pos_samples), min_count))
        balanced_neg = random.sample(neg_samples, min(len(neg_samples), min_count))
        
        all_samples = balanced_pos + balanced_neg
        random.shuffle(all_samples)
        
        # Split: 70% train, 15% val, 15% test
        train_split = int(len(all_samples) * 0.7)
        val_split = int(len(all_samples) * 0.85)
        
        for sample in all_samples[:train_split]:
            train_data.append({**sample, 'tool_name': tool_name})
        
        for sample in all_samples[train_split:val_split]:
            val_data.append({**sample, 'tool_name': tool_name})
        
        for sample in all_samples[val_split:]:
            test_data.append({**sample, 'tool_name': tool_name})
        
        print(f"  Train: {len([s for s in all_samples[:train_split] if s['label'] == 1])} pos, " +
              f"{len([s for s in all_samples[:train_split] if s['label'] == 0])} neg")
        print(f"  Val:   {len([s for s in all_samples[train_split:val_split] if s['label'] == 1])} pos, " +
              f"{len([s for s in all_samples[train_split:val_split] if s['label'] == 0])} neg")
        print(f"  Test:  {len([s for s in all_samples[val_split:] if s['label'] == 1])} pos, " +
              f"{len([s for s in all_samples[val_split:] if s['label'] == 0])} neg")
    
    return train_data, val_data, test_data


def main():
    """Main pipeline."""
    print("=" * 80)
    print("CREATING BALANCED TOOL CHOICE DATASET")
    print("=" * 80)
    
    # Load sample indices
    print("\n1. Loading sample indices...")
    test_indices, train_indices = load_test_sample_indices()
    print(f"   Test samples: {len(test_indices)}")
    print(f"   Train samples: {len(train_indices)}")
    
    # Load predictions from best run
    run_folder = 'qwen2.5-7bLowErrorBetterAns_Acc'
    print(f"\n2. Loading predictions from {run_folder}...")
    predictions = load_predictions(run_folder)
    print(f"   Loaded {len(predictions)} samples")
    
    # Extract tool samples
    print("\n3. Extracting tool calls from predictions...")
    positive_samples, tool_counts = extract_tool_samples(predictions, test_indices, train_indices)
    print(f"   Found {len(positive_samples)} unique tools")
    
    for tool_name, counts in sorted(tool_counts.items()):
        print(f"   - {tool_name}: {counts['positive']} positive samples")
    
    # Create negative samples
    print("\n4. Creating negative samples for balance...")
    all_tools = set(positive_samples.keys())
    negative_samples = create_negative_samples(positive_samples, all_tools)
    
    # Create balanced splits
    print("\n5. Creating balanced train/val/test splits...")
    train_data, val_data, test_data = create_balanced_splits(
        positive_samples, negative_samples, test_indices, train_indices
    )
    
    # Save datasets
    print("\n6. Saving datasets...")
    output_dir = '/home/james/ThinkGeo/tool_choice_data_balanced'
    os.makedirs(output_dir, exist_ok=True)
    
    train_file = os.path.join(output_dir, 'train.json')
    val_file = os.path.join(output_dir, 'val.json')
    test_file = os.path.join(output_dir, 'test.json')
    
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"   Saved {len(train_data)} training samples to {train_file}")
    
    with open(val_file, 'w') as f:
        json.dump(val_data, f, indent=2)
    print(f"   Saved {len(val_data)} validation samples to {val_file}")
    
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"   Saved {len(test_data)} test samples to {test_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total_pos = sum(1 for s in train_data + val_data + test_data if s['label'] == 1)
    total_neg = sum(1 for s in train_data + val_data + test_data if s['label'] == 0)
    print(f"Total samples: {total_pos + total_neg}")
    print(f"  Positive (correct tools): {total_pos} ({100*total_pos/(total_pos+total_neg):.1f}%)")
    print(f"  Negative (incorrect tools): {total_neg} ({100*total_neg/(total_pos+total_neg):.1f}%)")
    print(f"\nDistribution:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val:   {len(val_data)} samples")
    print(f"  Test:  {len(test_data)} samples")


if __name__ == '__main__':
    main()
