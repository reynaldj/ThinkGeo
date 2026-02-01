#!/usr/bin/env python3
"""
Create a balanced tool choice dataset using ALL predictions from all model runs.

This script:
1. Loads predictions from ALL available model runs (not just one)
2. Extracts ALL tool calls from each sample (multiple steps per sample)
3. Uses all 436 samples across test and train indices
4. Creates balanced train/val/test splits where each tool has equal positive/negative samples
5. Saves the balanced datasets
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set
import random

# Set seed for reproducibility
random.seed(42)


def load_test_sample_indices():
    """Load test and train sample indices."""
    indices_file = '/home/james/ThinkGeo/tool_choice_data_from_predictions/test_sample_indices.json'
    with open(indices_file, 'r') as f:
        data = json.load(f)
    return set(data['test_sample_indices']), set(data['train_val_sample_indices'])


def find_all_run_folders() -> List[str]:
    """Find all available run folders with predictions."""
    outputs_dir = '/home/james/ThinkGeo/opencompass/outputs/default'
    run_folders = []
    
    for item in os.listdir(outputs_dir):
        item_path = os.path.join(outputs_dir, item)
        if not os.path.isdir(item_path):
            continue
        
        pred_dir = os.path.join(item_path, 'predictions')
        if os.path.isdir(pred_dir):
            run_folders.append(item)
    
    return sorted(run_folders)


def load_predictions_from_run(run_folder: str) -> Dict[str, dict]:
    """Load all predictions from a run folder."""
    all_predictions = {}
    pred_dir = f'/home/james/ThinkGeo/opencompass/outputs/default/{run_folder}/predictions'
    
    if not os.path.isdir(pred_dir):
        return all_predictions
    
    # Find model subdirectory (could be qwen2.5-7b-instruct, qwen1.5-7b-chat, qwen3-8b, etc.)
    for model_dir in os.listdir(pred_dir):
        model_path = os.path.join(pred_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
        
        # Load all bench files (ThinkGeo_bench_0 to ThinkGeo_bench_8)
        for i in range(9):
            bench_file = os.path.join(model_path, f'ThinkGeo_bench_{i}.json')
            if os.path.exists(bench_file):
                try:
                    with open(bench_file, 'r') as f:
                        data = json.load(f)
                        all_predictions.update(data)
                except (json.JSONDecodeError, IOError) as e:
                    print(f"   Warning: Could not load {bench_file}: {e}")
    
    return all_predictions


def extract_tool_samples(all_run_predictions: Dict[str, Dict[str, dict]], 
                        test_indices: Set[int], train_indices: Set[int]):
    """Extract tool calls as positive/negative samples from ALL predictions."""
    samples = defaultdict(list)
    tool_counts = defaultdict(lambda: {'positive': 0, 'negative': 0})
    tool_descriptions = {}
    
    processed_samples = set()
    total_tool_calls = 0
    
    for run_name, predictions in all_run_predictions.items():
        print(f"\n   Processing {run_name}...")
        run_tool_calls = 0
        
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
            
            processed_samples.add(sample_id)
            
            # Extract ALL tool calls from this sample (multiple steps)
            gold_responses = sample_data.get('gold', [])
            
            for response in gold_responses:
                if response.get('role') == 'assistant' and response.get('tool_calls'):
                    for tool_call in response.get('tool_calls', []):
                        func = tool_call.get('function', {})
                        tool_name = func.get('name')
                        args = func.get('arguments', {})
                        
                        if tool_name and tool_name != 'NoAction':
                            # This is a positive sample (correct tool choice)
                            samples[tool_name].append({
                                'sample_id': sample_id,
                                'label': 1,  # Correct tool
                                'split': 'test' if is_test else 'train',
                                'run': run_name,
                                'tool_args': args
                            })
                            tool_counts[tool_name]['positive'] += 1
                            total_tool_calls += 1
                            run_tool_calls += 1
                            
                            # Store tool description if not already stored
                            if tool_name not in tool_descriptions:
                                # Try to extract description from args or store a generic one
                                tool_descriptions[tool_name] = f"Tool {tool_name}"
        
        print(f"      Found {run_tool_calls} tool calls")
    
    print(f"\n   Total samples processed: {len(processed_samples)}")
    print(f"   Total tool calls extracted: {total_tool_calls}")
    
    return samples, tool_counts, tool_descriptions


def create_negative_samples(positive_samples: Dict[str, list], all_tools: set, 
                           test_indices: Set[int]) -> Dict[str, list]:
    """Create negative samples by pairing tools with wrong contexts."""
    negative_samples = defaultdict(list)
    
    tools_list = list(all_tools)
    
    for tool_name in all_tools:
        positive_count = len(positive_samples.get(tool_name, []))
        
        # Create roughly equal number of negatives
        for i in range(positive_count):
            # Pick a random different tool and use one of its positive samples as context
            wrong_tool = random.choice([t for t in tools_list if t != tool_name])
            other_samples = positive_samples.get(wrong_tool, [])
            
            if other_samples:
                source_sample = random.choice(other_samples)
                negative_samples[tool_name].append({
                    'sample_id': source_sample['sample_id'],
                    'label': 0,  # Incorrect tool (tool_name is wrong for this sample)
                    'split': source_sample['split'],
                    'run': source_sample['run'],
                    'actual_tool': wrong_tool
                })
    
    return negative_samples


def create_balanced_splits(positive_samples: Dict[str, list], negative_samples: Dict[str, list]) -> Tuple[list, list, list]:
    """Create balanced train/val/test splits for each tool."""
    train_data = []
    val_data = []
    test_data = []
    
    print("\nTool balance summary:")
    for tool_name in sorted(positive_samples.keys()):
        pos_samples = positive_samples.get(tool_name, [])
        neg_samples = negative_samples.get(tool_name, [])
        
        # Balance positive and negative
        min_count = min(len(pos_samples), len(neg_samples))
        balanced_pos = random.sample(pos_samples, min(len(pos_samples), min_count))
        balanced_neg = random.sample(neg_samples, min(len(neg_samples), min_count))
        
        all_samples = balanced_pos + balanced_neg
        random.shuffle(all_samples)
        
        # Split: 70% train, 15% val, 15% test
        train_split = int(len(all_samples) * 0.7)
        val_split = int(len(all_samples) * 0.85)
        
        train_samples = all_samples[:train_split]
        val_samples = all_samples[train_split:val_split]
        test_samples = all_samples[val_split:]
        
        for sample in train_samples:
            train_data.append({**sample, 'tool_name': tool_name})
        
        for sample in val_samples:
            val_data.append({**sample, 'tool_name': tool_name})
        
        for sample in test_samples:
            test_data.append({**sample, 'tool_name': tool_name})
        
        train_pos = len([s for s in train_samples if s['label'] == 1])
        train_neg = len([s for s in train_samples if s['label'] == 0])
        val_pos = len([s for s in val_samples if s['label'] == 1])
        val_neg = len([s for s in val_samples if s['label'] == 0])
        test_pos = len([s for s in test_samples if s['label'] == 1])
        test_neg = len([s for s in test_samples if s['label'] == 0])
        
        print(f"\n{tool_name}:")
        print(f"  Total: {len(all_samples)} (pos: {len(balanced_pos)}, neg: {len(balanced_neg)})")
        print(f"  Train: {train_pos} pos, {train_neg} neg")
        print(f"  Val:   {val_pos} pos, {val_neg} neg")
        print(f"  Test:  {test_pos} pos, {test_neg} neg")
    
    return train_data, val_data, test_data


def main():
    """Main pipeline."""
    print("=" * 80)
    print("CREATING BALANCED TOOL CHOICE DATASET FROM ALL RUNS")
    print("=" * 80)
    
    # Load sample indices
    print("\n1. Loading sample indices...")
    test_indices, train_indices = load_test_sample_indices()
    print(f"   Test samples: {len(test_indices)}")
    print(f"   Train samples: {len(train_indices)}")
    print(f"   Total: {len(test_indices) + len(train_indices)}")
    
    # Find all run folders
    print("\n2. Finding all run folders...")
    run_folders = find_all_run_folders()
    print(f"   Found {len(run_folders)} run folders:")
    for rf in run_folders:
        print(f"      - {rf}")
    
    # Load predictions from all runs
    print("\n3. Loading predictions from all runs...")
    all_run_predictions = {}
    total_samples_loaded = 0
    
    for run_folder in run_folders:
        predictions = load_predictions_from_run(run_folder)
        if predictions:
            all_run_predictions[run_folder] = predictions
            total_samples_loaded += len(predictions)
            print(f"   {run_folder}: {len(predictions)} samples")
    
    print(f"   Total predictions loaded: {total_samples_loaded}")
    
    if not all_run_predictions:
        print("ERROR: No predictions found!")
        return
    
    # Extract tool samples
    print("\n4. Extracting tool calls from all predictions...")
    positive_samples, tool_counts, tool_descriptions = extract_tool_samples(
        all_run_predictions, test_indices, train_indices
    )
    
    print(f"\n   Found {len(positive_samples)} unique tools")
    for tool_name in sorted(tool_counts.keys()):
        counts = tool_counts[tool_name]
        print(f"   - {tool_name}: {counts['positive']} positive samples")
    
    # Create negative samples
    print("\n5. Creating negative samples for balance...")
    all_tools = set(positive_samples.keys())
    negative_samples = create_negative_samples(positive_samples, all_tools, test_indices)
    print(f"   Created negative samples for {len(negative_samples)} tools")
    
    # Create balanced splits
    print("\n6. Creating balanced train/val/test splits...")
    train_data, val_data, test_data = create_balanced_splits(positive_samples, negative_samples)
    
    # Save datasets
    print("\n7. Saving datasets...")
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
    total_samples = total_pos + total_neg
    
    print(f"Total samples: {total_samples}")
    print(f"  Positive (correct tools): {total_pos} ({100*total_pos/total_samples:.1f}%)")
    print(f"  Negative (incorrect tools): {total_neg} ({100*total_neg/total_samples:.1f}%)")
    print(f"\nDistribution:")
    print(f"  Train: {len(train_data)} samples ({100*len(train_data)/total_samples:.1f}%)")
    print(f"  Val:   {len(val_data)} samples ({100*len(val_data)/total_samples:.1f}%)")
    print(f"  Test:  {len(test_data)} samples ({100*len(test_data)/total_samples:.1f}%)")
    print("\n✓ Dataset creation complete!")


if __name__ == '__main__':
    main()
