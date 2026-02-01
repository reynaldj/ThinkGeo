"""
Comprehensive augmentation: Fix BOTH first-step AND multi-step tool sequence issues.

Problem 1: First-step calls get low confidence (model bias toward history)
Problem 2: Multi-step tool repeats are underrepresented

Solution:
- Duplicate CORRECT first-step samples (especially TextToBbox) to emphasize first-step correctness
- Duplicate CORRECT multi-step samples to balance sequences
"""

import json
from collections import Counter

def comprehensive_augmentation():
    """Augment dataset for both first-step and multi-step scenarios."""
    
    train_path = '/home/james/ThinkGeo/tool_choice_data_schema_aware/train.json'
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    
    # Categorize all samples
    first_step_correct = []
    multi_step_correct = []
    all_other = []
    
    for sample in train_data:
        if sample.get('label') != 1:  # Only correct samples
            all_other.append(sample)
            continue
        
        history = sample.get('full_history', [])
        prior_tools = sum(1 for h in history if h.get('role') in ('assistant', 'tool', 'function'))
        
        if prior_tools == 0:
            first_step_correct.append(sample)
        else:
            multi_step_correct.append(sample)
    
    print(f"Original dataset breakdown (CORRECT samples only):")
    print(f"  First-step correct: {len(first_step_correct)}")
    print(f"  Multi-step correct: {len(multi_step_correct)}")
    print(f"  Other (incorrect): {len(all_other)}")
    
    # Identify which first-step tools are underrepresented
    first_tools = Counter(s['tool_name'] for s in first_step_correct)
    print(f"\nFirst-step tool distribution:")
    for tool, count in first_tools.most_common():
        print(f"  {tool}: {count}")
    
    # Strategy:
    # 1. Duplicate ALL first-step correct samples 2x (emphasize first-step correctness)
    # 2. Duplicate multi-step correct samples 3x (fix sequence imbalance)
    
    augmented = train_data.copy()
    
    # Add 2x first-step correct (total will be 3x: original + 2x)
    duplicates_first = first_step_correct * 2
    
    # Add 3x multi-step correct
    duplicates_multi = multi_step_correct * 3
    
    max_idx = max(s.get('global_sample_idx', 0) for s in train_data)
    idx = max_idx + 1
    
    for sample in duplicates_first:
        sample = sample.copy()
        sample['global_sample_idx'] = idx
        idx += 1
        augmented.append(sample)
    
    for sample in duplicates_multi:
        sample = sample.copy()
        sample['global_sample_idx'] = idx
        idx += 1
        augmented.append(sample)
    
    print(f"\nAugmentation strategy:")
    print(f"  First-step correct x2: +{len(duplicates_first)} samples")
    print(f"  Multi-step correct x3: +{len(duplicates_multi)} samples")
    print(f"  Total added: {len(duplicates_first) + len(duplicates_multi)}")
    print(f"\nNew dataset size: {len(train_data)} + {len(duplicates_first) + len(duplicates_multi)} = {len(augmented)}")
    
    # New distribution
    total_correct = len(first_step_correct) * 3 + len(multi_step_correct) * 4  # x3 first + x4 multi
    print(f"\nNew correct distribution (approximate):")
    print(f"  First-step correct: {len(first_step_correct) * 3} ({100*len(first_step_correct)*3/total_correct:.1f}%)")
    print(f"  Multi-step correct: {len(multi_step_correct) * 4} ({100*len(multi_step_correct)*4/total_correct:.1f}%)")
    
    # Save augmented dataset
    out_train = '/home/james/ThinkGeo/tool_choice_data_schema_aware/train_augmented.json'
    with open(out_train, 'w') as f:
        json.dump(augmented, f)
    print(f"\nSaved to {out_train}")
    
    # Also copy val and test
    for split in ['val', 'test']:
        src = f'/home/james/ThinkGeo/tool_choice_data_schema_aware/{split}.json'
        dst = f'/home/james/ThinkGeo/tool_choice_data_schema_aware/{split}_augmented.json'
        with open(src, 'r') as f:
            data = json.load(f)
        with open(dst, 'w') as f:
            json.dump(data, f)

if __name__ == '__main__':
    comprehensive_augmentation()
