"""
Better augmentation: Increase TextToBbox representation in multi-step scenarios.

Current problem:
- TextToBbox has HIGH confidence as FIRST tool (97.6% correct in training)
- TextToBbox has LOW confidence as SECOND+ tool (model not trained on this)
- Need to balance by duplicating TextToBbox samples that appear later in sequences
"""

import json

def augment_training_data_better():
    """Augment dataset focusing on TextToBbox in multi-step scenarios."""
    
    train_path = '/home/james/ThinkGeo/tool_choice_data_schema_aware/train.json'
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    
    # Find all TextToBbox samples (correct ones)
    all_texttobbox = [s for s in train_data if s['tool_name'] == 'TextToBbox' and s.get('label') == 1]
    
    # Separate by position in sequence
    first_step = []
    multi_step = []
    
    for sample in all_texttobbox:
        history = sample.get('full_history', [])
        prior_tools = sum(1 for h in history if h.get('role') in ('assistant', 'tool', 'function'))
        
        if prior_tools == 0:
            first_step.append(sample)
        else:
            multi_step.append(sample)
    
    print(f"TextToBbox correct samples:")
    print(f"  First-step: {len(first_step)} (confident)")
    print(f"  Multi-step: {len(multi_step)} (underrepresented)")
    print(f"  Ratio: {100*len(first_step)/(len(first_step)+len(multi_step)):.1f}% first-step")
    
    # Augment by duplicating multi-step TextToBbox 5x to match first-step prevalence
    augmented = train_data.copy()
    
    # Add 5 copies of each multi-step TextToBbox
    duplicates = multi_step * 5
    
    max_idx = max(s.get('global_sample_idx', 0) for s in train_data)
    for i, sample in enumerate(duplicates):
        sample = sample.copy()
        sample['global_sample_idx'] = max_idx + i + 1
        augmented.append(sample)
    
    print(f"\nAugmentation:")
    print(f"  Added {len(duplicates)} samples (multi-step TextToBbox x5)")
    print(f"  New dataset size: {len(train_data)} + {len(duplicates)} = {len(augmented)}")
    
    # Save augmented dataset
    out_train = '/home/james/ThinkGeo/tool_choice_data_schema_aware/train_augmented.json'
    with open(out_train, 'w') as f:
        json.dump(augmented, f)
    print(f"  Saved to {out_train}")
    
    # Also save val and test unchanged
    for split in ['val', 'test']:
        src = f'/home/james/ThinkGeo/tool_choice_data_schema_aware/{split}.json'
        dst = f'/home/james/ThinkGeo/tool_choice_data_schema_aware/{split}_augmented.json'
        with open(src, 'r') as f:
            data = json.load(f)
        with open(dst, 'w') as f:
            json.dump(data, f)
        print(f"  Copied {split} to {split}_augmented")

if __name__ == '__main__':
    augment_training_data_better()
