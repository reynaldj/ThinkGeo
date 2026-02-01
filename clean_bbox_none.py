#!/usr/bin/env python3
"""
Clean dataset to remove bbox keys with None values.
bbox=None should be completely removed from arguments so schema doesn't include it.
"""

import json

print("="*80)
print("CLEANING DATASET: Removing bbox=None from arguments")
print("="*80)

for split in ['train', 'val', 'test']:
    filepath = f'/home/james/ThinkGeo/tool_choice_data_schema_aware/{split}.json'
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    cleaned_count = 0
    for sample in data:
        tool_args = sample.get('tool_arguments', {})
        
        # Remove bbox if it's None
        if 'bbox' in tool_args and tool_args['bbox'] is None:
            del tool_args['bbox']
            cleaned_count += 1
            
            # Regenerate schema without bbox
            sample['argument_schema'] = {k: '<MASKED>' for k in tool_args.keys()}
    
    # Save cleaned data
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"{split}.json: Cleaned {cleaned_count} samples (removed bbox=None)")

print("\n" + "="*80)
print("✓ Dataset cleaned successfully!")
print("="*80)
