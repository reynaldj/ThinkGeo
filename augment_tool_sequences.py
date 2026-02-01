"""
Augment training data with repeated tool sequences.

Problem: The model gives low confidence when tools repeat (e.g., TextToBbox after CountGivenObject)
because these patterns are rare in training data (3.7% for TextToBbox).

Solution: Duplicate these rare patterns to increase representation.
"""

import json
from collections import Counter

def augment_training_data():
    """Augment dataset with repeated tool patterns."""
    
    train_path = '/home/james/ThinkGeo/tool_choice_data_schema_aware/train.json'
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    
    # Identify rare patterns: tool X appearing after tool Y
    tool_pair_counts = Counter()
    rare_patterns = []
    
    for sample in train_data:
        current_tool = sample['tool_name']
        label = sample.get('label', 0)
        
        if label != 1:  # Only keep correct ones
            continue
        
        history = sample.get('full_history', [])
        
        # Find last tool in history
        last_tool = None
        for h in reversed(history):
            if h.get('role') == 'assistant' and h.get('tool_calls'):
                for tc in reversed(h['tool_calls']):
                    func_name = tc.get('function', {}).get('name', '')
                    if func_name:
                        last_tool = func_name
                        break
                if last_tool:
                    break
        
        if last_tool:
            pair = (last_tool, current_tool)
            tool_pair_counts[pair] += 1
            
            # Identify rare pairs (less than 5% of samples)
            if pair[0] in ['CountGivenObject', 'Calculator'] and pair[1] == 'TextToBbox':
                rare_patterns.append(sample)
    
    print(f"Total samples: {len(train_data)}")
    print(f"Rare patterns found: {len(rare_patterns)}")
    print(f"\nTool pair distribution (top 10):")
    for pair, count in tool_pair_counts.most_common(10):
        print(f"  {pair[0]} → {pair[1]}: {count}")
    
    # Augment by duplicating rare patterns 3x
    augmented = train_data.copy()
    duplicates = [s for s in rare_patterns] * 3  # Triplicate rare patterns
    
    # Update global_sample_idx for new samples
    max_idx = max(s.get('global_sample_idx', 0) for s in train_data)
    for i, sample in enumerate(duplicates):
        sample = sample.copy()
        sample['global_sample_idx'] = max_idx + i + 1
        augmented.append(sample)
    
    print(f"\nAugmented dataset size: {len(train_data)} + {len(duplicates)} = {len(augmented)}")
    
    # Save augmented dataset
    out_train = '/home/james/ThinkGeo/tool_choice_data_schema_aware/train_augmented.json'
    with open(out_train, 'w') as f:
        json.dump(augmented, f)
    print(f"Saved to {out_train}")
    
    return len(augmented)

if __name__ == '__main__':
    augment_training_data()
