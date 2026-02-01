#!/usr/bin/env python3
"""
Create complete balanced dataset with ALL steps from ThinkGeoBench.
- Extract all 1537 steps from ThinkGeoBench.json
- Split by test_sample_indices FIRST (244 questions → ~1067 steps)
- Split remaining into train/val using indices SECOND (70/30)
- Create 1 negative per positive within each split
- Use "FinishAction" for answer-only steps
- Ensure per-tool balance in each split
"""

import json
import random
import os
from collections import defaultdict

print("="*80)
print("CREATING COMPLETE BALANCED DATASET - ALL STEPS")
print("="*80)

# Load test/train sample indices
with open('/home/james/ThinkGeo/tool_choice_data_from_predictions/test_sample_indices.json', 'r') as f:
    split_indices = json.load(f)

test_sample_indices = set(split_indices['test_sample_indices'])
train_val_sample_indices = set(split_indices['train_val_sample_indices'])

print(f"\nLoaded indices:")
print(f"  Test sample indices: {len(test_sample_indices)} questions")
print(f"  Train/Val sample indices: {len(train_val_sample_indices)} questions")

# Load ThinkGeoBench
with open('/home/james/ThinkGeo/opencompass/data/ThinkGeo_dataset/ThinkGeoBench.json', 'r') as f:
    bench_data = json.load(f)

print(f"  ThinkGeoBench: {len(bench_data)} questions")

# Extract all steps
print("\n1. Extracting all steps from ThinkGeoBench...")
all_steps = []

for question_id, question_data in bench_data.items():
    if not isinstance(question_data, dict) or not question_data.get('dialogs'):
        continue
    
    question_id_int = int(question_id)
    
    # Process dialogs to extract steps
    dialogs = question_data['dialogs']
    step_num = 0
    
    for dialog in dialogs:
        if not isinstance(dialog, dict):
            continue
        
        role = dialog.get('role')
        
        # Tool call step
        if role == 'assistant' and dialog.get('tool_calls'):
            tool_calls = dialog.get('tool_calls', [])
            if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
                tool_call = tool_calls[0]  # Take first tool call
                if isinstance(tool_call, dict) and 'function' in tool_call:
                    func_info = tool_call['function']
                    tool_name = func_info.get('name')
                    tool_args = func_info.get('arguments', {})
                    
                    step_data = {
                        'question_id': question_id,
                        'global_sample_idx': question_id_int,
                        'step': step_num,
                        'tool_name': tool_name,
                        'gold_tool': tool_name,
                        'tool_description': '',
                        'tool_arguments': tool_args,
                        'full_history': dialog.get('full_history', []),
                        'thought': dialog.get('thought', ''),
                        'label': 1  # This is positive (correct tool)
                    }
                    all_steps.append(step_data)
                    step_num += 1
        
        # Answer step (no tool call) - mark as FinishAction
        elif role == 'assistant' and not dialog.get('tool_calls'):
            # This is a final answer step
            step_data = {
                'question_id': question_id,
                'global_sample_idx': question_id_int,
                'step': step_num,
                'tool_name': 'FinishAction',
                'gold_tool': 'FinishAction',
                'tool_description': 'Final answer to the user query',
                'tool_arguments': {'answer': dialog.get('content', '')},
                'full_history': dialog.get('full_history', []),
                'thought': dialog.get('thought', ''),
                'label': 1  # This is positive
            }
            all_steps.append(step_data)
            step_num += 1

print(f"   Total steps extracted: {len(all_steps)}")

# Split by indices FIRST
print("\n2. Splitting by test/train/val sample indices...")
test_steps = [s for s in all_steps if s['global_sample_idx'] in test_sample_indices]
train_val_steps = [s for s in all_steps if s['global_sample_idx'] in train_val_sample_indices]

print(f"   Test steps: {len(test_steps)} ({len(test_sample_indices)} questions)")
print(f"   Train/Val steps: {len(train_val_steps)} ({len(train_val_sample_indices)} questions)")

# Further split train_val into train/val (70/30 by question index)
print("\n3. Splitting train/val 70/30 by question indices...")
train_val_list = list(train_val_sample_indices)
random.shuffle(train_val_list)
split_point = int(len(train_val_list) * 0.7)
train_indices = set(train_val_list[:split_point])
val_indices = set(train_val_list[split_point:])

train_steps = [s for s in train_val_steps if s['global_sample_idx'] in train_indices]
val_steps = [s for s in train_val_steps if s['global_sample_idx'] in val_indices]

print(f"   Train steps: {len(train_steps)} ({len(train_indices)} questions)")
print(f"   Val steps: {len(val_steps)} ({len(val_indices)} questions)")

# Collect all tools
print("\n4. Analyzing tool distribution...")
all_tools = set()
test_tools_count = defaultdict(int)
train_tools_count = defaultdict(int)
val_tools_count = defaultdict(int)

for step in test_steps:
    tool = step['tool_name']
    all_tools.add(tool)
    test_tools_count[tool] += 1

for step in train_steps:
    tool = step['tool_name']
    all_tools.add(tool)
    train_tools_count[tool] += 1

for step in val_steps:
    tool = step['tool_name']
    all_tools.add(tool)
    val_tools_count[tool] += 1

all_tools = sorted(list(all_tools))
print(f"   Total tools: {len(all_tools)}")
print(f"   Tools: {all_tools}")
print(f"   Test tool distribution: {dict(sorted(test_tools_count.items()))}")
print(f"   Train tool distribution: {dict(sorted(train_tools_count.items()))}")
print(f"   Val tool distribution: {dict(sorted(val_tools_count.items()))}")

# Create balanced datasets by creating 1 negative per positive within each split
print("\n5. Creating balanced samples (1 positive + 1 negative per step in each split)...")

def create_negative_sample(positive_sample, all_tools_list):
    """Create a negative sample by randomly selecting a different tool"""
    negative_tool = random.choice([t for t in all_tools_list if t != positive_sample['tool_name']])
    negative_sample = positive_sample.copy()
    negative_sample['tool_name'] = negative_tool
    negative_sample['label'] = 0  # Negative sample
    return negative_sample

test_balanced = []
train_balanced = []
val_balanced = []

# Test set
for step in test_steps:
    test_balanced.append(step)  # Add positive
    test_balanced.append(create_negative_sample(step, all_tools))  # Add negative

# Train set
for step in train_steps:
    train_balanced.append(step)  # Add positive
    train_balanced.append(create_negative_sample(step, all_tools))  # Add negative

# Val set
for step in val_steps:
    val_balanced.append(step)  # Add positive
    val_balanced.append(create_negative_sample(step, all_tools))  # Add negative

print(f"   Test balanced samples: {len(test_balanced)} ({len(test_steps)} pos, {len(test_steps)} neg)")
print(f"   Train balanced samples: {len(train_balanced)} ({len(train_steps)} pos, {len(train_steps)} neg)")
print(f"   Val balanced samples: {len(val_balanced)} ({len(val_steps)} pos, {len(val_steps)} neg)")

# Shuffle
random.shuffle(test_balanced)
random.shuffle(train_balanced)
random.shuffle(val_balanced)

# Verify balance in each split
print("\n6. Verifying balance in each split...")

def verify_balance(dataset, split_name, tools_list):
    pos_count = sum(1 for s in dataset if s['label'] == 1)
    neg_count = sum(1 for s in dataset if s['label'] == 0)
    pos_pct = (pos_count / len(dataset) * 100) if len(dataset) > 0 else 0
    print(f"   {split_name}: {pos_count} pos, {neg_count} neg ({pos_pct:.1f}% pos)")
    
    # Check per-tool balance
    tool_pos = defaultdict(int)
    tool_neg = defaultdict(int)
    for sample in dataset:
        if sample['label'] == 1:
            tool_pos[sample['tool_name']] += 1
        else:
            tool_neg[sample['tool_name']] += 1
    
    # Check if all tools have at least 1 pos and 1 neg
    all_balanced = True
    missing_tools = []
    for tool in tools_list:
        if tool_pos[tool] < 1 or tool_neg[tool] < 1:
            all_balanced = False
            missing_tools.append(f"{tool}(p:{tool_pos[tool]},n:{tool_neg[tool]})")
    
    if all_balanced:
        print(f"      ✓ All tools have 1+ pos and 1+ neg")
    else:
        print(f"      ✗ Some tools lack balance: {', '.join(missing_tools)}")
    
    return all_balanced

verify_balance(test_balanced, "Test", all_tools)
verify_balance(train_balanced, "Train", all_tools)
verify_balance(val_balanced, "Val", all_tools)

# Save datasets
print("\n7. Saving datasets...")
output_dir = '/home/james/ThinkGeo/tool_choice_data_balanced_from_original'
os.makedirs(output_dir, exist_ok=True)

with open(f'{output_dir}/test.json', 'w') as f:
    json.dump(test_balanced, f, indent=2)

with open(f'{output_dir}/train.json', 'w') as f:
    json.dump(train_balanced, f, indent=2)

with open(f'{output_dir}/val.json', 'w') as f:
    json.dump(val_balanced, f, indent=2)

# Save summary
summary = {
    'total_steps_extracted': len(all_steps),
    'test_steps': len(test_steps),
    'train_steps': len(train_steps),
    'val_steps': len(val_steps),
    'test_balanced_samples': len(test_balanced),
    'train_balanced_samples': len(train_balanced),
    'val_balanced_samples': len(val_balanced),
    'test_questions': len(test_sample_indices),
    'train_questions': len(train_indices),
    'val_questions': len(val_indices),
    'all_tools': all_tools,
    'test_tool_distribution': dict(sorted(test_tools_count.items())),
    'train_tool_distribution': dict(sorted(train_tools_count.items())),
    'val_tool_distribution': dict(sorted(val_tools_count.items())),
}

with open(f'{output_dir}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"   Saved to: {output_dir}")
print(f"   Files: test.json ({len(test_balanced)}), train.json ({len(train_balanced)}), val.json ({len(val_balanced)}), summary.json")

print("\n" + "="*80)
print("DATASET CREATION COMPLETE")
print("="*80)
