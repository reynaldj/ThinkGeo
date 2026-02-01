#!/usr/bin/env python3
"""
Create balanced dataset using realistic wrong tools from previous model runs.
- Extract all 1921 steps from ThinkGeoBench
- For each step, find what wrong tools models actually called (from predictions)
- Use tiered fallback for negatives:
    1) exact step wrong tools
    2) nearby steps in same question
    3) same gold tool across questions
    4) random wrong tool
- Create 1 negative per positive using these realistic wrong tools
"""

import json
import random
import os
from collections import defaultdict
import glob

print("="*80)
print("CREATING BALANCED DATASET WITH REALISTIC WRONG TOOLS FROM PREDICTIONS")
print("="*80)

# Load test/train sample indices
with open('/home/james/ThinkGeo/tool_choice_data_from_predictions/test_sample_indices.json', 'r') as f:
    split_indices = json.load(f)

test_sample_indices = set(split_indices['test_sample_indices'])
train_val_sample_indices = set(split_indices['train_val_sample_indices'])

# Load tool descriptions from toolmeta.json
with open('opencompass/data/ThinkGeo_dataset/toolmeta.json', 'r') as f:
    toolmeta = json.load(f)
    tool_descriptions = {tool_name: tool_info.get('description', '') 
                        for tool_name, tool_info in toolmeta.items()}

print(f"\nLoaded indices:")
print(f"  Test sample indices: {len(test_sample_indices)} questions")
print(f"  Train/Val sample indices: {len(train_val_sample_indices)} questions")

# Load ThinkGeoBench
with open('/home/james/ThinkGeo/opencompass/data/ThinkGeo_dataset/ThinkGeoBench.json', 'r') as f:
    bench_data = json.load(f)

print(f"  ThinkGeoBench: {len(bench_data)} questions")

# Collect all prediction files from different model runs
print("\n1. Collecting predictions from all model runs...")
prediction_dirs = glob.glob('/home/james/ThinkGeo/opencompass/outputs/default/*/predictions/*/ThinkGeo_bench_*.json')
prediction_dirs = sorted(set([os.path.dirname(d) for d in prediction_dirs]))

print(f"   Found {len(prediction_dirs)} prediction directories:")
for pred_dir in prediction_dirs[:5]:
    print(f"     - {pred_dir}")
if len(prediction_dirs) > 5:
    print(f"     ... and {len(prediction_dirs) - 5} more")

# Build maps of observed wrong tools
wrong_tools_map = defaultdict(set)  # (question_id, step_num) -> set(wrong tools)
wrong_tools_by_question = defaultdict(lambda: defaultdict(set))  # question_id -> step_num -> set(wrong tools)
wrong_tools_by_gold = defaultdict(set)  # gold_tool -> set(wrong tools)

for pred_dir in prediction_dirs:
    pred_files = glob.glob(f"{pred_dir}/ThinkGeo_bench_*.json")
    
    for pred_file in pred_files:
        try:
            with open(pred_file, 'r') as f:
                preds = json.load(f)
            
            # Each entry is keyed by question ID
            for q_id_str, entry in preds.items():
                q_id = int(q_id_str)
                
                if 'prediction' not in entry:
                    continue
                
                prediction_steps = entry.get('prediction', [])
                gold_steps = entry.get('gold', [])
                
                # Compare prediction vs gold to find wrong tools
                for step_idx in range(min(len(prediction_steps), len(gold_steps))):
                    pred_step = prediction_steps[step_idx]
                    gold_step = gold_steps[step_idx]
                    
                    # Extract tool names
                    pred_tool = None
                    gold_tool = None
                    
                    if isinstance(pred_step, dict):
                        if pred_step.get('tool_calls'):
                            tool_calls = pred_step.get('tool_calls', [])
                            if tool_calls and len(tool_calls) > 0:
                                pred_tool = tool_calls[0].get('function', {}).get('name')
                        elif pred_step.get('role') == 'assistant':
                            pred_tool = 'FinishAction'
                    
                    if isinstance(gold_step, dict):
                        if gold_step.get('tool_calls'):
                            tool_calls = gold_step.get('tool_calls', [])
                            if tool_calls and len(tool_calls) > 0:
                                gold_tool = tool_calls[0].get('function', {}).get('name')
                        elif gold_step.get('role') == 'assistant':
                            gold_tool = 'FinishAction'
                    
                    # If prediction differs from gold, it's a wrong tool
                    if pred_tool and gold_tool and pred_tool != gold_tool:
                        wrong_tools_map[(q_id, step_idx)].add(pred_tool)
                        wrong_tools_by_question[q_id][step_idx].add(pred_tool)
                        wrong_tools_by_gold[gold_tool].add(pred_tool)
        
        except Exception as e:
            pass

print(f"   Collected wrong tools for {len(wrong_tools_map)} (question, step) pairs")

# Extract all steps from ThinkGeoBench
print("\n2. Extracting all steps from ThinkGeoBench...")
all_steps = []

for question_id, question_data in bench_data.items():
    if not isinstance(question_data, dict) or not question_data.get('dialogs'):
        continue
    
    question_id_int = int(question_id)
    
    dialogs = question_data['dialogs']
    step_num = 0
    accumulated_history = []  # Build history as we go through dialogs
    
    for dialog_idx, dialog in enumerate(dialogs):
        if not isinstance(dialog, dict):
            continue
        
        role = dialog.get('role')
        
        # Tool call step - save BEFORE adding this dialog to history
        if role == 'assistant' and dialog.get('tool_calls'):
            tool_calls = dialog.get('tool_calls', [])
            if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
                tool_call = tool_calls[0]
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
                        'tool_description': tool_descriptions.get(tool_name, ''),
                        'tool_arguments': tool_args,
                        'full_history': list(accumulated_history),  # Copy history up to this point
                        'thought': dialog.get('thought', ''),
                        'label': 1,
                        'wrong_tools_from_runs': list(wrong_tools_map.get((question_id_int, step_num), set()))
                    }
                    all_steps.append(step_data)
                    step_num += 1
        
        # Answer step - save BEFORE adding this dialog to history
        elif role == 'assistant' and not dialog.get('tool_calls'):
            step_data = {
                'question_id': question_id,
                'global_sample_idx': question_id_int,
                'step': step_num,
                'tool_name': 'FinishAction',
                'gold_tool': 'FinishAction',
                'tool_description': tool_descriptions.get('FinishAction', 'Final answer to the user query'),
                'tool_arguments': {'answer': dialog.get('content', '')},
                'full_history': list(accumulated_history),  # Copy history up to this point
                'thought': dialog.get('thought', ''),
                'label': 1,
                'wrong_tools_from_runs': list(wrong_tools_map.get((question_id_int, step_num), set()))
            }
            all_steps.append(step_data)
            step_num += 1
        
        # Add this dialog to accumulated history for future steps
        accumulated_history.append(dialog)

print(f"   Total steps extracted: {len(all_steps)}")

# Check how many steps have wrong tool alternatives
steps_with_wrong_tools = sum(1 for s in all_steps if len(s.get('wrong_tools_from_runs', [])) > 0)
print(f"   Steps with observed wrong tools: {steps_with_wrong_tools} ({steps_with_wrong_tools/len(all_steps)*100:.1f}%)")

# Split by indices
print("\n3. Splitting by test/train/val sample indices...")
test_steps = [s for s in all_steps if s['global_sample_idx'] in test_sample_indices]
train_val_steps = [s for s in all_steps if s['global_sample_idx'] in train_val_sample_indices]

print(f"   Test steps: {len(test_steps)}")
print(f"   Train/Val steps: {len(train_val_steps)}")

# Split train/val 70/30
print("\n4. Splitting train/val 70/30 by question indices...")
train_val_list = list(train_val_sample_indices)
random.shuffle(train_val_list)
split_point = int(len(train_val_list) * 0.7)
train_indices = set(train_val_list[:split_point])
val_indices = set(train_val_list[split_point:])

train_steps = [s for s in train_val_steps if s['global_sample_idx'] in train_indices]
val_steps = [s for s in train_val_steps if s['global_sample_idx'] in val_indices]

print(f"   Train steps: {len(train_steps)}")
print(f"   Val steps: {len(val_steps)}")

# Get all tools
all_tools = set()
for step in all_steps:
    all_tools.add(step['tool_name'])
all_tools = sorted(list(all_tools))
print(f"\n5. Available tools ({len(all_tools)}): {all_tools}")

# Create balanced datasets with realistic wrong tools
print("\n6. Creating negative samples with realistic wrong tools...")

NEARBY_STEP_WINDOW = 1

def get_nearby_wrong_tools(question_id, step_num):
    nearby_tools = set()
    for offset in range(1, NEARBY_STEP_WINDOW + 1):
        for neighbor in (step_num - offset, step_num + offset):
            if neighbor in wrong_tools_by_question.get(question_id, {}):
                nearby_tools.update(wrong_tools_by_question[question_id][neighbor])
    return list(nearby_tools)

def create_negative_sample(positive_sample, all_tools_list):
    """Create negative by tiered selection of realistic wrong tools."""
    question_id = int(positive_sample['global_sample_idx'])
    step_num = positive_sample['step']
    gold_tool = positive_sample['gold_tool']

    # Tier 1: exact step wrong tools
    exact_wrong_tools = positive_sample.get('wrong_tools_from_runs', [])
    if exact_wrong_tools:
        wrong_tool = random.choice(exact_wrong_tools)
        tier = 'exact'
    else:
        # Tier 2: nearby steps in same question
        nearby_wrong_tools = get_nearby_wrong_tools(question_id, step_num)
        if nearby_wrong_tools:
            wrong_tool = random.choice(nearby_wrong_tools)
            tier = 'nearby'
        else:
            # Tier 3: same gold tool across questions
            same_tool_wrong_tools = list(wrong_tools_by_gold.get(gold_tool, set()))
            if same_tool_wrong_tools:
                wrong_tool = random.choice(same_tool_wrong_tools)
                tier = 'same_tool'
            else:
                # Tier 4: random wrong tool
                wrong_tool = random.choice([t for t in all_tools_list if t != positive_sample['tool_name']])
                tier = 'random'

    negative_sample = positive_sample.copy()
    negative_sample['tool_name'] = wrong_tool
    negative_sample['tool_description'] = tool_descriptions.get(wrong_tool, '')
    negative_sample['label'] = 0
    return negative_sample, tier

test_balanced = []
train_balanced = []
val_balanced = []

test_tier_counts = defaultdict(int)
train_tier_counts = defaultdict(int)
val_tier_counts = defaultdict(int)

# Test set
for step in test_steps:
    test_balanced.append(step)
    neg, tier = create_negative_sample(step, all_tools)
    test_balanced.append(neg)
    test_tier_counts[tier] += 1

# Train set
for step in train_steps:
    train_balanced.append(step)
    neg, tier = create_negative_sample(step, all_tools)
    train_balanced.append(neg)
    train_tier_counts[tier] += 1

# Val set
for step in val_steps:
    val_balanced.append(step)
    neg, tier = create_negative_sample(step, all_tools)
    val_balanced.append(neg)
    val_tier_counts[tier] += 1

def format_tier_counts(tier_counts, total):
    parts = []
    for tier in ['exact', 'nearby', 'same_tool', 'random']:
        count = tier_counts.get(tier, 0)
        parts.append(f"{tier}:{count}")
    return f"{', '.join(parts)} (total:{total})"

print(f"   Test: {len(test_balanced)} samples - {format_tier_counts(test_tier_counts, len(test_steps))}")
print(f"   Train: {len(train_balanced)} samples - {format_tier_counts(train_tier_counts, len(train_steps))}")
print(f"   Val: {len(val_balanced)} samples - {format_tier_counts(val_tier_counts, len(val_steps))}")

# Shuffle
random.shuffle(test_balanced)
random.shuffle(train_balanced)
random.shuffle(val_balanced)

# Verify balance
print("\n7. Verifying balance in each split...")

def verify_balance(dataset, split_name, tools_list):
    pos_count = sum(1 for s in dataset if s['label'] == 1)
    neg_count = sum(1 for s in dataset if s['label'] == 0)
    pos_pct = (pos_count / len(dataset) * 100) if len(dataset) > 0 else 0
    print(f"   {split_name}: {pos_count} pos, {neg_count} neg ({pos_pct:.1f}% pos)")
    
    tool_pos = defaultdict(int)
    tool_neg = defaultdict(int)
    for sample in dataset:
        if sample['label'] == 1:
            tool_pos[sample['tool_name']] += 1
        else:
            tool_neg[sample['tool_name']] += 1
    
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

def get_tool_distribution(samples):
    dist = defaultdict(lambda: {'positive': 0, 'negative': 0, 'total': 0})
    for sample in samples:
        tool = sample.get('tool_name', 'Unknown')
        label = sample.get('label', 0)
        if label == 1:
            dist[tool]['positive'] += 1
        else:
            dist[tool]['negative'] += 1
        dist[tool]['total'] += 1
    return {tool: dist[tool] for tool in sorted(dist.keys())}

verify_balance(test_balanced, "Test", all_tools)
verify_balance(train_balanced, "Train", all_tools)
verify_balance(val_balanced, "Val", all_tools)

# Save datasets
print("\n8. Saving datasets...")
output_dir = '/home/james/ThinkGeo/tool_choice_data_balanced_realistic_negatives'
os.makedirs(output_dir, exist_ok=True)

# Remove wrong_tools_from_runs before saving (internal use only)
def clean_sample(s):
    cleaned = s.copy()
    cleaned.pop('wrong_tools_from_runs', None)
    return cleaned

test_clean = [clean_sample(s) for s in test_balanced]
train_clean = [clean_sample(s) for s in train_balanced]
val_clean = [clean_sample(s) for s in val_balanced]

with open(f'{output_dir}/test.json', 'w') as f:
    json.dump(test_clean, f, indent=2)

with open(f'{output_dir}/train.json', 'w') as f:
    json.dump(train_clean, f, indent=2)

with open(f'{output_dir}/val.json', 'w') as f:
    json.dump(val_clean, f, indent=2)

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
    'negatives_strategy': (
        'Tiered: exact-step wrong tools, then nearby-step wrong tools, '
        'then same-gold-tool wrong tools across questions, then random'
    ),
    'prediction_sources': len(prediction_dirs),
    'negatives_tiers': {
        'test': dict(test_tier_counts),
        'train': dict(train_tier_counts),
        'val': dict(val_tier_counts),
    },
    'nearby_step_window': NEARBY_STEP_WINDOW,
    'tool_distribution': {
        'test': get_tool_distribution(test_balanced),
        'train': get_tool_distribution(train_balanced),
        'val': get_tool_distribution(val_balanced),
    },
}

with open(f'{output_dir}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"   Saved to: {output_dir}")
print(f"   Files: test.json, train.json, val.json, summary.json")

print("\n" + "="*80)
print("DATASET CREATION COMPLETE")
print("="*80)
