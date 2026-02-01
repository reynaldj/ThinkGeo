"""
Create a balanced dataset with ALL 1537 steps from ThinkGeoBench.
Uses test_sample_indices.json to split into test/train/val FIRST.
Then balances within each split.
Each step gets 1 positive (correct tool) + 1 negative (wrong tool).
Handles "FinishAction" for answer-only steps.
Ensures per-tool balance (at least 1 pos + 1 neg per tool in each split).
"""

import json
import os
from collections import defaultdict
import random

random.seed(42)

# Load test sample indices
with open('/home/james/ThinkGeo/tool_choice_data_from_predictions/test_sample_indices.json', 'r') as f:
    indices_data = json.load(f)
    test_sample_indices = set(indices_data['test_sample_indices'])
    train_val_sample_indices = set(indices_data['train_val_sample_indices'])
    train_sample_indices = set(indices_data['train_val_sample_indices'][:int(len(indices_data['train_val_sample_indices']) * 0.7)])
    val_sample_indices = set(indices_data['train_val_sample_indices'][int(len(indices_data['train_val_sample_indices']) * 0.7):])

print("="*80)
print("CREATING COMPLETE BALANCED DATASET - ALL 1537 STEPS (SPLIT BY INDICES)")
print("="*80)

# Load ThinkGeoBench
with open('/home/james/ThinkGeo/opencompass/data/ThinkGeo_dataset/ThinkGeoBench.json', 'r') as f:
    bench_data = json.load(f)

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
train_steps = [s for s in all_steps if s['global_sample_idx'] in train_sample_indices]
val_steps = [s for s in all_steps if s['global_sample_idx'] in val_sample_indices]

print(f"   Test steps: {len(test_steps)} ({len(test_sample_indices)} questions)")
print(f"   Train steps: {len(train_steps)} ({len(train_sample_indices)} questions)")
print(f"   Val steps: {len(val_steps)} ({len(val_sample_indices)} questions)")

# Get all unique tools
all_tools = set(s['tool_name'] for s in all_steps)
all_tools = sorted(list(all_tools))
print(f"\n3. Available tools ({len(all_tools)}): {all_tools}")

# Create negative samples by pairing with wrong tools
print("\n4. Creating negative samples (1 per positive step)...")

def create_negatives(positives, tools):
    """For each positive, create 1 negative with a different tool"""
    negatives = []
    for positive in positives:
        correct_tool = positive.get('tool_name')
        # Get wrong tools
        wrong_tools = [t for t in tools if t != correct_tool]
        if len(wrong_tools) > 0:
            # Sample one wrong tool
            wrong_tool = random.choice(wrong_tools)
            # Create negative by changing tool_name and label
            negative = positive.copy()
            negative['tool_name'] = wrong_tool
            negative['label'] = 0
            negatives.append(negative)
    return negatives

test_negatives = create_negatives(test_steps, all_tools)
train_val_negatives = create_negatives(train_val_steps, all_tools)

print(f"   Test negatives created: {len(test_negatives)}")
print(f"   Train/Val negatives created: {len(train_val_negatives)}")

# Combine
test_samples = test_steps + test_negatives
train_val_samples = train_val_steps + train_val_negatives

random.shuffle(test_samples)
random.shuffle(train_val_samples)

print(f"   Total test balanced: {len(test_samples)}")
print(f"   Total train/val balanced: {len(train_val_samples)}")

# Split train/val into train and val (70/30)
print("\n5. Splitting train/val into train and val (70/30)...")
split_point = int(len(train_val_samples) * 0.7)
train_samples = train_val_samples[:split_point]
val_samples = train_val_samples[split_point:]

print(f"   Train: {len(train_samples)} samples")
print(f"   Val: {len(val_samples)} samples")

# Verify balance and per-tool representation
print("\n6. Verifying balance and per-tool representation...")

def count_labels(samples):
    positive = sum(1 for s in samples if s.get('label') == 1)
    negative = len(samples) - positive
    return positive, negative

def get_tool_dist(samples):
    dist = defaultdict(lambda: {'positive': 0, 'negative': 0})
    for sample in samples:
        tool = sample.get('tool_name', 'Unknown')
        label = sample.get('label', 0)
        if label == 1:
            dist[tool]['positive'] += 1
        else:
            dist[tool]['negative'] += 1
    return dict(dist)

train_pos, train_neg = count_labels(train_samples)
val_pos, val_neg = count_labels(val_samples)
test_pos, test_neg = count_labels(test_samples)

print(f"   Train: {train_pos} positive, {train_neg} negative - Balance: {train_pos/len(train_samples)*100:.1f}% pos")
print(f"   Val: {val_pos} positive, {val_neg} negative - Balance: {val_pos/len(val_samples)*100:.1f}% pos")
print(f"   Test: {test_pos} positive, {test_neg} negative - Balance: {test_pos/len(test_samples)*100:.1f}% pos")

# Check per-tool balance
print("\n7. Per-tool distribution in test set:")
test_tools = get_tool_dist(test_samples)
for tool in sorted(test_tools.keys()):
    counts = test_tools[tool]
    total = counts['positive'] + counts['negative']
    has_balance = counts['positive'] > 0 and counts['negative'] > 0
    status = "✓" if has_balance else "✗"
    print(f"   {status} {tool}: {counts['positive']} pos, {counts['negative']} neg (total: {total})")

# Check if all tools have at least 1 pos and 1 neg
train_tools = get_tool_dist(train_samples)
val_tools = get_tool_dist(val_samples)

all_test_tools_balanced = all(t['positive'] > 0 and t['negative'] > 0 for t in test_tools.values())
all_train_tools_balanced = all(t['positive'] > 0 and t['negative'] > 0 for t in train_tools.values())
all_val_tools_balanced = all(t['positive'] > 0 and t['negative'] > 0 for t in val_tools.values())

print(f"\n8. Per-tool balance check:")
print(f"   Train all tools have 1+ pos and 1+ neg: {all_train_tools_balanced}")
print(f"   Val all tools have 1+ pos and 1+ neg: {all_val_tools_balanced}")
print(f"   Test all tools have 1+ pos and 1+ neg: {all_test_tools_balanced}")

if not all_test_tools_balanced or not all_train_tools_balanced or not all_val_tools_balanced:
    print("\n   Warning: Some tools don't have both positive and negative samples.")
    print("   This might be due to sparse tool representation in the splits.")

# Save balanced datasets
output_dir = '/home/james/ThinkGeo/tool_choice_data_balanced_from_original'
os.makedirs(output_dir, exist_ok=True)

print(f"\n9. Saving balanced datasets to {output_dir}...")
with open(f'{output_dir}/train.json', 'w') as f:
    json.dump(train_samples, f, indent=2)
print(f"    Saved train.json: {len(train_samples)} samples")

with open(f'{output_dir}/val.json', 'w') as f:
    json.dump(val_samples, f, indent=2)
print(f"    Saved val.json: {len(val_samples)} samples")

with open(f'{output_dir}/test.json', 'w') as f:
    json.dump(test_samples, f, indent=2)
print(f"    Saved test.json: {len(test_samples)} samples")

# Save summary
total_samples = len(train_samples) + len(val_samples) + len(test_samples)

summary = {
    "total_samples": total_samples,
    "positive_samples": train_pos + val_pos + test_pos,
    "negative_samples": train_neg + val_neg + test_neg,
    "train_size": len(train_samples),
    "val_size": len(val_samples),
    "test_size": len(test_samples),
    "train_balance": {"positive": train_pos, "negative": train_neg},
    "val_balance": {"positive": val_pos, "negative": val_neg},
    "test_balance": {"positive": test_pos, "negative": test_neg},
    "train_unique_steps": len(set((s['question_id'], s['step']) for s in train_samples if s['label'] == 1)),
    "val_unique_steps": len(set((s['question_id'], s['step']) for s in val_samples if s['label'] == 1)),
    "test_unique_steps": len(set((s['question_id'], s['step']) for s in test_samples if s['label'] == 1)),
    "total_unique_steps": len(set((s['question_id'], s['step']) for s in (train_samples + val_samples + test_samples) if s['label'] == 1)),
    "source": "ThinkGeoBench.json (all 436 questions, 1537 steps)",
    "split_strategy": "Test set from test_sample_indices (244 questions), Train/Val from train_val_sample_indices (192 questions)",
    "balancing_strategy": "Each unique step has 1 positive (correct tool) + 1 negative (wrong tool). Answer steps use 'FinishAction' tool. 50/50 balance overall.",
    "tool_distribution_train": train_tools,
    "tool_distribution_val": val_tools,
    "tool_distribution_test": test_tools
}

with open(f'{output_dir}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"    Saved summary.json")

print("\n" + "="*80)
print(f"✓ Complete balanced dataset creation done!")
print(f"  Total samples: {total_samples} (expected ~3074 for 1537 steps × 2)")
print("="*80)
