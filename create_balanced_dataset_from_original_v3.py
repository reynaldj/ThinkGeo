"""
Create a balanced dataset from the original tool_choice_data_from_predictions.
CORRECTED: Each step should appear only ONCE as positive (correct tool).
Create negatives by pairing steps with wrong tools.
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

print("="*80)
print("CREATING BALANCED DATASET - CORRECT VERSION")
print("="*80)

# Load original datasets
original_train_file = '/home/james/ThinkGeo/tool_choice_data_from_predictions/train.json'
original_val_file = '/home/james/ThinkGeo/tool_choice_data_from_predictions/val.json'
original_test_file = '/home/james/ThinkGeo/tool_choice_data_from_predictions/test.json'

print("\n1. Loading original datasets...")
with open(original_train_file, 'r') as f:
    train_data = json.load(f)
with open(original_val_file, 'r') as f:
    val_data = json.load(f)
with open(original_test_file, 'r') as f:
    test_data = json.load(f)

all_data = train_data + val_data + test_data
print(f"   Total samples loaded: {len(all_data)}")

# Split by indices
print("\n2. Splitting by test_sample_indices...")
test_split = []
train_val_split = []

for sample in all_data:
    sample_id = int(sample.get('global_sample_idx', -1))
    if sample_id in test_sample_indices:
        test_split.append(sample)
    elif sample_id in train_val_sample_indices:
        train_val_split.append(sample)

print(f"   Test split: {len(test_split)} samples")
print(f"   Train/Val split: {len(train_val_split)} samples")

# Key insight: Keep only POSITIVE samples (correct tools) and remove duplicates per step
print("\n3. Extracting unique positive samples per step...")

def extract_unique_positives(samples):
    """Keep only one positive sample per (question_id, step)"""
    seen_steps = {}
    for sample in samples:
        if sample.get('label') == 1:  # Only positive
            step_key = (sample.get('question_id'), sample.get('step'))
            if step_key not in seen_steps:
                seen_steps[step_key] = sample
    return list(seen_steps.values())

test_positives = extract_unique_positives(test_split)
train_val_positives = extract_unique_positives(train_val_split)

print(f"   Test unique positive steps: {len(test_positives)}")
print(f"   Train/Val unique positive steps: {len(train_val_positives)}")

# Collect all tools for creating negatives
all_tools = set()
for sample in all_data:
    tool = sample.get('tool_name')
    if tool:
        all_tools.add(tool)
all_tools = sorted(list(all_tools))
print(f"   Available tools: {len(all_tools)} - {all_tools}")

# Create negative samples by pairing each positive with a DIFFERENT tool
print("\n4. Creating negative samples (each positive paired with wrong tools)...")

def create_negatives(positives, tools, num_negatives_per_positive=2):
    """For each positive, create num_negatives_per_positive negative samples"""
    negatives = []
    for positive in positives:
        correct_tool = positive.get('tool_name')
        # Get wrong tools
        wrong_tools = [t for t in tools if t != correct_tool]
        if len(wrong_tools) > 0:
            # Sample wrong tools
            sampled_wrong = random.sample(wrong_tools, min(num_negatives_per_positive, len(wrong_tools)))
            for wrong_tool in sampled_wrong:
                # Create negative by changing tool_name and label
                negative = positive.copy()
                negative['tool_name'] = wrong_tool
                negative['label'] = 0
                negatives.append(negative)
    return negatives

# Create 1 negative per positive for balance
test_negatives = create_negatives(test_positives, all_tools, num_negatives_per_positive=1)
train_val_negatives = create_negatives(train_val_positives, all_tools, num_negatives_per_positive=1)

print(f"   Test positives: {len(test_positives)}, negatives created: {len(test_negatives)}")
print(f"   Train/Val positives: {len(train_val_positives)}, negatives created: {len(train_val_negatives)}")

# Combine and balance
test_samples = test_positives + test_negatives
train_val_samples = train_val_positives + train_val_negatives

random.shuffle(test_samples)
random.shuffle(train_val_samples)

print(f"   Total test balanced samples: {len(test_samples)}")
print(f"   Total train/val balanced samples: {len(train_val_samples)}")

# Split train/val into train and val (70/30)
print("\n5. Splitting train/val into train and val (70/30)...")
split_point = int(len(train_val_samples) * 0.7)
train_samples = train_val_samples[:split_point]
val_samples = train_val_samples[split_point:]

print(f"   Train: {len(train_samples)} samples")
print(f"   Val: {len(val_samples)} samples")

# Verify balance
def count_labels(samples):
    positive = sum(1 for s in samples if s.get('label') == 1)
    negative = len(samples) - positive
    return positive, negative

train_pos, train_neg = count_labels(train_samples)
val_pos, val_neg = count_labels(val_samples)
test_pos, test_neg = count_labels(test_samples)

print("\n6. Verifying balance...")
print(f"   Train: {train_pos} positive (50% expected), {train_neg} negative (50% expected) - Balance: {train_pos/len(train_samples)*100:.1f}% pos")
print(f"   Val: {val_pos} positive (50% expected), {val_neg} negative (50% expected) - Balance: {val_pos/len(val_samples)*100:.1f}% pos")
print(f"   Test: {test_pos} positive (50% expected), {test_neg} negative (50% expected) - Balance: {test_pos/len(test_samples)*100:.1f}% pos")

# Verify unique steps
def count_unique_steps(samples):
    return len(set((s.get('question_id'), s.get('step')) for s in samples))

print("\n7. Verifying unique steps...")
print(f"   Train unique steps: {count_unique_steps(train_samples)}")
print(f"   Val unique steps: {count_unique_steps(val_samples)}")
print(f"   Test unique steps: {count_unique_steps(test_samples)}")

# Analyze tool distribution
print("\n8. Tool distribution in test set...")
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

test_tools = get_tool_dist(test_samples)
for tool in sorted(test_tools.keys()):
    counts = test_tools[tool]
    total = counts['positive'] + counts['negative']
    print(f"   {tool}: {counts['positive']} pos, {counts['negative']} neg (total: {total})")

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
train_tools = get_tool_dist(train_samples)
val_tools = get_tool_dist(val_samples)
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
    "train_unique_steps": count_unique_steps(train_samples),
    "val_unique_steps": count_unique_steps(val_samples),
    "test_unique_steps": count_unique_steps(test_samples),
    "source": "tool_choice_data_from_predictions (all 436 samples via indices)",
    "split_strategy": "Test set from test_sample_indices (244 samples), Train/Val from train_val_sample_indices (192 samples)",
    "balancing_strategy": "Each unique step has 1 positive (correct tool) + 1 negative (wrong tool). 50/50 balance.",
    "tool_distribution_train": train_tools,
    "tool_distribution_val": val_tools,
    "tool_distribution_test": test_tools
}

with open(f'{output_dir}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"    Saved summary.json")

print("\n" + "="*80)
print("✓ Balanced dataset creation complete!")
print("="*80)
