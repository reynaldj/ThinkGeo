"""
Create a balanced dataset from the original tool_choice_data_from_predictions.
Uses the test_sample_indices.json to properly split into test and train/val.
Ensures test and train have SAME tool distributions and reduces NoAction samples.
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
print("CREATING BALANCED DATASET - MATCHING TRAIN/TEST DISTRIBUTIONS")
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

# Analyze tool distribution in train/val split first
print("\n3. Analyzing tool distribution in train/val...")
train_val_by_tool = defaultdict(lambda: {'positive': [], 'negative': []})

for sample in train_val_split:
    tool_name = sample.get('tool_name', 'Unknown')
    label = sample.get('label', 0)
    
    if label == 1:
        train_val_by_tool[tool_name]['positive'].append(sample)
    else:
        train_val_by_tool[tool_name]['negative'].append(sample)

# Count tool distribution
train_val_tool_counts = {}
for tool, samples_dict in train_val_by_tool.items():
    pos_count = len(samples_dict['positive'])
    neg_count = len(samples_dict['negative'])
    train_val_tool_counts[tool] = {'positive': pos_count, 'negative': neg_count}
    print(f"   {tool}: {pos_count} pos, {neg_count} neg")

# Now balance train/val first
print("\n4. Balancing TRAIN/VAL split per tool...")
balanced_train_val_samples = []

for tool, samples_dict in train_val_by_tool.items():
    pos_samples = samples_dict['positive']
    neg_samples = samples_dict['negative']
    
    # Balance positive and negative within each tool
    min_count = min(len(pos_samples), len(neg_samples))
    if min_count > 0:
        balanced_tool_samples = random.sample(pos_samples, min_count) + random.sample(neg_samples, min_count)
        balanced_train_val_samples.extend(balanced_tool_samples)
        print(f"   {tool}: {min_count} pos + {min_count} neg = {min_count*2} samples")

random.shuffle(balanced_train_val_samples)
print(f"   Total balanced train/val: {len(balanced_train_val_samples)} samples")

# Now process test split - use SAME tool distribution as train/val
print("\n5. Balancing TEST split with SAME tool distribution as train/val...")
test_by_tool = defaultdict(lambda: {'positive': [], 'negative': []})

for sample in test_split:
    tool_name = sample.get('tool_name', 'Unknown')
    label = sample.get('label', 0)
    
    if label == 1:
        test_by_tool[tool_name]['positive'].append(sample)
    else:
        test_by_tool[tool_name]['negative'].append(sample)

# For test, use the SAME per-tool sample counts as train/val
balanced_test_samples = []

for tool, target_count_dict in train_val_tool_counts.items():
    target_pos = target_count_dict['positive']
    target_neg = target_count_dict['negative']
    
    if tool not in test_by_tool:
        print(f"   Warning: {tool} not found in test set")
        continue
    
    test_pos_samples = test_by_tool[tool]['positive']
    test_neg_samples = test_by_tool[tool]['negative']
    
    # Sample exactly target_pos and target_neg from test (with replacement if needed)
    actual_pos = min(target_pos, len(test_pos_samples))
    actual_neg = min(target_neg, len(test_neg_samples))
    
    if actual_pos > 0:
        sampled_pos = random.sample(test_pos_samples, actual_pos)
    else:
        sampled_pos = []
    
    if actual_neg > 0:
        sampled_neg = random.sample(test_neg_samples, actual_neg)
    else:
        sampled_neg = []
    
    balanced_test_samples.extend(sampled_pos + sampled_neg)
    print(f"   {tool}: {actual_pos} pos + {actual_neg} neg = {actual_pos + actual_neg} samples (target was {target_pos} pos, {target_neg} neg)")

# Reduce NoAction if present
print("\n6. Reducing NoAction samples...")
noaction_samples = [s for s in balanced_test_samples if s.get('tool_name') == 'NoAction']
other_samples = [s for s in balanced_test_samples if s.get('tool_name') != 'NoAction']

if len(noaction_samples) > 500:
    print(f"   NoAction before: {len(noaction_samples)}")
    noaction_samples = random.sample(noaction_samples, 500)
    print(f"   NoAction after: {len(noaction_samples)}")

balanced_test_samples = other_samples + noaction_samples
random.shuffle(balanced_test_samples)
print(f"   Total test after NoAction reduction: {len(balanced_test_samples)}")

# Split balanced train/val into train and val (70/30)
print("\n7. Splitting train/val into train and val (70/30)...")
split_point = int(len(balanced_train_val_samples) * 0.7)
balanced_train = balanced_train_val_samples[:split_point]
balanced_val = balanced_train_val_samples[split_point:]

print(f"   Train: {len(balanced_train)} samples")
print(f"   Val: {len(balanced_val)} samples")

# Verify balance
def count_labels(samples):
    positive = sum(1 for s in samples if s.get('label', 0) == 1)
    negative = len(samples) - positive
    return positive, negative

train_pos, train_neg = count_labels(balanced_train)
val_pos, val_neg = count_labels(balanced_val)
test_pos, test_neg = count_labels(balanced_test_samples)

print("\n8. Verifying balance...")
print(f"   Train: {train_pos} positive ({train_pos/len(balanced_train)*100:.1f}%), {train_neg} negative ({train_neg/len(balanced_train)*100:.1f}%)")
print(f"   Val: {val_pos} positive ({val_pos/len(balanced_val)*100:.1f}%), {val_neg} negative ({val_neg/len(balanced_val)*100:.1f}%)")
print(f"   Test: {test_pos} positive ({test_pos/len(balanced_test_samples)*100:.1f}%), {test_neg} negative ({test_neg/len(balanced_test_samples)*100:.1f}%)")

# Analyze and verify tool distributions match
print("\n9. Verifying tool distributions match...")
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

train_tools = get_tool_dist(balanced_train)
test_tools = get_tool_dist(balanced_test_samples)

print("   Train vs Test distribution:")
all_tools = set(train_tools.keys()) | set(test_tools.keys())
for tool in sorted(all_tools):
    train_info = train_tools.get(tool, {'positive': 0, 'negative': 0})
    test_info = test_tools.get(tool, {'positive': 0, 'negative': 0})
    print(f"   {tool}:")
    print(f"      Train: {train_info['positive']} pos, {train_info['negative']} neg")
    print(f"      Test:  {test_info['positive']} pos, {test_info['negative']} neg")

# Save balanced datasets
output_dir = '/home/james/ThinkGeo/tool_choice_data_balanced_from_original'
os.makedirs(output_dir, exist_ok=True)

print(f"\n10. Saving balanced datasets to {output_dir}...")
with open(f'{output_dir}/train.json', 'w') as f:
    json.dump(balanced_train, f, indent=2)
print(f"    Saved train.json: {len(balanced_train)} samples")

with open(f'{output_dir}/val.json', 'w') as f:
    json.dump(balanced_val, f, indent=2)
print(f"    Saved val.json: {len(balanced_val)} samples")

with open(f'{output_dir}/test.json', 'w') as f:
    json.dump(balanced_test_samples, f, indent=2)
print(f"    Saved test.json: {len(balanced_test_samples)} samples")

# Save summary with per-tool distributions
val_tools = get_tool_dist(balanced_val)
total_samples = len(balanced_train) + len(balanced_val) + len(balanced_test_samples)

summary = {
    "total_samples": total_samples,
    "positive_samples": train_pos + val_pos + test_pos,
    "negative_samples": train_neg + val_neg + test_neg,
    "train_size": len(balanced_train),
    "val_size": len(balanced_val),
    "test_size": len(balanced_test_samples),
    "train_balance": {"positive": train_pos, "negative": train_neg},
    "val_balance": {"positive": val_pos, "negative": val_neg},
    "test_balance": {"positive": test_pos, "negative": test_neg},
    "source": "tool_choice_data_from_predictions (all 436 samples via indices)",
    "split_strategy": "Test set from test_sample_indices (244 samples), Train/Val from train_val_sample_indices (192 samples)",
    "distribution_note": "Test and Train have SAME tool distributions. NoAction reduced to ~500 samples.",
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
