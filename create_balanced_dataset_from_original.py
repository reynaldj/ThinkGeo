"""
Create a balanced dataset from the original tool_choice_data_from_predictions.
Uses the test_sample_indices.json to properly split into test and train/val,
then balances each split separately to maintain the original intended split.
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
print("CREATING BALANCED DATASET - PROPERLY SPLIT BY INDICES")
print("="*80)

# Load original datasets
original_train_file = '/home/james/ThinkGeo/tool_choice_data_from_predictions/train.json'
original_val_file = '/home/james/ThinkGeo/tool_choice_data_from_predictions/val.json'
original_test_file = '/home/james/ThinkGeo/tool_choice_data_from_predictions/test.json'

print("\n1. Loading original datasets...")
with open(original_train_file, 'r') as f:
    train_data = json.load(f)
print(f"   Train samples: {len(train_data)}")

with open(original_val_file, 'r') as f:
    val_data = json.load(f)
print(f"   Val samples: {len(val_data)}")

with open(original_test_file, 'r') as f:
    test_data = json.load(f)
print(f"   Test samples: {len(test_data)}")

# Combine all data
all_data = train_data + val_data + test_data
print(f"   Total samples: {len(all_data)}")

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

print(f"   Test split: {len(test_split)} samples (should be ~{len(test_sample_indices)*9} due to multiple steps)")
print(f"   Train/Val split: {len(train_val_split)} samples (should be ~{len(train_val_sample_indices)*9} due to multiple steps)")

# Separate positive and negative for TEST split
print("\n3. Balancing TEST split...")
test_positive = [s for s in test_split if s.get('label', 0) == 1]
test_negative = [s for s in test_split if s.get('label', 0) == 0]

print(f"   Positive: {len(test_positive)}, Negative: {len(test_negative)}")
test_min = min(len(test_positive), len(test_negative))
balanced_test = random.sample(test_positive, test_min) + random.sample(test_negative, test_min)
random.shuffle(balanced_test)
print(f"   Balanced test: {len(balanced_test)} samples (50/50 split)")

# Separate positive and negative for TRAIN/VAL split
print("\n4. Balancing TRAIN/VAL split...")
train_val_positive = [s for s in train_val_split if s.get('label', 0) == 1]
train_val_negative = [s for s in train_val_split if s.get('label', 0) == 0]

print(f"   Positive: {len(train_val_positive)}, Negative: {len(train_val_negative)}")
train_val_min = min(len(train_val_positive), len(train_val_negative))
balanced_train_val = random.sample(train_val_positive, train_val_min) + random.sample(train_val_negative, train_val_min)
random.shuffle(balanced_train_val)
print(f"   Balanced train/val: {len(balanced_train_val)} samples (50/50 split)")

# Further split train/val into train and val (split the 192 sample train_val indices)
# 70/30 split of train/val samples
print("\n5. Splitting train/val into train and val (70/30 within the train_val_sample_indices)...")
split_point = int(len(balanced_train_val) * 0.7)
balanced_train = balanced_train_val[:split_point]
balanced_val = balanced_train_val[split_point:]

print(f"   Train: {len(balanced_train)} samples")
print(f"   Val: {len(balanced_val)} samples")

# Verify balance
def count_labels(samples):
    positive = sum(1 for s in samples if s.get('label', 0) == 1)
    negative = len(samples) - positive
    return positive, negative

train_pos, train_neg = count_labels(balanced_train)
val_pos, val_neg = count_labels(balanced_val)
test_pos, test_neg = count_labels(balanced_test)

print("\n6. Verifying balance...")
print(f"   Train: {train_pos} positive ({train_pos/len(balanced_train)*100:.1f}%), {train_neg} negative ({train_neg/len(balanced_train)*100:.1f}%)")
print(f"   Val: {val_pos} positive ({val_pos/len(balanced_val)*100:.1f}%), {val_neg} negative ({val_neg/len(balanced_val)*100:.1f}%)")
print(f"   Test: {test_pos} positive ({test_pos/len(balanced_test)*100:.1f}%), {test_neg} negative ({test_neg/len(balanced_test)*100:.1f}%)")

# Save balanced datasets
output_dir = '/home/james/ThinkGeo/tool_choice_data_balanced_from_original'
os.makedirs(output_dir, exist_ok=True)

print(f"\n7. Saving balanced datasets to {output_dir}...")
with open(f'{output_dir}/train.json', 'w') as f:
    json.dump(balanced_train, f, indent=2)
print(f"   Saved train.json: {len(balanced_train)} samples")

with open(f'{output_dir}/val.json', 'w') as f:
    json.dump(balanced_val, f, indent=2)
print(f"   Saved val.json: {len(balanced_val)} samples")

with open(f'{output_dir}/test.json', 'w') as f:
    json.dump(balanced_test, f, indent=2)
print(f"   Saved test.json: {len(balanced_test)} samples")

# Save summary
total_samples = len(balanced_train) + len(balanced_val) + len(balanced_test)
summary = {
    "total_samples": total_samples,
    "positive_samples": train_pos + val_pos + test_pos,
    "negative_samples": train_neg + val_neg + test_neg,
    "train_size": len(balanced_train),
    "val_size": len(balanced_val),
    "test_size": len(balanced_test),
    "train_balance": {"positive": train_pos, "negative": train_neg},
    "val_balance": {"positive": val_pos, "negative": val_neg},
    "test_balance": {"positive": test_pos, "negative": test_neg},
    "source": "tool_choice_data_from_predictions (all 436 samples via indices)",
    "split_strategy": "Test set from test_sample_indices (244 samples), Train/Val from train_val_sample_indices (192 samples)",
    "overall_distribution": f"Train {len(balanced_train)/total_samples*100:.1f}%, Val {len(balanced_val)/total_samples*100:.1f}%, Test {len(balanced_test)/total_samples*100:.1f}%"
}

with open(f'{output_dir}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"   Saved summary.json")

print("\n" + "="*80)
print("✓ Balanced dataset creation complete!")
print("="*80)
