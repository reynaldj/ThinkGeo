#!/usr/bin/env python3
"""
Create balanced dataset with SCHEMA-AWARE inputs and CountGivenObject-first instances in training.

Key changes:
1. Include argument schema (parameter names) in dataset with MASKED VALUES
2. Transfer 12 CountGivenObject first-step samples from test to train
3. Maintain proper splits with adjusted indices

Input format now includes:
  [CONTEXT] query
  [HISTORY] tool_calls
  [TOOL_CALLED] name: description
  [SCHEMA] image=<PATH>, text=<MASKED>, bbox=<BBOX>  # Parameter names visible, values masked
"""

import json
import random
import os
from collections import defaultdict
import glob

random.seed(42)

print("="*80)
print("CREATING BALANCED DATASET WITH SCHEMA-AWARE INPUTS")
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

print(f"   Found {len(prediction_dirs)} prediction directories")

# Build maps of observed wrong tools
wrong_tools_map = defaultdict(set)
wrong_tools_by_question = defaultdict(lambda: defaultdict(set))

for pred_dir in prediction_dirs:
    pred_files = glob.glob(f"{pred_dir}/ThinkGeo_bench_*.json")
    
    for pred_file in pred_files:
        try:
            with open(pred_file, 'r') as f:
                preds = json.load(f)
            
            for q_id_str, entry in preds.items():
                q_id = int(q_id_str)
                
                if 'prediction' not in entry:
                    continue
                
                prediction_steps = entry.get('prediction', [])
                gold_steps = entry.get('gold', [])
                
                for step_idx in range(min(len(prediction_steps), len(gold_steps))):
                    pred_step = prediction_steps[step_idx]
                    gold_step = gold_steps[step_idx]
                    
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
                    
                    if pred_tool and gold_tool and pred_tool != gold_tool:
                        wrong_tools_map[(q_id, step_idx)].add(pred_tool)
                        wrong_tools_by_question[q_id][step_idx].add(pred_tool)
        except:
            pass

print(f"   Built wrong tools maps from {len(prediction_dirs)} prediction directories")

# Define schema building function
def _build_schema(tool_name, tool_args):
    """Build argument schema with parameter names but masked values."""
    if tool_name == 'FinishAction':
        return {'answer': '<MASKED>'}
    
    # Define expected arguments for each tool
    tool_schemas = {
        'TextToBbox': {'image': '<PATH>', 'text': '<MASKED>', 'top1': '<BOOL>'},
        'RegionAttributeDescription': {'image': '<PATH>', 'bbox': '<BBOX>', 'attribute': '<MASKED>'},
        'CountGivenObject': {'image': '<PATH>', 'text': '<MASKED>', 'bbox': '<BBOX_OPT>'},
        'DrawBox': {'image': '<PATH>', 'bbox': '<BBOX>', 'text': '<MASKED>', 'color': '<MASKED>'},
        'Calculator': {'expression': '<MASKED>'},
        'Solver': {'formula': '<MASKED>', 'unit': '<MASKED>'},
        'Plot': {'command': '<CODE>'},
        'GoogleSearch': {'query': '<MASKED>'},
        'ImageDescription': {'image': '<PATH>'},
        'OCR': {'image': '<PATH>', 'bbox': '<BBOX_OPT>'},
        'ObjectDetection': {'image': '<PATH>', 'text': '<MASKED>'},
        'ChangeDetection': {'image1': '<PATH>', 'image2': '<PATH>'},
        'SegmentObjectPixels': {'image': '<PATH>', 'text': '<MASKED>'},
        'AddText': {'image': '<PATH>', 'text': '<MASKED>', 'bbox': '<BBOX>', 'font_size': '<INT>'},
    }
    
    schema = tool_schemas.get(tool_name, {})
    
    # Filter to only include provided arguments in schema
    filtered_schema = {}
    for key in tool_args.keys():
        if key in schema:
            filtered_schema[key] = schema[key]
        else:
            filtered_schema[key] = '<VALUE>'
    
    return filtered_schema

# Extract all steps
print("\n2. Extracting steps from ThinkGeoBench...")
all_steps = []

for question_id, question_data in bench_data.items():
    if not isinstance(question_data, dict) or not question_data.get('dialogs'):
        continue
    
    question_id_int = int(question_id)
    dialogs = question_data['dialogs']
    step_num = 0
    accumulated_history = []
    
    for dialog in dialogs:
        if not isinstance(dialog, dict):
            continue
        
        role = dialog.get('role')
        
        # Tool call step
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
                        'argument_schema': _build_schema(tool_name, tool_args),  # NEW: schema info
                        'full_history': list(accumulated_history),
                        'thought': dialog.get('thought', ''),
                        'label': 1,
                        'wrong_tools_from_runs': list(wrong_tools_map.get((question_id_int, step_num), set()))
                    }
                    all_steps.append(step_data)
                    step_num += 1
        
        # Answer step
        elif role == 'assistant' and not dialog.get('tool_calls'):
            step_data = {
                'question_id': question_id,
                'global_sample_idx': question_id_int,
                'step': step_num,
                'tool_name': 'FinishAction',
                'gold_tool': 'FinishAction',
                'tool_description': tool_descriptions.get('FinishAction', 'Final answer to the user query'),
                'tool_arguments': {'answer': dialog.get('content', '')},
                'argument_schema': {'answer': '<MASKED>'},  # NEW: schema info
                'full_history': list(accumulated_history),
                'thought': dialog.get('thought', ''),
                'label': 1,
                'wrong_tools_from_runs': list(wrong_tools_map.get((question_id_int, step_num), set()))
            }
            all_steps.append(step_data)
            step_num += 1
        
        accumulated_history.append(dialog)

print(f"   Total steps extracted: {len(all_steps)}")

# Split by indices
print("\n3. Splitting by test/train/val sample indices...")
test_steps = [s for s in all_steps if s['global_sample_idx'] in test_sample_indices]
train_val_steps = [s for s in all_steps if s['global_sample_idx'] in train_val_sample_indices]

print(f"   Test steps (before transfer): {len(test_steps)}")
print(f"   Train/Val steps (before transfer): {len(train_val_steps)}")

# TRANSFER: Move CountGivenObject first-step instances from test to train
print("\n4. Transferring CountGivenObject first-step instances from test to train...")
count_first_in_test = [s for s in test_steps if s['step'] == 0 and s['tool_name'] == 'CountGivenObject']
count_to_transfer = count_first_in_test[:12]  # Transfer 12 instances

print(f"   Found {len(count_first_in_test)} CountGivenObject first-step in test")
print(f"   Transferring {len(count_to_transfer)} to training")

# Remove transferred from test
test_steps = [s for s in test_steps if s not in count_to_transfer]

# Add to train/val
train_val_steps.extend(count_to_transfer)

print(f"   Test steps (after transfer): {len(test_steps)}")
print(f"   Train/Val steps (after transfer): {len(train_val_steps)}")

# Split train/val 70/30
print("\n5. Splitting train/val 70/30 by question indices...")
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
print(f"\n6. Available tools ({len(all_tools)}): {all_tools}")

# Create balanced datasets
print("\n7. Creating negative samples with realistic wrong tools...")

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

    exact_wrong_tools = positive_sample.get('wrong_tools_from_runs', [])
    if exact_wrong_tools:
        wrong_tool = random.choice(exact_wrong_tools)
        tier = 'exact'
    else:
        nearby_wrong_tools = get_nearby_wrong_tools(question_id, step_num)
        if nearby_wrong_tools:
            wrong_tool = random.choice(nearby_wrong_tools)
            tier = 'nearby'
        else:
            wrong_tools = [t for t in all_tools_list if t != gold_tool]
            if len(wrong_tools) > 0:
                wrong_tool = random.choice(wrong_tools)
                tier = 'random'
            else:
                return None
    
    negative = positive_sample.copy()
    negative['tool_name'] = wrong_tool
    negative['tool_description'] = tool_descriptions.get(wrong_tool, '')
    negative['argument_schema'] = _build_schema(wrong_tool, {})  # Empty args for negative
    negative['label'] = 0
    return negative

def _build_schema(tool_name, tool_args):
    """Build argument schema with parameter names but masked values."""
    if tool_name == 'FinishAction':
        return {'answer': '<MASKED>'}
    
    # Define expected arguments for each tool
    tool_schemas = {
        'TextToBbox': {'image': '<PATH>', 'text': '<MASKED>', 'top1': '<BOOL>'},
        'RegionAttributeDescription': {'image': '<PATH>', 'bbox': '<BBOX>', 'attribute': '<MASKED>'},
        'CountGivenObject': {'image': '<PATH>', 'text': '<MASKED>', 'bbox': '<BBOX_OPT>'},
        'DrawBox': {'image': '<PATH>', 'bbox': '<BBOX>', 'text': '<MASKED>', 'color': '<MASKED>'},
        'Calculator': {'expression': '<MASKED>'},
        'Solver': {'formula': '<MASKED>', 'unit': '<MASKED>'},
        'Plot': {'command': '<CODE>'},
        'GoogleSearch': {'query': '<MASKED>'},
        'ImageDescription': {'image': '<PATH>'},
        'OCR': {'image': '<PATH>', 'bbox': '<BBOX_OPT>'},
        'ObjectDetection': {'image': '<PATH>', 'text': '<MASKED>'},
        'ChangeDetection': {'image1': '<PATH>', 'image2': '<PATH>'},
        'SegmentObjectPixels': {'image': '<PATH>', 'text': '<MASKED>'},
        'AddText': {'image': '<PATH>', 'text': '<MASKED>', 'bbox': '<BBOX>', 'font_size': '<INT>'},
    }
    
    schema = tool_schemas.get(tool_name, {})
    
    # Filter to only include provided arguments in schema
    filtered_schema = {}
    for key in tool_args.keys():
        if key in schema:
            filtered_schema[key] = schema[key]
        else:
            filtered_schema[key] = '<VALUE>'
    
    return filtered_schema

test_negatives = []
for positive in test_steps:
    neg = create_negative_sample(positive, all_tools)
    if neg:
        test_negatives.append(neg)

train_negatives = []
for positive in train_steps:
    neg = create_negative_sample(positive, all_tools)
    if neg:
        train_negatives.append(neg)

val_negatives = []
for positive in val_steps:
    neg = create_negative_sample(positive, all_tools)
    if neg:
        val_negatives.append(neg)

print(f"   Test negatives: {len(test_negatives)}")
print(f"   Train negatives: {len(train_negatives)}")
print(f"   Val negatives: {len(val_negatives)}")

# Combine and shuffle
test_samples = test_steps + test_negatives
train_samples = train_steps + train_negatives
val_samples = val_steps + val_negatives

random.shuffle(test_samples)
random.shuffle(train_samples)
random.shuffle(val_samples)

print(f"\n8. Final balanced datasets:")
print(f"   Test: {len(test_samples)} samples ({len(test_steps)} pos, {len(test_negatives)} neg)")
print(f"   Train: {len(train_samples)} samples ({len(train_steps)} pos, {len(train_negatives)} neg)")
print(f"   Val: {len(val_samples)} samples ({len(val_steps)} pos, {len(val_negatives)} neg)")

# Verify CountGivenObject first-step in training
count_first_in_new_train = [s for s in train_samples if s['step'] == 0 and s['tool_name'] == 'CountGivenObject' and s['label'] == 1]
print(f"\n9. Verification - CountGivenObject first-step in training: {len(count_first_in_new_train)} (was 6, now should be ~18)")

# Save datasets
output_dir = '/home/james/ThinkGeo/tool_choice_data_schema_aware'
os.makedirs(output_dir, exist_ok=True)

print(f"\n10. Saving datasets to {output_dir}...")

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
total_pos = len(train_steps) + len(val_steps) + len(test_steps)
total_neg = len(train_negatives) + len(val_negatives) + len(test_negatives)

summary = {
    "total_steps_extracted": len(all_steps),
    "test_steps": len(test_steps),
    "train_steps": len(train_steps),
    "val_steps": len(val_steps),
    "test_balanced_samples": len(test_samples),
    "train_balanced_samples": len(train_samples),
    "val_balanced_samples": len(val_samples),
    "transfer_note": f"Moved {len(count_to_transfer)} CountGivenObject first-step instances from test to train",
    "countgivenobject_first_step_train": len(count_first_in_new_train),
    "all_tools": all_tools,
    "negatives_strategy": "Tiered: exact-step wrong tools, then nearby-step wrong tools, then same-gold-tool wrong tools across questions, then random",
    "prediction_sources": len(prediction_dirs),
    "schema_aware": True,
    "schema_format": "Parameter names visible with <TYPE> masks for values (e.g., image=<PATH>, text=<MASKED>, bbox=<BBOX>)"
}

with open(f'{output_dir}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"    Saved summary.json")

print("\n" + "="*80)
print(f"✓ Schema-aware balanced dataset creation complete!")
print(f"  Total samples: {total_samples} ({total_pos} positive, {total_neg} negative)")
print(f"  CountGivenObject first-step in training: {len(count_first_in_new_train)}")
print(f"  Schema format: Parameter names visible, values masked")
print("="*80)
