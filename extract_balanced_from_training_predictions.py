"""
Extract balanced training data from qwen2.5-7b-instruct predictions on training subset

This script:
1. Loads predictions from training sample indices (192 samples)
2. Extracts POSITIVE examples (correct tool calls from gold)
3. Extracts NEGATIVE examples (wrong tool calls from predictions)
4. Creates 50-50 balanced train/val/test splits
"""

import json
import glob
from pathlib import Path
from collections import defaultdict, Counter
import random

random.seed(42)


def extract_tool_calls_from_dialog(dialogs):
    """Extract tool calls from dialog sequence."""
    if dialogs and isinstance(dialogs[0], list):
        flat_dialogs = []
        for sublist in dialogs:
            if isinstance(sublist, list):
                flat_dialogs.extend(sublist)
        dialogs = flat_dialogs
    
    tool_calls = []
    context = []
    
    for msg in dialogs:
        if not isinstance(msg, dict):
            continue
        
        context.append(msg)
        
        if msg.get("role") == "assistant":
            if msg.get("tool_calls"):
                for tool_call in msg["tool_calls"]:
                    func = tool_call.get("function", {})
                    tool_calls.append({
                        "context": context.copy(),
                        "thought": msg.get("thought", ""),
                        "tool_name": func.get("name", ""),
                        "tool_arguments": func.get("arguments", {})
                    })
    
    return tool_calls


def extract_balanced_data_from_training_predictions(
    pred_dir: str,
    toolmeta_path: str,
    train_val_indices: list,
    output_dir: str,
    balance_ratio: float = 0.5  # 0.5 = 50% positive, 50% negative
):
    """
    Extract balanced training data from predictions on training indices.
    
    Args:
        pred_dir: Directory with prediction JSON files
        toolmeta_path: Path to toolmeta.json
        train_val_indices: List of sample indices to use for training
        output_dir: Where to save output
        balance_ratio: Target ratio of positive examples (0.5 = balanced)
    """
    print("="*80)
    print("Extracting Balanced Training Data from Training Subset Predictions")
    print("="*80)
    
    # Convert indices to set for fast lookup
    train_val_set = set(train_val_indices)
    
    # Load toolmeta
    with open(toolmeta_path) as f:
        toolmeta = json.load(f)
    
    # Find prediction files
    pred_files = sorted(Path(pred_dir).glob("ThinkGeo_bench_train_subset_*.json"))
    if not pred_files:
        # Try alternative naming
        pred_files = sorted(Path(pred_dir).glob("ThinkGeo_bench_*.json"))
    
    print(f"\nFound {len(pred_files)} prediction files in {pred_dir}")
    
    positive_examples = []
    negative_examples = []
    
    # Track statistics
    tool_correct_count = Counter()
    tool_wrong_count = Counter()
    error_patterns = Counter()
    
    for pred_file in pred_files:
        with open(pred_file) as f:
            predictions = json.load(f)
        
        for sample_id, sample_data in predictions.items():
            # Calculate global sample index
            file_num = int(pred_file.stem.split('_')[-1])
            qid_int = int(sample_id)
            global_sample_idx = file_num * 49 + qid_int
            
            # Only process training indices
            if global_sample_idx not in train_val_set:
                continue
            
            gold_dialogs = sample_data.get("gold", [])
            pred_dialogs = sample_data.get("prediction", [])
            origin_prompts = sample_data.get("origin_prompt", [])
            
            if not gold_dialogs:
                continue
            
            gold_tool_calls = extract_tool_calls_from_dialog(gold_dialogs)
            pred_tool_calls = extract_tool_calls_from_dialog(pred_dialogs) if pred_dialogs else []
            
            # Create mapping of step -> gold tool
            gold_tools_by_step = {}
            for step_idx, gold_call in enumerate(gold_tool_calls):
                gold_tools_by_step[step_idx] = gold_call["tool_name"]
                
                # Extract POSITIVE example (gold tool call)
                full_history = []
                if step_idx < len(origin_prompts):
                    history_msgs = origin_prompts[step_idx]
                    if isinstance(history_msgs, list):
                        full_history = history_msgs
                
                if not full_history:
                    full_history = gold_call["context"][-10:]
                
                tool_name = gold_call["tool_name"]
                tool_desc = toolmeta.get(tool_name, {}).get(
                    "description",
                    f"Tool: {tool_name}"
                )
                
                positive_example = {
                    "question_id": sample_id,
                    "step": step_idx,
                    "full_history": full_history,
                    "thought": gold_call["thought"],
                    "tool_name": tool_name,
                    "tool_description": tool_desc,
                    "tool_arguments": gold_call["tool_arguments"],
                    "gold_tool": tool_name,
                    "label": 1,
                    "global_sample_idx": global_sample_idx
                }
                positive_examples.append(positive_example)
                tool_correct_count[tool_name] += 1
            
            # Extract NEGATIVE examples (wrong predictions)
            for step_idx, pred_call in enumerate(pred_tool_calls):
                gold_tool = gold_tools_by_step.get(step_idx)
                pred_tool = pred_call["tool_name"]
                
                # Only add if wrong tool
                if gold_tool and pred_tool != gold_tool:
                    full_history = []
                    if step_idx < len(origin_prompts):
                        history_msgs = origin_prompts[step_idx]
                        if isinstance(history_msgs, list):
                            full_history = history_msgs
                    
                    if not full_history:
                        full_history = pred_call["context"][-10:]
                    
                    tool_desc = toolmeta.get(pred_tool, {}).get(
                        "description",
                        f"Tool: {pred_tool}"
                    )
                    
                    negative_example = {
                        "question_id": sample_id,
                        "step": step_idx,
                        "full_history": full_history,
                        "thought": pred_call["thought"],
                        "tool_name": pred_tool,
                        "tool_description": tool_desc,
                        "tool_arguments": pred_call["tool_arguments"],
                        "gold_tool": gold_tool,
                        "label": 0,
                        "global_sample_idx": global_sample_idx,
                        "error_type": f"{gold_tool}_mispredicted_as_{pred_tool}"
                    }
                    negative_examples.append(negative_example)
                    tool_wrong_count[pred_tool] += 1
                    error_patterns[f"{gold_tool} → {pred_tool}"] += 1
    
    print(f"\n{'='*80}")
    print("Extraction Results")
    print(f"{'='*80}")
    print(f"Positive examples (correct): {len(positive_examples)}")
    print(f"Negative examples (wrong): {len(negative_examples)}")
    print(f"Raw ratio: {len(positive_examples)/(len(negative_examples)+0.001):.2f}:1")
    
    print(f"\nTop error patterns:")
    for pattern, count in error_patterns.most_common(10):
        print(f"  {pattern}: {count}")
    
    print(f"\nTool distribution in CORRECT examples:")
    for tool, count in tool_correct_count.most_common():
        print(f"  {tool}: {count}")
    
    print(f"\nTool distribution in WRONG examples:")
    for tool, count in tool_wrong_count.most_common():
        print(f"  {tool}: {count}")
    
    # Balance the dataset
    print(f"\n{'='*80}")
    print(f"Creating Balanced Dataset (target ratio: {balance_ratio:.0%} positive)")
    print(f"{'='*80}")
    
    if balance_ratio == 0.5:
        # 50-50 balance
        target_positives = min(len(positive_examples), len(negative_examples))
        target_negatives = target_positives
    else:
        # Custom ratio
        total = len(positive_examples) + len(negative_examples)
        target_positives = int(total * balance_ratio)
        target_negatives = total - target_positives
        target_positives = min(target_positives, len(positive_examples))
        target_negatives = min(target_negatives, len(negative_examples))
    
    # Sample to balance
    sampled_positives = random.sample(positive_examples, target_positives)
    sampled_negatives = random.sample(negative_examples, target_negatives)
    
    print(f"Sampled {target_positives} positive, {target_negatives} negative")
    print(f"Final ratio: {target_positives/(target_negatives+0.001):.2f}:1")
    
    # Combine and split
    all_data = sampled_positives + sampled_negatives
    random.shuffle(all_data)
    
    # Split: 70% train, 15% val, 15% test
    total = len(all_data)
    train_size = int(total * 0.7)
    val_size = int(total * 0.15)
    
    train_data = all_data[:train_size]
    val_data = all_data[train_size:train_size + val_size]
    test_data = all_data[train_size + val_size:]
    
    print(f"\n{'='*80}")
    print("Dataset Splits")
    print(f"{'='*80}")
    print(f"Train: {len(train_data)} ({len([x for x in train_data if x['label']==1])} pos, {len([x for x in train_data if x['label']==0])} neg)")
    print(f"Val:   {len(val_data)} ({len([x for x in val_data if x['label']==1])} pos, {len([x for x in val_data if x['label']==0])} neg)")
    print(f"Test:  {len(test_data)} ({len([x for x in test_data if x['label']==1])} pos, {len([x for x in test_data if x['label']==0])} neg)")
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "train.json", 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(output_path / "val.json", 'w') as f:
        json.dump(val_data, f, indent=2)
    
    with open(output_path / "test.json", 'w') as f:
        json.dump(test_data, f, indent=2)
    
    # Save statistics
    stats = {
        "strategy": "Balanced 50-50 from qwen2.5-7b-instruct training subset predictions",
        "source_model": "qwen2.5-7b-instruct",
        "source_run": "latest inference on training indices",
        "num_training_indices": len(train_val_indices),
        "total_positive_raw": len(positive_examples),
        "total_negative_raw": len(negative_examples),
        "balance_ratio": balance_ratio,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "positive_tool_distribution": dict(tool_correct_count),
        "negative_tool_distribution": dict(tool_wrong_count),
        "top_error_patterns": dict(error_patterns.most_common(20))
    }
    
    with open(output_path / "dataset_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nSaved to {output_path}/")
    print(f"  - train.json")
    print(f"  - val.json")
    print(f"  - test.json")
    print(f"  - dataset_stats.json")
    
    print(f"\n{'='*80}")
    print("Next Steps")
    print(f"{'='*80}")
    print("1. Train classifier:")
    print(f"   python train_simple_classifier.py \\")
    print(f"     --train-file {output_path}/train.json \\")
    print(f"     --val-file {output_path}/val.json \\")
    print(f"     --test-file {output_path}/test.json \\")
    print(f"     --output-dir checkpoints_balanced \\")
    print(f"     --num-epochs 5")
    print()
    
    return train_data, val_data, test_data


if __name__ == "__main__":
    # Load training indices
    indices_file = Path("tool_choice_data_from_predictions/test_sample_indices.json")
    with open(indices_file) as f:
        indices_data = json.load(f)
        train_val_indices = indices_data["train_val_sample_indices"]
    
    print(f"Loaded {len(train_val_indices)} training sample indices")
    
    # IMPORTANT: Update this path after running inference!
    # Replace with your actual prediction directory
    PRED_DIR = "opencompass/outputs/default/YYYYMMDD_HHMMSS/predictions/qwen2.5-7b-instruct"
    
    # Check if directory exists
    if not Path(PRED_DIR).exists():
        print(f"\n{'!'*80}")
        print(f"ERROR: Prediction directory not found: {PRED_DIR}")
        print(f"{'!'*80}")
        print("\nPlease:")
        print("1. Run inference on training subset first:")
        print("   python run.py configs/eval_qwen2.5_7b_train_subset.py")
        print("\n2. Update PRED_DIR in this script to point to the output directory")
        print(f"{'!'*80}\n")
        exit(1)
    
    # Extract balanced data
    extract_balanced_data_from_training_predictions(
        pred_dir=PRED_DIR,
        toolmeta_path="opencompass/data/ThinkGeo_dataset/toolmeta.json",
        train_val_indices=train_val_indices,
        output_dir="tool_choice_data_balanced_from_training",
        balance_ratio=0.5  # 50-50 balance
    )
