#!/usr/bin/env python3
"""
Extract gold tool calls from ThinkGeoBench for supervised finetuning.

Output format: JSONL with one training example per line.
Each example contains the full conversation up to a tool call, 
teaching the model to generate exact gold arguments.

Usage:
  python scripts/prepare_sft_data.py \
      --input opencompass/data/ThinkGeo_dataset/ThinkGeoBench.json \
      --output data/sft_train.jsonl \
      --val-split 0.1
"""
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any


def build_conversation_history(dialogs: List[Dict], up_to_step: int) -> List[Dict]:
    """Build conversation history up to (but not including) the target step."""
    history = []
    for i, step in enumerate(dialogs[:up_to_step]):
        role = step.get("role", "assistant")
        
        if role == "user":
            history.append({"role": "user", "content": step.get("content", "")})
        elif role == "tool":
            # Tool responses
            tool_name = step.get("name", "")
            content = step.get("content", {})
            if isinstance(content, dict):
                content_text = content.get("content", str(content))
            else:
                content_text = str(content)
            history.append({
                "role": "tool",
                "name": tool_name,
                "content": content_text
            })
        elif role == "assistant":
            # Assistant turns (with or without tool calls)
            msg = {"role": "assistant"}
            if "thought" in step:
                msg["thought"] = step["thought"]
            if "tool_calls" in step and step["tool_calls"]:
                msg["tool_calls"] = step["tool_calls"]
            if "content" in step:
                msg["content"] = step["content"]
            if msg.get("thought") or msg.get("tool_calls") or msg.get("content"):
                history.append(msg)
    
    return history


def extract_training_examples(data: Dict[str, Dict], focus_tools: List[str] = None) -> List[Dict]:
    """
    Extract supervised training examples from ThinkGeoBench.
    
    Each example is a conversation leading up to a tool call, with the gold
    tool call as the target output.
    
    Args:
        data: ThinkGeoBench samples (dict with sample IDs as keys)
        focus_tools: Optional list of tool names to prioritize (for balancing)
    
    Returns:
        List of training examples in chat format
    """
    examples = []
    
    for sample_id, sample in data.items():
        dialogs = sample.get("dialogs", [])
        
        # Find all assistant steps with tool calls
        for step_idx, step in enumerate(dialogs):
            if step.get("role") != "assistant":
                continue
            
            tool_calls = step.get("tool_calls")
            if not tool_calls:
                continue
            
            # Extract the first tool call (most samples have one per step)
            tool_call = tool_calls[0]
            func = tool_call.get("function", {})
            tool_name = func.get("name")
            args = func.get("arguments", {})
            
            # Skip if focusing on specific tools and this isn't one
            if focus_tools and tool_name not in focus_tools:
                continue
            
            # Build conversation history up to this step
            history = build_conversation_history(dialogs, step_idx)
            
            # Create target assistant turn with gold tool call
            target = {"role": "assistant"}
            if "thought" in step:
                target["thought"] = step["thought"]
            target["tool_calls"] = [tool_call]
            
            # Full training example
            examples.append({
                "messages": history + [target],
                "metadata": {
                    "sample_id": sample_id,
                    "step_idx": step_idx,
                    "tool": tool_name
                }
            })
    
    return examples


def balance_by_tool(examples: List[Dict], max_per_tool: int = 300) -> List[Dict]:
    """
    Balance dataset to avoid over-representation of common tools.
    
    Args:
        examples: All training examples
        max_per_tool: Maximum examples per tool
    
    Returns:
        Balanced subset
    """
    by_tool = {}
    for ex in examples:
        tool = ex["metadata"]["tool"]
        if tool not in by_tool:
            by_tool[tool] = []
        by_tool[tool].append(ex)
    
    balanced = []
    for tool, tool_examples in by_tool.items():
        sampled = random.sample(tool_examples, min(len(tool_examples), max_per_tool))
        balanced.extend(sampled)
        print(f"  {tool}: {len(sampled)} examples (from {len(tool_examples)} total)")
    
    return balanced


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT data from ThinkGeoBench")
    parser.add_argument("--input", required=True, help="Path to ThinkGeoBench.json")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--train-samples", type=int, default=200, help="Number of samples for training")
    parser.add_argument("--val-samples", type=int, default=36, help="Number of samples for validation")
    parser.add_argument("--test-samples", type=int, default=200, help="Number of samples for testing")
    parser.add_argument("--max-per-tool", type=int, default=300, help="Max examples per tool (for balancing within each split)")
    parser.add_argument("--focus-tools", nargs="*", help="Optional: focus on specific tools")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Load ThinkGeoBench
    print(f"Loading {args.input}...")
    data = json.loads(Path(args.input).read_text())
    all_sample_ids = list(data.keys())
    print(f"Loaded {len(all_sample_ids)} samples")
    
    # Validate split sizes
    total_requested = args.train_samples + args.val_samples + args.test_samples
    if total_requested > len(all_sample_ids):
        print(f"Warning: Requested {total_requested} samples but only {len(all_sample_ids)} available")
        print(f"Adjusting to use all available samples...")
    
    # Shuffle sample IDs
    random.shuffle(all_sample_ids)
    
    # Split at sample level
    train_ids = all_sample_ids[:args.train_samples]
    val_ids = all_sample_ids[args.train_samples:args.train_samples + args.val_samples]
    test_ids = all_sample_ids[args.train_samples + args.val_samples:args.train_samples + args.val_samples + args.test_samples]
    
    print(f"\nSample-level split:")
    print(f"  Train: {len(train_ids)} samples")
    print(f"  Val:   {len(val_ids)} samples")
    print(f"  Test:  {len(test_ids)} samples")
    
    # Extract examples from each split
    focus_tools = args.focus_tools if args.focus_tools else None
    
    print("\nExtracting training examples...")
    train_data = {sid: data[sid] for sid in train_ids}
    train_examples = extract_training_examples(train_data, focus_tools=focus_tools)
    print(f"  Raw train examples: {len(train_examples)}")
    
    print("Extracting validation examples...")
    val_data = {sid: data[sid] for sid in val_ids}
    val_examples = extract_training_examples(val_data, focus_tools=focus_tools)
    print(f"  Raw val examples: {len(val_examples)}")
    
    print("Extracting test examples...")
    test_data = {sid: data[sid] for sid in test_ids}
    test_examples = extract_training_examples(test_data, focus_tools=focus_tools)
    print(f"  Raw test examples: {len(test_examples)}")
    
    # Balance each split independently
    print("\nBalancing training set...")
    train_balanced = balance_by_tool(train_examples, max_per_tool=args.max_per_tool)
    
    print("\nBalancing validation set...")
    val_balanced = balance_by_tool(val_examples, max_per_tool=args.max_per_tool)
    
    print("\nBalancing test set...")
    test_balanced = balance_by_tool(test_examples, max_per_tool=args.max_per_tool)
    
    print(f"\nFinal counts:")
    print(f"  Train: {len(train_balanced)} examples")
    print(f"  Val:   {len(val_balanced)} examples")
    print(f"  Test:  {len(test_balanced)} examples")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    train_path = output_path.parent / (output_path.stem + "_train.jsonl")
    val_path = output_path.parent / (output_path.stem + "_val.jsonl")
    test_path = output_path.parent / (output_path.stem + "_test.jsonl")
    
    with open(train_path, "w") as f:
        for ex in train_balanced:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    with open(val_path, "w") as f:
        for ex in val_balanced:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    with open(test_path, "w") as f:
        for ex in test_balanced:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    # Save test sample IDs for evaluation
    test_ids_path = output_path.parent / (output_path.stem + "_test_sample_ids.json")
    with open(test_ids_path, "w") as f:
        json.dump(test_ids, f, indent=2)
    
    print(f"\nSaved:")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    print(f"  Test:  {test_path}")
    print(f"  Test IDs: {test_ids_path}")
    
    # Print sample
    print("\n" + "="*80)
    print("SAMPLE TRAINING EXAMPLE:")
    print("="*80)
    sample = train_balanced[0]
    print(json.dumps(sample, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
