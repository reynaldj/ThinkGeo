#!/usr/bin/env python3
"""
Alternative: Direct Hugging Face Trainer-based SFT (no external dependencies).

This uses the standard transformers Trainer for supervised finetuning.
More control than LLaMA-Factory but requires more setup.

Prerequisites:
  pip install transformers datasets peft accelerate bitsandbytes

Usage:
  python scripts/train_sft_hf.py \
      --model Qwen/Qwen2.5-7B-Instruct \
      --train-data data/sft_phase1_train.jsonl \
      --val-data data/sft_phase1_val.jsonl \
      --output models/qwen2.5-7b-sft-phase1 \
      --epochs 2 \
      --batch-size 4
"""
import argparse
import json
import torch
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    with open(path) as f:
        return [json.loads(line) for line in f]


def format_messages_for_training(messages: List[Dict], tokenizer) -> str:
    """
    Format conversation messages using the tokenizer's chat template.
    
    For Qwen2.5, this applies the proper chat formatting including special tokens.
    """
    # Use tokenizer's apply_chat_template
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return formatted


def preprocess_function(examples: Dict, tokenizer, max_length: int = 2048):
    """
    Tokenize training examples.
    
    Args:
        examples: Batch of examples with 'messages' field
        tokenizer: Tokenizer
        max_length: Max sequence length
    
    Returns:
        Dict with input_ids and labels
    """
    texts = []
    for messages in examples["messages"]:
        text = format_messages_for_training(messages, tokenizer)
        texts.append(text)
    
    # Tokenize
    model_inputs = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None,
    )
    
    # Labels = input_ids (standard causal LM training)
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    
    return model_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--gradient-accum", type=int, default=4)
    args = parser.parse_args()
    
    # Load tokenizer and model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        padding_side="right",
    )
    
    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Setup LoRA
    print("Setting up LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load data
    print(f"Loading train data: {args.train_data}")
    train_data = load_jsonl(args.train_data)
    print(f"Loading val data: {args.val_data}")
    val_data = load_jsonl(args.val_data)
    
    # Create datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # Tokenize
    print("Tokenizing...")
    train_dataset = train_dataset.map(
        lambda ex: preprocess_function(ex, tokenizer, args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
    )
    val_dataset = val_dataset.map(
        lambda ex: preprocess_function(ex, tokenizer, args.max_length),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing val",
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=["tensorboard"],
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print(f"Saving model to {args.output}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output)
    
    print("Training complete!")


if __name__ == "__main__":
    main()
