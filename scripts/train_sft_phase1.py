#!/usr/bin/env python3
"""
Supervised finetuning script for Phase 1: teach exact gold argument generation.

Uses LLaMA-Factory for simplicity. Finetunes Qwen2.5-7B with LoRA on gold tool calls.

Prerequisites:
  pip install llamafactory transformers datasets peft accelerate

Usage:
  python scripts/train_sft_phase1.py --config configs/sft_phase1.yaml
"""
import os
import sys
import yaml
import subprocess
from pathlib import Path


def create_llamafactory_config(
    model_name: str,
    data_dir: str,
    output_dir: str,
    train_file: str,
    val_file: str,
    num_epochs: int = 2,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> dict:
    """Generate LLaMA-Factory training config for SFT."""
    return {
        # Model
        "model_name_or_path": model_name,
        "trust_remote_code": True,
        
        # Dataset
        "dataset": "custom",
        "dataset_dir": data_dir,
        "train_file": train_file,
        "val_file": val_file,
        "split": "train",
        "eval_split": "validation",
        
        # Training args
        "stage": "sft",  # supervised finetuning
        "do_train": True,
        "do_eval": True,
        "finetuning_type": "lora",
        "lora_target": "all",
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.05,
        
        # Optimization
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": 4,
        "learning_rate": learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": True,
        
        # Logging & saving
        "output_dir": output_dir,
        "logging_steps": 10,
        "save_steps": 100,
        "eval_steps": 100,
        "save_total_limit": 2,
        "evaluation_strategy": "steps",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        
        # Data processing
        "preprocessing_num_workers": 4,
        "max_length": 2048,
        "cutoff_len": 2048,
        
        # Other
        "report_to": ["tensorboard"],
        "ddp_find_unused_parameters": False,
    }


def main():
    # Paths
    project_root = Path(__file__).parent.parent
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    data_dir = project_root / "data"
    train_file = "sft_phase1_train.jsonl"
    val_file = "sft_phase1_val.jsonl"
    output_dir = project_root / "models" / "qwen2.5-7b-sft-phase1"
    config_path = project_root / "configs" / "sft_phase1.yaml"
    
    # Create config
    config = create_llamafactory_config(
        model_name=model_name,
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        train_file=train_file,
        val_file=val_file,
        num_epochs=2,
        batch_size=4,
        learning_rate=5e-5,
        lora_r=16,
        lora_alpha=32,
    )
    
    # Save config
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Config saved to: {config_path}")
    print("\nTo start training, run:")
    print(f"  llamafactory-cli train {config_path}")
    print("\nOr use Hugging Face Trainer directly (see alternative script).")


if __name__ == "__main__":
    main()
