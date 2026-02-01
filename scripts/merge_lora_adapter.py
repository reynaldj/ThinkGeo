#!/usr/bin/env python3
"""
Merge LoRA adapters with base model for deployment.

Usage:
  python scripts/merge_lora_adapter.py \
    --adapter-path models/qwen2.5-7b-sft-phase1 \
    --base-model /home/james/models/qwen2.5-7b-instruct \
    --output-path models/qwen2.5-7b-sft-phase1-merged
"""
import argparse
import shutil
from pathlib import Path
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", required=True, help="Path to LoRA adapter")
    parser.add_argument("--base-model", required=True, help="Path to base model")
    parser.add_argument("--output-path", required=True, help="Output path for merged model")
    args = parser.parse_args()
    
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading adapter from: {args.adapter_path}")
    model = AutoPeftModelForCausalLM.from_pretrained(args.adapter_path)
    
    print("Merging adapters...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to: {args.output_path}")
    merged_model.save_pretrained(args.output_path)
    
    print(f"Copying tokenizer from: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_path)
    
    # Copy chat template if exists
    chat_template_src = Path(args.base_model) / "chat_template.jinja"
    if chat_template_src.exists():
        shutil.copy(chat_template_src, output_path / "chat_template.jinja")
        print("Copied chat_template.jinja")
    
    print(f"\n✓ Merge complete! Model saved to {args.output_path}")
    print(f"  Ready to deploy with:")
    print(f"  lmdeploy serve api_server {args.output_path} --server-port 12584")


if __name__ == "__main__":
    main()
