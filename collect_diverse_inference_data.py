#!/usr/bin/env python3
"""
Collect diverse LLM predictions using stochastic decoding.

This script runs inference multiple times with different temperature settings
to collect diverse tool choices. These will be used to train a better classifier.

Usage:
    python collect_diverse_inference_data.py \\
        --api-url http://localhost:12581 \\
        --config-path configs/eval_ThinkGeo_bench.py \\
        --num-runs 5 \\
        --temperatures 0.7 0.75 0.8 0.85 0.9 \\
        --output-dir diverse_predictions
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional
import time

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Install with: pip install requests")
    sys.exit(1)


def load_config(config_path: str) -> Dict:
    """Load OpenCompass config to get model info."""
    try:
        # This is a simplified loader - adjust based on your config structure
        with open(config_path, 'r') as f:
            content = f.read()
        # Try to extract model info (simplified)
        if "qwen" in content.lower():
            return {
                "model_name": "qwen2.5-7b-instruct",
                "description": "Loaded from config"
            }
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
    
    return {"model_name": "qwen2.5-7b-instruct", "description": "default"}


def call_llm(
    prompt: str,
    model_name: str,
    api_url: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 2000,
    timeout: int = 60
) -> Optional[str]:
    """Call LLM via lmdeploy API with stochastic decoding parameters."""
    
    data = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": 50,
    }
    
    try:
        response = requests.post(
            f"{api_url}/v1/completions",
            json=data,
            timeout=timeout
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["text"]
    except requests.exceptions.Timeout:
        print(f"  ⏱ Timeout after {timeout}s")
        return None
    except requests.exceptions.ConnectionError:
        print(f"  ❌ Connection error - is server running at {api_url}?")
        return None
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


def collect_diverse_predictions(
    api_url: str,
    model_name: str,
    num_runs: int = 5,
    temperatures: Optional[List[float]] = None,
    output_dir: str = "./diverse_predictions",
    max_samples_per_run: Optional[int] = None,
):
    """
    Collect diverse predictions with stochastic decoding.
    
    Args:
        api_url: Base URL of lmdeploy API server (e.g., http://localhost:12581)
        model_name: Model name to use
        num_runs: Number of runs to perform
        temperatures: List of temperatures to try
        output_dir: Directory to save results
        max_samples_per_run: Limit samples per run (for testing)
    """
    
    if temperatures is None:
        temperatures = [0.7, 0.75, 0.8, 0.85, 0.9]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*90)
    print("COLLECTING DIVERSE PREDICTIONS FOR CLASSIFIER TRAINING")
    print("="*90)
    print(f"API URL: {api_url}")
    print(f"Model: {model_name}")
    print(f"Runs: {num_runs}")
    print(f"Temperatures: {temperatures}")
    print(f"Output: {output_path.absolute()}")
    print("="*90)
    
    # For now, use test prompts - you should load from your benchmark
    test_prompts = [
        "What is the capital of France?",
        "How many continents are there?",
        "What year was the Internet invented?",
        "What is the largest ocean on Earth?",
        "Who wrote Romeo and Juliet?",
    ]
    
    print(f"\nUsing {len(test_prompts)} test prompts for demo")
    print("TODO: Load actual prompts from your ThinkGeoBench dataset\n")
    
    all_results = []
    successful_runs = 0
    total_predictions = 0
    
    for run_idx in range(num_runs):
        print(f"\n{'='*90}")
        print(f"RUN {run_idx + 1}/{num_runs}")
        print(f"{'='*90}")
        
        run_results = []
        run_start = time.time()
        
        for temp_idx, temperature in enumerate(temperatures):
            print(f"\n  Temperature: {temperature} ({temp_idx + 1}/{len(temperatures)})")
            
            for prompt_idx, prompt in enumerate(test_prompts):
                if max_samples_per_run and prompt_idx >= max_samples_per_run:
                    break
                
                # Call LLM
                response = call_llm(
                    prompt=prompt,
                    model_name=model_name,
                    api_url=api_url,
                    temperature=temperature,
                    top_p=0.95,
                    max_tokens=100,  # Use smaller for testing
                )
                
                if response:
                    run_results.append({
                        "run": run_idx,
                        "temperature": temperature,
                        "prompt": prompt,
                        "response": response[:200],  # Truncate for storage
                        "timestamp": time.time()
                    })
                    total_predictions += 1
                    print(f"    ✓ Prompt {prompt_idx + 1}/{len(test_prompts)}")
                else:
                    print(f"    ✗ Prompt {prompt_idx + 1}/{len(test_prompts)} (failed)")
        
        if run_results:
            successful_runs += 1
            all_results.extend(run_results)
            
            # Save run file
            run_file = output_path / f"run_{run_idx:02d}.json"
            with open(run_file, 'w') as f:
                json.dump(run_results, f, indent=2)
            
            run_duration = time.time() - run_start
            print(f"\n  ✓ Run completed in {run_duration:.1f}s")
            print(f"  ✓ Saved {len(run_results)} predictions to {run_file.name}")
        else:
            print(f"\n  ✗ Run failed - no predictions collected")
    
    # Save combined results
    combined_file = output_path / "all_runs.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save metadata
    metadata = {
        "num_runs": num_runs,
        "successful_runs": successful_runs,
        "temperatures": temperatures,
        "total_predictions": total_predictions,
        "prompts_per_run": len(test_prompts),
        "output_directory": str(output_path.absolute()),
        "timestamp": time.time()
    }
    
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*90)
    print("COLLECTION COMPLETE")
    print("="*90)
    print(f"Successful runs: {successful_runs}/{num_runs}")
    print(f"Total predictions: {total_predictions}")
    print(f"Average per run: {total_predictions / max(successful_runs, 1):.0f}")
    print(f"\nOutput files:")
    print(f"  - {combined_file.name} (all predictions)")
    print(f"  - {metadata_file.name} (collection metadata)")
    print(f"  - run_*.json (per-run results)")
    print("\nNEXT STEPS:")
    print("  1. Label predictions against gold standard")
    print("  2. Create training dataset with labels")
    print("  3. Train classifier on realistic 72/28 distribution")
    print("  4. Evaluate with >75% accuracy target")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Collect diverse LLM predictions for classifier training"
    )
    
    parser.add_argument(
        "--api-url",
        default="http://localhost:12581",
        help="Base URL of lmdeploy API server (default: http://localhost:12581)"
    )
    
    parser.add_argument(
        "--model-name",
        default="qwen2.5-7b-instruct",
        help="Model name (default: qwen2.5-7b-instruct)"
    )
    
    parser.add_argument(
        "--config-path",
        default="opencompass/configs/eval_ThinkGeo_bench.py",
        help="Path to OpenCompass config (for reference)"
    )
    
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of runs to perform (default: 5)"
    )
    
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.7, 0.75, 0.8, 0.85, 0.9],
        help="Temperature values to try (default: 0.7 0.75 0.8 0.85 0.9)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="./diverse_predictions",
        help="Output directory (default: ./diverse_predictions)"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per run (for testing, default: None = all)"
    )
    
    args = parser.parse_args()
    
    # Collect predictions
    collect_diverse_predictions(
        api_url=args.api_url,
        model_name=args.model_name,
        num_runs=args.num_runs,
        temperatures=args.temperatures,
        output_dir=args.output_dir,
        max_samples_per_run=args.max_samples,
    )


if __name__ == "__main__":
    main()
