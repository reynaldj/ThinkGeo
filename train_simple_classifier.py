"""
Training script for Simple Tool Choice Classifier

Trains the classifier to predict whether a tool choice is correct or incorrect.
Input: full_history, thought, tool_name, tool_arguments, tool_description
Output: Binary classification (0 = incorrect, 1 = correct)
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from simple_tool_classifier import SimpleToolChoiceClassifier
import os
from tqdm import tqdm
from typing import Dict, List, Tuple
import argparse


class ToolChoiceDataset(Dataset):
    """
    Dataset for tool choice classification.
    
    Loads examples from JSON files and creates input-output pairs.
    Input: Formatted text with context, thought, tool info
    Output: Binary label (0 = incorrect, 1 = correct)
    """
    
    def __init__(
        self,
        json_files: List[str],
        tokenizer,
        max_length: int = 512,
        include_thought: bool = False
    ):
        """
        Args:
            json_files: List of JSON file paths (train.json, val.json, etc.)
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            include_thought: Whether to include the LLM's thought
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_thought = include_thought
        self.examples = []
        
        # Load all examples from JSON files
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            for example in data:
                self.examples.append(example)
        
        print(f"Loaded {len(self.examples)} examples from {len(json_files)} files")
    
    def _extract_user_query(self, example: Dict) -> str:
        full_history = example.get("full_history") or []
        for item in reversed(full_history):
            if item.get("role") == "user" and item.get("content"):
                return str(item.get("content"))
        return ""

    def _summarize_history(self, example: Dict) -> str:
        """
        Summarize tool calls in history using schema placeholders instead of actual values.
        This prevents the model from memorizing specific argument/result values and instead
        learning to validate tool choices based on semantic appropriateness.
        """
        full_history = example.get("full_history") or []
        history_parts = []
        
        # Build a map of tool calls from assistant messages to track argument names
        last_tool_args = {}  # tool_name -> arg names for that tool

        # Collect tool calls from both assistant tool_calls and tool result messages
        for i, item in enumerate(full_history):
            role = item.get("role")
            
            # From assistant messages with tool_calls
            if role == "assistant" and item.get("tool_calls"):
                for tool_call in item["tool_calls"]:
                    func = tool_call.get("function", {})
                    tool_name = func.get("name", "")
                    args = func.get("arguments", {})
                    
                    if isinstance(args, dict):
                        arg_names = ", ".join(args.keys())
                    else:
                        arg_names = "<args>"
                    
                    # Store for later reference
                    last_tool_args[tool_name] = arg_names
                    history_parts.append(f"[TOOL] {tool_name}({arg_names}) -> <result>")
            
            # From tool result messages (fallback)
            elif role in ("tool", "function"):
                name = item.get("name") or item.get("tool_name") or item.get("function_name") or ""
                args = item.get("arguments", {})
                
                if isinstance(args, dict) and args:
                    # Use arguments from tool message if available
                    arg_names = ", ".join(args.keys())
                    history_parts.append(f"[TOOL] {name}({arg_names}) -> <result>")
                elif name and name in last_tool_args:
                    # Use stored args from the most recent tool_call for this tool
                    arg_names = last_tool_args[name]
                    history_parts.append(f"[TOOL] {name}({arg_names}) -> <result>")

        history_str = " ".join(history_parts)
        # Truncate full history string if too long
        if len(history_str) > 500:
            history_str = history_str[:500] + "..."
        return history_str

    def format_input(self, example: Dict) -> str:
        """
        Format the input text for the classifier.
        
        Input format:
          [CONTEXT] user_query
          [HISTORY] prior_tool_calls
          [TOOL_CALLED] tool_name: description
          [SCHEMA] param1=<TYPE>, param2=<TYPE>  # Parameter names with masked values
        
        Args:
            example: Single example from the dataset
            
        Returns:
            Formatted input string
        """
        parts = []

        # Context: latest user query
        user_query = self._extract_user_query(example)
        if user_query:
            parts.append(f"[CONTEXT] {user_query}")

        # History: tool calls only
        history_str = self._summarize_history(example)
        if history_str:
            parts.append(f"[HISTORY] {history_str}")

        # Tool information
        tool_name = example.get("tool_name", "")
        tool_desc = example.get("tool_description", "")
        parts.append(f"[TOOL_CALLED] {tool_name}: {tool_desc}")
        
        # Arguments: Use actual arguments if no schema available
        tool_args = example.get("tool_arguments", {})
        arg_schema = example.get("argument_schema", {})
        
        if arg_schema:
            # Use schema if available (parameter names with types)
            schema_str = ", ".join([f"{k}={v}" for k, v in arg_schema.items()])
            parts.append(f"[SCHEMA] {schema_str}")
        elif tool_args and isinstance(tool_args, dict):
            # Fallback: use argument names only (without values for privacy)
            args_str = ", ".join(tool_args.keys())
            if args_str:
                parts.append(f"[ARGS] {args_str}")
        
        return " ".join(parts)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single example.
        
        Returns:
            Dict with 'input_ids', 'attention_mask', and 'label'
        """
        example = self.examples[idx]
        
        # Format input text
        input_text = self.format_input(example)
        
        # Get label directly from the dataset (1 = correct, 0 = incorrect)
        label = example.get("label", 0)
        
        # Debug: Show formatted input for first 5 examples
        if idx < 5:
            import sys
            print(f"\n{'='*80}", file=sys.stderr)
            print(f"[TRAINING EXAMPLE {idx}]", file=sys.stderr)
            print(f"Label: {label} ({'CORRECT' if label == 1 else 'INCORRECT'})", file=sys.stderr)
            print(f"Formatted Input:", file=sys.stderr)
            print(f"{input_text}", file=sys.stderr)
            print(f"{'='*80}\n", file=sys.stderr)
        
        # Tokenize
        encoded = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "example_id": example.get("global_sample_idx", idx)
        }


def create_dataloaders(
    train_file: str,
    val_file: str,
    test_file: str,
    tokenizer,
    batch_size: int = 32,
    max_length: int = 512,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, and test dataloaders.
    """
    train_dataset = ToolChoiceDataset(
        [train_file],
        tokenizer,
        max_length=max_length
    )
    
    val_dataset = ToolChoiceDataset(
        [val_file],
        tokenizer,
        max_length=max_length
    )
    
    test_dataset = ToolChoiceDataset(
        [test_file],
        tokenizer,
        max_length=max_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer,
    criterion,
    device: torch.device,
    epoch: int
) -> float:
    """
    Train for one epoch.
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [TRAIN]")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        # Forward pass
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({"loss": loss.item()})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion,
    device: torch.device,
    phase: str = "VAL"
) -> Tuple[float, float, float, float]:
    """
    Evaluate on validation or test set.
    
    Returns:
        Tuple of (loss, accuracy, precision, recall)
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    correct = 0
    total = 0
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives
    
    pbar = tqdm(data_loader, desc=f"[{phase}]")
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Compute metrics
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Precision/Recall (for class 1 = correct)
            tp += ((predictions == 1) & (labels == 1)).sum().item()
            fp += ((predictions == 1) & (labels == 0)).sum().item()
            fn += ((predictions == 0) & (labels == 1)).sum().item()
            
            pbar.set_postfix({"loss": loss.item()})
    
    avg_loss = total_loss / num_batches
    accuracy = correct / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return avg_loss, accuracy, precision, recall


def train(
    train_file: str,
    val_file: str,
    test_file: str,
    output_dir: str = "./checkpoints",
    num_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Main training loop.
    """
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)
    
    print(f"Using device: {device}")
    print(f"Output directory: {output_dir}")
    
    # Initialize model, tokenizer, optimizer
    print("\nInitializing model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = SimpleToolChoiceClassifier().to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_file=train_file,
        val_file=val_file,
        test_file=test_file,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Analyze class distribution and calculate class weights
    print("\nAnalyzing class distribution...")
    train_labels = [ex.get("label", 0) for ex in train_loader.dataset.examples]
    num_class_0 = sum(1 for label in train_labels if label == 0)
    num_class_1 = sum(1 for label in train_labels if label == 1)
    total_samples = len(train_labels)
    
    print(f"Class 0 (incorrect): {num_class_0} samples ({100*num_class_0/total_samples:.1f}%)")
    print(f"Class 1 (correct): {num_class_1} samples ({100*num_class_1/total_samples:.1f}%)")
    print(f"Distribution: {100*num_class_0/total_samples:.1f}% incorrect, {100*num_class_1/total_samples:.1f}% correct")
    
    # Calculate inverse frequency weights
    weight_class_0 = total_samples / (2 * num_class_0)
    weight_class_1 = total_samples / (2 * num_class_1)
    class_weights = torch.tensor([weight_class_0, weight_class_1], dtype=torch.float32).to(device)
    
    print(f"\nClass weights:")
    print(f"  Class 0 (incorrect): {weight_class_0:.4f}")
    print(f"  Class 1 (correct): {weight_class_1:.4f}")
    print("Using weighted loss to balance class importance")
    
    # Setup optimizer and loss WITH class weights
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training loop
    best_val_accuracy = 0
    best_model_path = os.path.join(output_dir, "best_model.pth")
    
    print("\nStarting training...\n")
    for epoch in range(num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch + 1)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss, val_acc, val_prec, val_rec = evaluate(
            model, val_loader, criterion, device, phase="VAL"
        )
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        print(f"Val Precision: {val_prec:.4f}")
        print(f"Val Recall: {val_rec:.4f}")
        
        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "val_loss": val_loss
            }, best_model_path)
            print(f"✓ Saved best model (acc: {val_acc:.4f})")
    
    # Test with best model
    print(f"\n{'='*80}")
    print("Loading best model and evaluating on test set...")
    print(f"{'='*80}")
    
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_loss, test_acc, test_prec, test_rec = evaluate(
        model, test_loader, criterion, device, phase="TEST"
    )
    
    print(f"\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall: {test_rec:.4f}")
    
    # Save final results
    results = {
        "best_val_accuracy": best_val_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_precision": test_prec,
        "test_recall": test_rec,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length
    }
    
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved results to {results_path}")
    print(f"✓ Best model saved to {best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Simple Tool Choice Classifier")
    parser.add_argument(
        "--train-file",
        default="/home/james/ThinkGeo/tool_choice_data_from_predictions/train.json",
        help="Path to training data"
    )
    parser.add_argument(
        "--val-file",
        default="/home/james/ThinkGeo/tool_choice_data_from_predictions/val.json",
        help="Path to validation data"
    )
    parser.add_argument(
        "--test-file",
        default="/home/james/ThinkGeo/tool_choice_data_from_predictions/test.json",
        help="Path to test data"
    )
    parser.add_argument(
        "--output-dir",
        default="./checkpoints2",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    
    args = parser.parse_args()
    
    train(
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length
    )
