"""
Simple Binary Tool Choice Classifier

This classifier predicts whether a tool choice is CORRECT or INCORRECT given:
- Context: The user's question/request
- LLM Response: The tool that was called
- Available Tools: List of tool descriptions

Architecture: Simple transformer-based text classifier
Input: "[CONTEXT] {context} [TOOL_CALLED] {tool_name}: {tool_description}"
Output: Binary classification (0 = incorrect, 1 = correct)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Optional
import json


class SimpleToolChoiceClassifier(nn.Module):
    """
    Simple binary classifier for tool choice validation.
    
    Uses a pretrained text encoder (BERT) to encode the context + tool choice,
    then passes through a simple MLP head for binary classification.
    
    Args:
        text_encoder_name: Name of the Hugging Face model (default: bert-base-uncased)
        hidden_dim: Hidden dimension for the MLP head
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        text_encoder_name: str = "bert-base-uncased",
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Text encoder (BERT/RoBERTa/etc)
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
        encoder_dim = self.text_encoder.config.hidden_size
        
        # Simple MLP head for binary classification
        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Binary: [incorrect, correct]
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Logits for binary classification [batch_size, 2]
        """
        # Encode text
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
        
        # Classify
        logits = self.classifier(cls_output)  # [batch_size, 2]
        
        return logits
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict tool choice correctness.
        
        Returns:
            predictions: 0 = incorrect, 1 = correct [batch_size]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=1)
        return predictions
    
    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Get probability of tool choice being correct.
        
        Returns:
            probabilities: Probability of being correct [batch_size]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            return probs[:, 1]  # Probability of class 1 (correct)


def format_input_text(
    context: str,
    tool_name: str,
    tool_description: str,
    include_image_context: bool = False,
    image_description: Optional[str] = None,
    history: Optional[str] = None,
    argument_schema: Optional[Dict] = None
) -> str:
    """
    Format the input text for the classifier.
    
    Format:
      [CONTEXT] user_question
      [HISTORY] prior_tool_calls
      [TOOL_CALLED] tool_name: description
      [SCHEMA] param1=<TYPE>, param2=<TYPE>  # NEW: Schema with masked values
    
    Args:
        context: The user's question/request
        tool_name: Name of the tool that was called
        tool_description: Description of the tool
        include_image_context: Whether to include image description
        image_description: Optional image description
        history: String with prior tool calls
        argument_schema: Dict of argument names to masked type values
        
    Returns:
        Formatted input string
    """
    parts = [f"[CONTEXT] {context}"]
    
    if include_image_context and image_description:
        parts.append(f"[IMAGE] {image_description}")
    
    if history:
        parts.append(f"[HISTORY] {history}")

    parts.append(f"[TOOL_CALLED] {tool_name}: {tool_description}")
    
    # NEW: Include argument schema with masked values
    if argument_schema and isinstance(argument_schema, dict):
        schema_str = ", ".join([f"{k}={v}" for k, v in argument_schema.items()])
        parts.append(f"[SCHEMA] {schema_str}")
    
    return " ".join(parts)


class ToolChoiceValidator:
    """
    High-level API for validating tool choices.
    
    Usage:
        validator = ToolChoiceValidator(model_path="checkpoints/best_model.pth")
        is_correct = validator.validate(
            context="What is the distance between the two buildings?",
            tool_name="Calculator",
            tool_description="A calculator tool for mathematical expressions"
        )
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the validator.
        
        Args:
            model_path: Path to trained model checkpoint (optional)
            device: Device to run on
        """
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = SimpleToolChoiceClassifier().to(self.device)
        
        if model_path:
            self.load_model(model_path)
        
        self.model.eval()
    
    def load_model(self, model_path: str):
        """Load a trained model checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        print(f"Loaded model from {model_path}")
    
    def validate(
        self,
        context: str,
        tool_name: str,
        tool_description: str,
        history: str = "",
        argument_schema: Optional[Dict] = None,
        threshold: float = 0.5,
        return_confidence: bool = False
    ) -> bool | tuple[bool, float]:
        """
        Validate if a tool choice is correct.
        
        Args:
            context: The user's question/request
            tool_name: Name of the tool that was called
            tool_description: Description of the tool
            history: String with prior tool calls
            argument_schema: Dict of argument names to masked type values
            threshold: Confidence threshold for classification
            return_confidence: Whether to return confidence score
            
        Returns:
            is_correct: True if tool choice is correct
            confidence: (optional) Confidence score [0, 1]
        """
        # Format input with full context including schema
        input_text = format_input_text(
            context, 
            tool_name, 
            tool_description, 
            history=history,
            argument_schema=argument_schema
        )
        
        # Tokenize
        encoded = self.tokenizer(
            input_text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        # Predict
        confidence = self.model.predict_proba(input_ids, attention_mask).item()
        is_correct = confidence >= threshold
        
        if return_confidence:
            return is_correct, confidence
        return is_correct
    
    def batch_validate(
        self,
        contexts: List[str],
        tool_names: List[str],
        tool_descriptions: List[str],
        threshold: float = 0.5
    ) -> List[tuple[bool, float]]:
        """
        Validate multiple tool choices at once.
        
        Returns:
            List of (is_correct, confidence) tuples
        """
        # Format inputs
        input_texts = [
            format_input_text(ctx, name, desc)
            for ctx, name, desc in zip(contexts, tool_names, tool_descriptions)
        ]
        
        # Tokenize
        encoded = self.tokenizer(
            input_texts,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        # Predict
        confidences = self.model.predict_proba(input_ids, attention_mask)
        
        results = []
        for conf in confidences:
            conf_val = conf.item()
            is_correct = conf_val >= threshold
            results.append((is_correct, conf_val))
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize model
    print("Initializing Simple Tool Choice Classifier...")
    model = SimpleToolChoiceClassifier()
    
    print(f"\nModel architecture:")
    print(model)
    
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Example input
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    context = "What is the distance between the two buildings in the image?"
    tool_name = "Calculator"
    tool_desc = "A calculator tool for mathematical expressions"
    
    input_text = format_input_text(context, tool_name, tool_desc)
    print(f"\nExample input text:\n{input_text}")
    
    # Encode and forward pass
    encoded = tokenizer(
        input_text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        logits = model(encoded["input_ids"], encoded["attention_mask"])
        probs = torch.softmax(logits, dim=1)
    
    print(f"\nOutput logits: {logits}")
    print(f"Probabilities [incorrect, correct]: {probs}")
    print(f"Prediction: {'CORRECT' if probs[0, 1] > 0.5 else 'INCORRECT'}")
    
    print("\n✓ Model initialized successfully!")
