"""
CrisisTriage AI - Neural Triage Classifier

HuggingFace-based text classifier for risk level prediction.

IMPORTANT SAFETY NOTICE:
    This model is for RESEARCH AND SIMULATION ONLY.
    It is NOT a medical device and NOT suitable for real-world crisis intervention.
    Model predictions should NEVER be used as the sole basis for crisis response decisions.
    
Architecture:
    - Uses a pre-trained transformer encoder (e.g., DistilBERT)
    - Fine-tuned classification head for 4-class risk prediction
    - Supports batch inference for efficiency
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from ml.src.config import TriageModelArtifact, DEFAULT_LABEL2ID, DEFAULT_ID2LABEL

logger = logging.getLogger(__name__)


# =============================================================================
# Model Class
# =============================================================================

class TriageClassifier(nn.Module):
    """
    Neural classifier for triage risk level prediction.
    
    Wraps a HuggingFace transformer model with convenience methods
    for training, inference, and serialization.
    
    Usage:
        # Training
        classifier = TriageClassifier.from_pretrained(
            "distilbert-base-uncased",
            num_labels=4,
        )
        outputs = classifier(input_ids, attention_mask, labels)
        loss = outputs["loss"]
        
        # Inference
        probs = classifier.predict_proba(texts)
        
        # Save/Load
        classifier.save_pretrained("./model_output")
        classifier = TriageClassifier.from_pretrained("./model_output")
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        label2id: dict[str, int],
        id2label: dict[int, str],
        max_seq_length: int = 256,
    ):
        """
        Initialize the classifier.
        
        Use from_pretrained() class method for standard initialization.
        
        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
            label2id: Label to ID mapping
            id2label: ID to label mapping
            max_seq_length: Maximum input sequence length
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.id2label = id2label
        self.max_seq_length = max_seq_length
        self.num_labels = len(label2id)
        
        # Safety notice
        self._disclaimer = (
            "RESEARCH/SIMULATION ONLY. NOT for clinical use."
        )
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        num_labels: int = 4,
        label2id: Optional[dict[str, int]] = None,
        id2label: Optional[dict[int, str]] = None,
        max_seq_length: int = 256,
        **model_kwargs,
    ) -> "TriageClassifier":
        """
        Load a classifier from a pretrained model or local checkpoint.
        
        Args:
            model_name_or_path: HuggingFace model name or path to local checkpoint
            num_labels: Number of classification labels
            label2id: Optional label to ID mapping
            id2label: Optional ID to label mapping
            max_seq_length: Maximum sequence length
            **model_kwargs: Additional arguments for model loading
            
        Returns:
            Initialized TriageClassifier
        """
        # Check if loading from local checkpoint with artifact
        artifact_path = os.path.join(model_name_or_path, "artifact.json")
        if os.path.exists(artifact_path):
            artifact = TriageModelArtifact.load(model_name_or_path)
            label2id = artifact.label2id
            id2label = artifact.id2label
            num_labels = artifact.num_labels
            max_seq_length = artifact.max_seq_length
            logger.info(f"Loaded artifact from {model_name_or_path}")
        
        # Use defaults if not specified
        if label2id is None:
            label2id = DEFAULT_LABEL2ID
        if id2label is None:
            id2label = DEFAULT_ID2LABEL
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            id2label={str(k): v for k, v in id2label.items()},
            label2id=label2id,
            **model_kwargs,
        )
        
        logger.info(f"Loaded model from {model_name_or_path} with {num_labels} labels")
        
        return cls(
            model=model,
            tokenizer=tokenizer,
            label2id=label2id,
            id2label=id2label,
            max_seq_length=max_seq_length,
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for training or inference.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Optional ground truth labels [batch_size]
            
        Returns:
            Dictionary with:
                - loss: Cross-entropy loss (if labels provided)
                - logits: Raw model outputs [batch_size, num_labels]
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }
    
    def predict_proba(
        self,
        texts: Union[str, list[str]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Get probability distribution over risk levels.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for inference
            
        Returns:
            Array of shape [n_texts, num_labels] with probabilities
        """
        if isinstance(texts, str):
            texts = [texts]
        
        self.model.eval()
        device = next(self.model.parameters()).device
        
        all_probs = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                encodings = self.tokenizer(
                    batch_texts,
                    max_length=self.max_seq_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                
                input_ids = encodings["input_ids"].to(device)
                attention_mask = encodings["attention_mask"].to(device)
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Convert to probabilities
                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs.cpu().numpy())
        
        return np.concatenate(all_probs, axis=0)
    
    def predict(
        self,
        texts: Union[str, list[str]],
        return_probs: bool = False,
    ) -> Union[list[str], tuple[list[str], np.ndarray]]:
        """
        Predict risk levels for texts.
        
        Args:
            texts: Single text or list of texts
            return_probs: If True, also return probability distribution
            
        Returns:
            List of predicted label strings (and optionally probabilities)
        """
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        probs = self.predict_proba(texts)
        predicted_ids = np.argmax(probs, axis=1)
        predicted_labels = [self.id2label[int(idx)] for idx in predicted_ids]
        
        if single_input:
            predicted_labels = predicted_labels[0]
            if return_probs:
                return predicted_labels, probs[0]
            return predicted_labels
        
        if return_probs:
            return predicted_labels, probs
        return predicted_labels
    
    def save_pretrained(
        self,
        output_dir: str,
        best_metric: Optional[float] = None,
        best_metric_name: Optional[str] = None,
    ) -> str:
        """
        Save model, tokenizer, and artifact metadata.
        
        Args:
            output_dir: Directory to save to
            best_metric: Optional metric value achieved
            best_metric_name: Optional name of the metric
            
        Returns:
            Path to saved artifact.json
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Create and save artifact
        artifact = TriageModelArtifact(
            artifact_dir=str(output_path.absolute()),
            model_name=self.model.config._name_or_path,
            num_labels=self.num_labels,
            label2id=self.label2id,
            id2label=self.id2label,
            max_seq_length=self.max_seq_length,
            best_metric=best_metric,
            best_metric_name=best_metric_name,
        )
        artifact_path = artifact.save(output_dir)
        
        logger.info(f"Saved model to {output_dir}")
        
        return artifact_path
    
    def to(self, device: Union[str, torch.device]) -> "TriageClassifier":
        """Move model to device."""
        self.model = self.model.to(device)
        return self
    
    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return next(self.model.parameters()).device
    
    def train_mode(self) -> "TriageClassifier":
        """Set model to training mode."""
        self.model.train()
        return self
    
    def eval_mode(self) -> "TriageClassifier":
        """Set model to evaluation mode."""
        self.model.eval()
        return self


# =============================================================================
# Utility Functions
# =============================================================================

def compute_loss_with_label_smoothing(
    logits: torch.Tensor,
    labels: torch.Tensor,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """
    Compute cross-entropy loss with label smoothing.
    
    Args:
        logits: Model outputs [batch_size, num_classes]
        labels: Ground truth labels [batch_size]
        smoothing: Label smoothing factor (0 = no smoothing)
        
    Returns:
        Scalar loss tensor
    """
    num_classes = logits.size(-1)
    
    if smoothing == 0:
        return F.cross_entropy(logits, labels)
    
    # Create smoothed labels
    with torch.no_grad():
        smooth_labels = torch.zeros_like(logits)
        smooth_labels.fill_(smoothing / (num_classes - 1))
        smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing)
    
    # Compute loss
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(smooth_labels * log_probs).sum(dim=-1).mean()
    
    return loss
