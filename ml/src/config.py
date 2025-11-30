"""
CrisisTriage AI - ML Configuration

Pydantic-based configuration for training, evaluation, and inference.
Supports loading from YAML files for reproducible experiments.

IMPORTANT SAFETY NOTICE:
    This ML system is for RESEARCH AND SIMULATION ONLY.
    It is NOT a medical device and NOT suitable for real-world crisis intervention.
    Use only with synthetic or de-identified data in controlled environments.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal, Any

import yaml


# =============================================================================
# Risk Label Configuration
# =============================================================================

# Standard label mapping for risk classification
# Matches backend/app/core/types.py RiskLevel enum
DEFAULT_LABEL2ID = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "imminent": 3,
}

DEFAULT_ID2LABEL = {v: k for k, v in DEFAULT_LABEL2ID.items()}


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """
    Configuration for training a triage classification model.
    
    All paths are relative to the ml/ directory unless absolute.
    Load from YAML using TrainingConfig.from_yaml(path).
    """
    
    # --- Model ---
    model_name_or_path: str = "distilbert-base-uncased"
    num_labels: int = 4  # low, medium, high, imminent
    
    # --- Tokenization ---
    max_seq_length: int = 256
    padding: str = "max_length"
    truncation: bool = True
    
    # --- Data ---
    train_path: Optional[str] = None
    val_path: Optional[str] = None
    test_path: Optional[str] = None
    text_column: str = "text"
    label_column: str = "label"
    
    # --- Training Hyperparameters ---
    batch_size: int = 16
    eval_batch_size: int = 32
    num_epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    warmup_steps: Optional[int] = None  # Overrides warmup_ratio if set
    max_grad_norm: float = 1.0
    
    # --- Optimization ---
    optimizer: str = "adamw"
    scheduler: str = "linear"  # linear, cosine, constant
    
    # --- Regularization ---
    dropout: float = 0.1
    label_smoothing: float = 0.0
    
    # --- Output ---
    output_dir: str = "./outputs/default"
    save_strategy: str = "epoch"  # epoch, steps, best
    save_total_limit: int = 2
    
    # --- Logging ---
    logging_steps: int = 50
    eval_steps: Optional[int] = None  # Eval every N steps (if None, eval per epoch)
    log_to_file: bool = True
    
    # --- Reproducibility ---
    seed: int = 42
    deterministic: bool = True
    
    # --- Hardware ---
    device: str = "auto"  # auto, cpu, cuda, mps
    fp16: bool = False
    bf16: bool = False
    gradient_accumulation_steps: int = 1
    
    # --- Experiment Metadata ---
    experiment_name: str = "triage_classifier"
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.num_labels < 2:
            raise ValueError("num_labels must be at least 2")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """
        Load configuration from a YAML file.
        
        Args:
            path: Path to YAML config file
            
        Returns:
            TrainingConfig instance
        """
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Handle nested structures if any
        if config_dict is None:
            config_dict = {}
        
        return cls(**config_dict)
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def get_effective_batch_size(self) -> int:
        """Get effective batch size including gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps


# =============================================================================
# Inference Configuration
# =============================================================================

@dataclass
class InferenceConfig:
    """Configuration for model inference."""
    
    model_dir: str
    device: str = "auto"
    max_seq_length: int = 256
    batch_size: int = 32
    
    # Probability thresholds for risk escalation
    high_risk_threshold: float = 0.7
    imminent_risk_threshold: float = 0.5
    
    @classmethod
    def from_yaml(cls, path: str) -> "InferenceConfig":
        with open(path, "r") as f:
            return cls(**yaml.safe_load(f))


# =============================================================================
# Model Artifact
# =============================================================================

@dataclass
class TriageModelArtifact:
    """
    Metadata for a trained triage model artifact.
    
    Saved alongside model weights to enable proper loading and versioning.
    This bridges the ML training outputs with the backend inference service.
    """
    
    artifact_dir: str
    model_name: str  # Original base model name
    num_labels: int
    label2id: dict[str, int]
    id2label: dict[int, str]
    max_seq_length: int
    
    # Training metadata
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    training_config_hash: Optional[str] = None
    best_metric: Optional[float] = None
    best_metric_name: Optional[str] = None
    
    # Safety notice
    disclaimer: str = (
        "RESEARCH/SIMULATION ONLY. This model is NOT a medical device and "
        "NOT suitable for real-world crisis intervention."
    )
    
    def save(self, output_dir: Optional[str] = None) -> str:
        """
        Save artifact metadata to JSON.
        
        Args:
            output_dir: Directory to save to (defaults to self.artifact_dir)
            
        Returns:
            Path to saved artifact file
        """
        save_dir = output_dir or self.artifact_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        artifact_path = os.path.join(save_dir, "artifact.json")
        
        # Convert to JSON-serializable dict
        data = asdict(self)
        # Convert int keys in id2label to strings for JSON
        data["id2label"] = {str(k): v for k, v in self.id2label.items()}
        
        with open(artifact_path, "w") as f:
            json.dump(data, f, indent=2)
        
        return artifact_path
    
    @classmethod
    def load(cls, artifact_dir: str) -> "TriageModelArtifact":
        """
        Load artifact metadata from JSON.
        
        Args:
            artifact_dir: Directory containing artifact.json
            
        Returns:
            TriageModelArtifact instance
        """
        artifact_path = os.path.join(artifact_dir, "artifact.json")
        
        with open(artifact_path, "r") as f:
            data = json.load(f)
        
        # Convert string keys back to int for id2label
        data["id2label"] = {int(k): v for k, v in data["id2label"].items()}
        
        return cls(**data)


# =============================================================================
# Factory Functions
# =============================================================================

def get_label_mappings(
    num_labels: int = 4,
) -> tuple[dict[str, int], dict[int, str]]:
    """
    Get standard label mappings for risk classification.
    
    Args:
        num_labels: Number of risk levels (2, 3, or 4)
        
    Returns:
        Tuple of (label2id, id2label) dictionaries
    """
    if num_labels == 2:
        label2id = {"low": 0, "high": 1}
    elif num_labels == 3:
        label2id = {"low": 0, "medium": 1, "high": 2}
    elif num_labels == 4:
        label2id = DEFAULT_LABEL2ID
    else:
        raise ValueError(f"num_labels must be 2, 3, or 4, got {num_labels}")
    
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label


def resolve_device(device: str = "auto") -> str:
    """
    Resolve device string to actual device.
    
    Args:
        device: "auto", "cpu", "cuda", or "mps"
        
    Returns:
        Resolved device string
    """
    if device == "auto":
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device
