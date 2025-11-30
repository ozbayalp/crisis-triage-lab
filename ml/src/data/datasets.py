"""
CrisisTriage AI - Dataset Utilities

Dataset loading and processing for triage classification training.

IMPORTANT SAFETY NOTICE:
    This code is designed for RESEARCH AND SIMULATION ONLY.
    Use only with synthetic or properly de-identified datasets.
    Never use real crisis hotline data without proper IRB approval,
    consent, and data governance procedures.

Supported formats:
    - CSV: Expects columns specified by config.text_column and config.label_column
    - JSON/JSONL: Expects objects with text_column and label_column keys
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, AutoTokenizer

from ml.src.config import TrainingConfig, DEFAULT_LABEL2ID


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TriageExample:
    """
    A single training example for triage classification.
    
    Attributes:
        text: Input text (e.g., transcribed speech, chat message)
        label: Risk level label (e.g., "low", "medium", "high", "imminent")
        example_id: Optional unique identifier
        metadata: Optional additional metadata
    """
    text: str
    label: str
    example_id: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    
    def __post_init__(self):
        if not self.text:
            raise ValueError("Text cannot be empty")
        if not self.label:
            raise ValueError("Label cannot be empty")


# =============================================================================
# PyTorch Dataset
# =============================================================================

class TriageDataset(Dataset):
    """
    PyTorch Dataset for triage classification.
    
    Tokenizes text and maps labels to integer IDs.
    
    Usage:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        dataset = TriageDataset(
            examples=examples,
            tokenizer=tokenizer,
            label2id={"low": 0, "medium": 1, "high": 2, "imminent": 3},
            max_length=256,
        )
        input_ids, attention_mask, label = dataset[0]
    """
    
    def __init__(
        self,
        examples: list[TriageExample],
        tokenizer: PreTrainedTokenizer,
        label2id: dict[str, int],
        max_length: int = 256,
        padding: str = "max_length",
        truncation: bool = True,
    ):
        """
        Initialize the dataset.
        
        Args:
            examples: List of TriageExample objects
            tokenizer: HuggingFace tokenizer
            label2id: Mapping from label string to integer ID
            max_length: Maximum sequence length
            padding: Padding strategy ("max_length", "longest", False)
            truncation: Whether to truncate sequences
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        
        # Validate labels
        unknown_labels = set()
        for ex in examples:
            if ex.label not in label2id:
                unknown_labels.add(ex.label)
        
        if unknown_labels:
            raise ValueError(
                f"Unknown labels in dataset: {unknown_labels}. "
                f"Expected one of: {list(label2id.keys())}"
            )
        
        # Pre-tokenize all examples for efficiency
        self._encodings = self._tokenize_all()
    
    def _tokenize_all(self) -> dict[str, torch.Tensor]:
        """Tokenize all examples at once."""
        texts = [ex.text for ex in self.examples]
        
        encodings = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt",
        )
        
        return encodings
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single tokenized example.
        
        Returns:
            Tuple of (input_ids, attention_mask, label_id)
        """
        input_ids = self._encodings["input_ids"][idx]
        attention_mask = self._encodings["attention_mask"][idx]
        label_id = torch.tensor(self.label2id[self.examples[idx].label], dtype=torch.long)
        
        return input_ids, attention_mask, label_id
    
    def get_labels(self) -> list[int]:
        """Get all label IDs for the dataset."""
        return [self.label2id[ex.label] for ex in self.examples]


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_examples_from_file(
    file_path: str,
    text_column: str = "text",
    label_column: str = "label",
) -> list[TriageExample]:
    """
    Load examples from a CSV or JSON/JSONL file.
    
    Args:
        file_path: Path to data file
        text_column: Name of column/key containing text
        label_column: Name of column/key containing labels
        
    Returns:
        List of TriageExample objects
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    suffix = file_path.suffix.lower()
    
    if suffix == ".csv":
        return _load_from_csv(file_path, text_column, label_column)
    elif suffix == ".json":
        return _load_from_json(file_path, text_column, label_column)
    elif suffix == ".jsonl":
        return _load_from_jsonl(file_path, text_column, label_column)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .csv, .json, or .jsonl")


def _load_from_csv(
    file_path: Path,
    text_column: str,
    label_column: str,
) -> list[TriageExample]:
    """Load examples from CSV file."""
    examples = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        # Validate columns exist
        if reader.fieldnames is None:
            raise ValueError("CSV file appears to be empty or malformed")
        
        if text_column not in reader.fieldnames:
            raise ValueError(f"Text column '{text_column}' not found in CSV. Available: {reader.fieldnames}")
        if label_column not in reader.fieldnames:
            raise ValueError(f"Label column '{label_column}' not found in CSV. Available: {reader.fieldnames}")
        
        for i, row in enumerate(reader):
            text = row[text_column].strip()
            label = row[label_column].strip().lower()
            
            if text and label:
                examples.append(TriageExample(
                    text=text,
                    label=label,
                    example_id=f"csv_{i}",
                ))
    
    return examples


def _load_from_json(
    file_path: Path,
    text_column: str,
    label_column: str,
) -> list[TriageExample]:
    """Load examples from JSON file (array of objects)."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("JSON file should contain an array of objects")
    
    examples = []
    for i, item in enumerate(data):
        if text_column not in item:
            raise ValueError(f"Text key '{text_column}' not found in JSON object at index {i}")
        if label_column not in item:
            raise ValueError(f"Label key '{label_column}' not found in JSON object at index {i}")
        
        text = str(item[text_column]).strip()
        label = str(item[label_column]).strip().lower()
        
        if text and label:
            examples.append(TriageExample(
                text=text,
                label=label,
                example_id=f"json_{i}",
            ))
    
    return examples


def _load_from_jsonl(
    file_path: Path,
    text_column: str,
    label_column: str,
) -> list[TriageExample]:
    """Load examples from JSONL file (one JSON object per line)."""
    examples = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            
            item = json.loads(line)
            
            if text_column not in item:
                raise ValueError(f"Text key '{text_column}' not found at line {i}")
            if label_column not in item:
                raise ValueError(f"Label key '{label_column}' not found at line {i}")
            
            text = str(item[text_column]).strip()
            label = str(item[label_column]).strip().lower()
            
            if text and label:
                examples.append(TriageExample(
                    text=text,
                    label=label,
                    example_id=f"jsonl_{i}",
                ))
    
    return examples


def load_triage_dataset(
    config: TrainingConfig,
    split: Literal["train", "val", "test"],
    tokenizer: Optional[PreTrainedTokenizer] = None,
    label2id: Optional[dict[str, int]] = None,
) -> TriageDataset:
    """
    Load a triage dataset for a specific split.
    
    Args:
        config: Training configuration
        split: Which split to load ("train", "val", or "test")
        tokenizer: Optional pre-loaded tokenizer (loads from config if None)
        label2id: Optional label mapping (uses default if None)
        
    Returns:
        TriageDataset ready for training/evaluation
    """
    # Get file path for split
    path_map = {
        "train": config.train_path,
        "val": config.val_path,
        "test": config.test_path,
    }
    
    file_path = path_map.get(split)
    if file_path is None:
        raise ValueError(f"No path configured for split '{split}'")
    
    # Load tokenizer if not provided
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    
    # Use default label mapping if not provided
    if label2id is None:
        label2id = DEFAULT_LABEL2ID
    
    # Load examples
    examples = load_examples_from_file(
        file_path,
        text_column=config.text_column,
        label_column=config.label_column,
    )
    
    # Create dataset
    dataset = TriageDataset(
        examples=examples,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=config.max_seq_length,
        padding=config.padding,
        truncation=config.truncation,
    )
    
    return dataset


def build_label_mapping(dataset_paths: list[str], label_column: str = "label") -> dict[str, int]:
    """
    Build label mapping from one or more data files.
    
    Scans all files to find unique labels and assigns integer IDs.
    
    Args:
        dataset_paths: List of paths to data files
        label_column: Name of label column/key
        
    Returns:
        Dictionary mapping label strings to integer IDs
    """
    labels = set()
    
    for path in dataset_paths:
        if not Path(path).exists():
            continue
        
        examples = load_examples_from_file(path, label_column=label_column)
        labels.update(ex.label for ex in examples)
    
    # Sort for deterministic ordering
    sorted_labels = sorted(labels)
    
    return {label: i for i, label in enumerate(sorted_labels)}


def create_data_loader(
    dataset: TriageDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader from a TriageDataset.
    
    Args:
        dataset: TriageDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory (for GPU training)
        
    Returns:
        DataLoader instance
    """
    from ml.src.utils.seed import get_seed_worker
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=get_seed_worker if num_workers > 0 else None,
    )
