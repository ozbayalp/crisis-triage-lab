"""
CrisisTriage AI - Model Evaluation Script

Comprehensive evaluation of trained triage classifiers.

IMPORTANT SAFETY NOTICE:
    This evaluation is for RESEARCH AND SIMULATION ONLY.
    Evaluation metrics do NOT indicate suitability for clinical use.
    The model is NOT a medical device.

Usage:
    python -m ml.src.eval.evaluate_triage_model \
        --config ml/experiments/configs/baseline_bert.yaml \
        --checkpoint ml/outputs/baseline/best_model
        
    Or evaluate on a specific test file:
    python -m ml.src.eval.evaluate_triage_model \
        --checkpoint ml/outputs/baseline/best_model \
        --test_path data/test.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from ml.src.config import TrainingConfig, TriageModelArtifact, resolve_device
from ml.src.data.datasets import load_triage_dataset, create_data_loader, load_examples_from_file
from ml.src.models.triage_classifier import TriageClassifier
from ml.src.utils.logging import setup_logging, get_logger


@dataclass
class EvaluationResults:
    """Complete evaluation results."""
    
    # Overall metrics
    accuracy: float
    f1_macro: float
    f1_weighted: float
    
    # Per-class metrics
    f1_per_class: dict[str, float]
    precision_per_class: dict[str, float]
    recall_per_class: dict[str, float]
    
    # Confusion matrix (as nested list for JSON serialization)
    confusion_matrix: list[list[int]]
    labels: list[str]
    
    # ROC-AUC (if available)
    roc_auc_macro: Optional[float] = None
    roc_auc_per_class: Optional[dict[str, float]] = None
    
    # Dataset info
    num_samples: int = 0
    test_path: str = ""
    checkpoint_path: str = ""
    
    # Safety notice
    disclaimer: str = (
        "RESEARCH/SIMULATION ONLY. These metrics do NOT indicate clinical validity. "
        "This model is NOT a medical device and NOT suitable for real crisis intervention."
    )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, output_path: str) -> None:
        """Save results to JSON file."""
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "EvaluationResults":
        """Load results from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained triage classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to config YAML (for test_path)",
    )
    
    parser.add_argument(
        "--test_path",
        type=str,
        default=None,
        help="Path to test data (overrides config)",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (defaults to checkpoint dir)",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Evaluation batch size",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda, mps)",
    )
    
    return parser.parse_args()


def evaluate_model(
    checkpoint_dir: str,
    test_path: str,
    batch_size: int = 32,
    device: str = "auto",
    text_column: str = "text",
    label_column: str = "label",
) -> EvaluationResults:
    """
    Evaluate a trained model on a test set.
    
    Args:
        checkpoint_dir: Path to model checkpoint
        test_path: Path to test data file
        batch_size: Batch size for evaluation
        device: Device to use
        text_column: Name of text column in data
        label_column: Name of label column in data
        
    Returns:
        EvaluationResults with all metrics
    """
    logger = get_logger("eval")
    logger.info("=" * 60)
    logger.info("CrisisTriage AI - Model Evaluation")
    logger.info("RESEARCH/SIMULATION ONLY - NOT FOR CLINICAL USE")
    logger.info("=" * 60)
    
    # Resolve device
    device = resolve_device(device)
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {checkpoint_dir}")
    model = TriageClassifier.from_pretrained(checkpoint_dir)
    model.to(device)
    model.eval_mode()
    
    # Get label info
    label2id = model.label2id
    id2label = model.id2label
    labels = [id2label[i] for i in range(len(id2label))]
    logger.info(f"Labels: {labels}")
    
    # Load test data
    logger.info(f"Loading test data from {test_path}")
    examples = load_examples_from_file(test_path, text_column, label_column)
    logger.info(f"Test samples: {len(examples)}")
    
    # Create dataset and loader
    from ml.src.data.datasets import TriageDataset
    
    test_dataset = TriageDataset(
        examples=examples,
        tokenizer=model.tokenizer,
        label2id=label2id,
        max_length=model.max_seq_length,
    )
    
    test_loader = create_data_loader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    
    # Run inference
    logger.info("Running inference...")
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for input_ids, attention_mask, batch_labels in tqdm(test_loader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    logger.info("Computing metrics...")
    
    # Overall
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")
    
    # Per-class
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    f1_dict = {labels[i]: float(f1_per_class[i]) for i in range(len(labels))}
    precision_dict = {labels[i]: float(precision[i]) for i in range(len(labels))}
    recall_dict = {labels[i]: float(recall[i]) for i in range(len(labels))}
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds).tolist()
    
    # ROC-AUC (one-vs-rest)
    try:
        from sklearn.preprocessing import label_binarize
        labels_binarized = label_binarize(all_labels, classes=list(range(len(labels))))
        
        if len(labels) == 2:
            roc_auc_macro = roc_auc_score(all_labels, all_probs[:, 1])
            roc_auc_per_class = None
        else:
            roc_auc_macro = roc_auc_score(
                labels_binarized, all_probs, average="macro", multi_class="ovr"
            )
            roc_auc_per_class_values = roc_auc_score(
                labels_binarized, all_probs, average=None, multi_class="ovr"
            )
            roc_auc_per_class = {
                labels[i]: float(roc_auc_per_class_values[i])
                for i in range(len(labels))
            }
    except Exception as e:
        logger.warning(f"Could not compute ROC-AUC: {e}")
        roc_auc_macro = None
        roc_auc_per_class = None
    
    # Print classification report
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(all_labels, all_preds, target_names=labels))
    
    # Build results
    results = EvaluationResults(
        accuracy=float(accuracy),
        f1_macro=float(f1_macro),
        f1_weighted=float(f1_weighted),
        f1_per_class=f1_dict,
        precision_per_class=precision_dict,
        recall_per_class=recall_dict,
        confusion_matrix=cm,
        labels=labels,
        roc_auc_macro=float(roc_auc_macro) if roc_auc_macro else None,
        roc_auc_per_class=roc_auc_per_class,
        num_samples=len(examples),
        test_path=str(test_path),
        checkpoint_path=str(checkpoint_dir),
    )
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Evaluation Summary:")
    logger.info(f"  Accuracy:    {accuracy:.4f}")
    logger.info(f"  F1 (macro):  {f1_macro:.4f}")
    logger.info(f"  F1 (weighted): {f1_weighted:.4f}")
    if roc_auc_macro:
        logger.info(f"  ROC-AUC:     {roc_auc_macro:.4f}")
    logger.info("=" * 60)
    
    return results


def main():
    """Main entry point."""
    args = parse_args()
    
    setup_logging("INFO")
    logger = get_logger("eval")
    
    # Determine test path
    test_path = args.test_path
    text_column = "text"
    label_column = "label"
    
    if test_path is None and args.config:
        config = TrainingConfig.from_yaml(args.config)
        test_path = config.test_path
        text_column = config.text_column
        label_column = config.label_column
    
    if test_path is None:
        raise ValueError("test_path must be specified via --test_path or in config")
    
    # Run evaluation
    results = evaluate_model(
        checkpoint_dir=args.checkpoint,
        test_path=test_path,
        batch_size=args.batch_size,
        device=args.device,
        text_column=text_column,
        label_column=label_column,
    )
    
    # Save results
    output_dir = args.output_dir or args.checkpoint
    output_path = os.path.join(output_dir, "eval_metrics.json")
    results.save(output_path)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
