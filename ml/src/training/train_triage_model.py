"""
CrisisTriage AI - Training Script for Triage Classifier

Config-driven training loop for the neural triage model.

IMPORTANT SAFETY NOTICE:
    This training code is for RESEARCH AND SIMULATION ONLY.
    Train only on synthetic or properly de-identified datasets.
    The resulting model is NOT suitable for real-world crisis intervention.

Usage:
    python -m ml.src.training.train_triage_model --config ml/experiments/configs/baseline_bert.yaml
    
    Or with overrides:
    python -m ml.src.training.train_triage_model --config config.yaml --learning_rate 1e-5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
    SequentialLR,
    ConstantLR,
)
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from ml.src.config import TrainingConfig, resolve_device, get_label_mappings
from ml.src.data.datasets import load_triage_dataset, create_data_loader
from ml.src.models.triage_classifier import TriageClassifier, compute_loss_with_label_smoothing
from ml.src.utils.seed import set_seed
from ml.src.utils.logging import setup_logging, get_logger, MetricsLogger, TrainingMetrics, EvalMetrics


# Suppress some warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a triage classifier model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    
    # Allow config overrides via CLI
    parser.add_argument("--model_name_or_path", type=str, help="Override model name")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--train_path", type=str, help="Override train data path")
    parser.add_argument("--val_path", type=str, help="Override validation data path")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--num_epochs", type=int, help="Override number of epochs")
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")
    parser.add_argument("--seed", type=int, help="Override random seed")
    
    return parser.parse_args()


def apply_cli_overrides(config: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
    """Apply command-line overrides to config."""
    override_fields = [
        "model_name_or_path", "output_dir", "train_path", "val_path",
        "batch_size", "num_epochs", "learning_rate", "seed"
    ]
    
    for field in override_fields:
        value = getattr(args, field, None)
        if value is not None:
            setattr(config, field, value)
    
    return config


def create_optimizer(model: TriageClassifier, config: TrainingConfig) -> AdamW:
    """Create optimizer with weight decay."""
    # Don't apply weight decay to bias and LayerNorm
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    
    return AdamW(optimizer_grouped_parameters, lr=config.learning_rate)


def create_scheduler(
    optimizer: AdamW,
    config: TrainingConfig,
    num_training_steps: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler."""
    # Calculate warmup steps
    if config.warmup_steps is not None:
        warmup_steps = config.warmup_steps
    else:
        warmup_steps = int(num_training_steps * config.warmup_ratio)
    
    if config.scheduler == "linear":
        # Linear warmup then linear decay
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        decay_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=num_training_steps - warmup_steps,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_steps],
        )
    
    elif config.scheduler == "cosine":
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - warmup_steps,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
    
    else:  # constant
        return ConstantLR(optimizer, factor=1.0, total_iters=num_training_steps)


def train_epoch(
    model: TriageClassifier,
    train_loader,
    optimizer: AdamW,
    scheduler,
    config: TrainingConfig,
    device: str,
    epoch: int,
    global_step: int,
    metrics_logger: MetricsLogger,
) -> tuple[float, float, float, int]:
    """
    Train for one epoch.
    
    Returns:
        Tuple of (avg_loss, accuracy, f1, global_step)
    """
    model.train_mode()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}",
        leave=False,
    )
    
    for batch_idx, (input_ids, attention_mask, labels) in enumerate(progress_bar):
        # Move to device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, labels)
        
        # Loss with optional label smoothing
        if config.label_smoothing > 0:
            loss = compute_loss_with_label_smoothing(
                outputs["logits"], labels, config.label_smoothing
            )
        else:
            loss = outputs["loss"]
        
        # Handle gradient accumulation
        loss = loss / config.gradient_accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            global_step += 1
            
            # Log every N steps
            if global_step % config.logging_steps == 0:
                current_lr = scheduler.get_last_lr()[0]
                metrics_logger.log_training_step(TrainingMetrics(
                    epoch=epoch,
                    step=global_step,
                    loss=loss.item() * config.gradient_accumulation_steps,
                    learning_rate=current_lr,
                ))
        
        # Track metrics
        total_loss += loss.item() * config.gradient_accumulation_steps
        
        with torch.no_grad():
            preds = torch.argmax(outputs["logits"], dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({
            "loss": f"{loss.item() * config.gradient_accumulation_steps:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        })
    
    # Compute epoch metrics
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    
    return avg_loss, accuracy, f1, global_step


def evaluate(
    model: TriageClassifier,
    eval_loader,
    config: TrainingConfig,
    device: str,
    epoch: int,
) -> EvalMetrics:
    """
    Evaluate the model on validation/test set.
    
    Returns:
        EvalMetrics with all evaluation results
    """
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )
    
    model.eval_mode()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(eval_loader, desc="Evaluating", leave=False):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            outputs = model(input_ids, attention_mask, labels)
            total_loss += outputs["loss"].item()
            
            preds = torch.argmax(outputs["logits"], dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    avg_loss = total_loss / len(eval_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    
    # Per-class metrics
    precision, recall, f1_per_class, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # Get label names
    id2label = model.id2label
    label_names = [id2label[i] for i in range(len(id2label))]
    
    f1_dict = {label_names[i]: float(f1_per_class[i]) for i in range(len(label_names))}
    precision_dict = {label_names[i]: float(precision[i]) for i in range(len(label_names))}
    recall_dict = {label_names[i]: float(recall[i]) for i in range(len(label_names))}
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds).tolist()
    
    return EvalMetrics(
        epoch=epoch,
        loss=avg_loss,
        accuracy=accuracy,
        f1_macro=f1_macro,
        f1_per_class=f1_dict,
        precision_per_class=precision_dict,
        recall_per_class=recall_dict,
        confusion_matrix=cm,
    )


def train(config: TrainingConfig) -> str:
    """
    Main training function.
    
    Args:
        config: Training configuration
        
    Returns:
        Path to best model checkpoint
    """
    logger = get_logger("training")
    logger.info("=" * 60)
    logger.info("CrisisTriage AI - Model Training")
    logger.info("RESEARCH/SIMULATION ONLY - NOT FOR CLINICAL USE")
    logger.info("=" * 60)
    
    # Set up reproducibility
    set_seed(config.seed, config.deterministic)
    logger.info(f"Random seed: {config.seed}")
    
    # Resolve device
    device = resolve_device(config.device)
    logger.info(f"Using device: {device}")
    
    # Get label mappings
    label2id, id2label = get_label_mappings(config.num_labels)
    logger.info(f"Labels: {label2id}")
    
    # Initialize model
    logger.info(f"Loading model: {config.model_name_or_path}")
    model = TriageClassifier.from_pretrained(
        config.model_name_or_path,
        num_labels=config.num_labels,
        label2id=label2id,
        id2label=id2label,
        max_seq_length=config.max_seq_length,
    )
    model.to(device)
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = load_triage_dataset(config, "train", model.tokenizer, label2id)
    logger.info(f"Train samples: {len(train_dataset)}")
    
    val_dataset = None
    if config.val_path:
        val_dataset = load_triage_dataset(config, "val", model.tokenizer, label2id)
        logger.info(f"Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = create_data_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    
    val_loader = None
    if val_dataset:
        val_loader = create_data_loader(
            val_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
        )
    
    # Calculate training steps
    num_training_steps = len(train_loader) * config.num_epochs
    logger.info(f"Total training steps: {num_training_steps}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, num_training_steps)
    
    # Set up logging
    metrics_logger = MetricsLogger(config.output_dir)
    metrics_logger.log_config(config.to_dict())
    
    if config.log_to_file:
        log_file = os.path.join(config.output_dir, "training.log")
        setup_logging("INFO", log_file)
    
    # Training loop
    best_metric = 0.0
    best_model_path = None
    global_step = 0
    
    logger.info("Starting training...")
    
    for epoch in range(1, config.num_epochs + 1):
        # Train
        train_loss, train_acc, train_f1, global_step = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
            epoch=epoch,
            global_step=global_step,
            metrics_logger=metrics_logger,
        )
        
        # Log training metrics
        metrics_logger.log_training_epoch(TrainingMetrics(
            epoch=epoch,
            step=global_step,
            loss=train_loss,
            learning_rate=scheduler.get_last_lr()[0],
            accuracy=train_acc,
            f1_macro=train_f1,
        ))
        
        # Evaluate
        if val_loader is not None:
            eval_metrics = evaluate(model, val_loader, config, device, epoch)
            metrics_logger.log_eval(eval_metrics)
            
            # Check if best model
            current_metric = eval_metrics.f1_macro
            if current_metric > best_metric:
                best_metric = current_metric
                best_model_path = os.path.join(config.output_dir, "best_model")
                model.save_pretrained(
                    best_model_path,
                    best_metric=best_metric,
                    best_metric_name="f1_macro",
                )
                metrics_logger.log_artifact(best_model_path, "f1_macro", best_metric)
                logger.info(f"New best model saved! F1: {best_metric:.4f}")
        else:
            # Save based on training loss if no validation
            best_model_path = os.path.join(config.output_dir, "best_model")
            model.save_pretrained(best_model_path)
    
    # Save final model
    final_model_path = os.path.join(config.output_dir, "final_model")
    model.save_pretrained(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best F1: {best_metric:.4f}")
    logger.info(f"Best model: {best_model_path}")
    logger.info("=" * 60)
    
    return best_model_path or final_model_path


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load config
    config = TrainingConfig.from_yaml(args.config)
    config = apply_cli_overrides(config, args)
    
    # Validate required paths
    if config.train_path is None:
        raise ValueError("train_path must be specified in config or via --train_path")
    
    # Run training
    train(config)


if __name__ == "__main__":
    main()
