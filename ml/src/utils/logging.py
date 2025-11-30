"""
CrisisTriage AI - ML Logging Utilities

Structured logging for training, evaluation, and experiments.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    name: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging with console and optional file output.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        name: Logger name (default: "ml")
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name or "ml")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "ml") -> logging.Logger:
    """Get or create a logger with the specified name."""
    return logging.getLogger(name)


@dataclass
class TrainingMetrics:
    """Metrics for a single training step or epoch."""
    
    epoch: int
    step: int
    loss: float
    learning_rate: float
    accuracy: Optional[float] = None
    f1_macro: Optional[float] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class EvalMetrics:
    """Metrics from model evaluation."""
    
    epoch: int
    loss: float
    accuracy: float
    f1_macro: float
    f1_per_class: dict[str, float]
    precision_per_class: dict[str, float]
    recall_per_class: dict[str, float]
    confusion_matrix: list[list[int]]
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


class MetricsLogger:
    """
    Logger for training and evaluation metrics.
    
    Writes metrics to JSONL file for easy parsing and analysis.
    """
    
    def __init__(self, output_dir: str, filename: str = "metrics.jsonl"):
        """
        Initialize metrics logger.
        
        Args:
            output_dir: Directory to write metrics file
            filename: Name of metrics file
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.output_dir / filename
        self.logger = get_logger("metrics")
    
    def log_training_step(self, metrics: TrainingMetrics) -> None:
        """Log metrics from a training step."""
        self._append_jsonl({
            "type": "train_step",
            **asdict(metrics),
        })
    
    def log_training_epoch(self, metrics: TrainingMetrics) -> None:
        """Log metrics from a training epoch."""
        self._append_jsonl({
            "type": "train_epoch",
            **asdict(metrics),
        })
        self.logger.info(
            f"Epoch {metrics.epoch} | Loss: {metrics.loss:.4f} | "
            f"Acc: {metrics.accuracy:.4f} | F1: {metrics.f1_macro:.4f}"
        )
    
    def log_eval(self, metrics: EvalMetrics) -> None:
        """Log evaluation metrics."""
        self._append_jsonl({
            "type": "eval",
            **asdict(metrics),
        })
        self.logger.info(
            f"Eval Epoch {metrics.epoch} | Loss: {metrics.loss:.4f} | "
            f"Acc: {metrics.accuracy:.4f} | F1: {metrics.f1_macro:.4f}"
        )
    
    def log_config(self, config: dict[str, Any]) -> None:
        """Log training configuration."""
        self._append_jsonl({
            "type": "config",
            "timestamp": datetime.utcnow().isoformat(),
            "config": config,
        })
    
    def log_artifact(self, artifact_path: str, metric_name: str, metric_value: float) -> None:
        """Log model artifact save event."""
        self._append_jsonl({
            "type": "artifact",
            "timestamp": datetime.utcnow().isoformat(),
            "artifact_path": artifact_path,
            "metric_name": metric_name,
            "metric_value": metric_value,
        })
    
    def _append_jsonl(self, data: dict[str, Any]) -> None:
        """Append a JSON line to the metrics file."""
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(data) + "\n")
    
    def load_metrics(self) -> list[dict[str, Any]]:
        """Load all metrics from file."""
        metrics = []
        if self.metrics_file.exists():
            with open(self.metrics_file, "r") as f:
                for line in f:
                    if line.strip():
                        metrics.append(json.loads(line))
        return metrics
