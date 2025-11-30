# CrisisTriage AI - ML Package

> Machine learning training, experimentation, and model development.

## ⚠️ IMPORTANT SAFETY NOTICE

**This ML system is for RESEARCH AND SIMULATION ONLY.**

- NOT a medical device
- NOT suitable for real-world crisis intervention
- Use only with synthetic or properly de-identified datasets
- Model predictions should NEVER be used as the sole basis for crisis response

---

## Overview

This package contains the ML training and evaluation pipeline for text-based triage classification:

- **Config-driven experiments** with YAML configuration files
- **HuggingFace transformers** for text classification (DistilBERT, etc.)
- **Training and evaluation scripts** with comprehensive metrics
- **Model artifacts** compatible with backend integration

## Structure

```
ml/
├── experiments/
│   └── configs/              # YAML experiment configs
│       ├── baseline_bert.yaml
│       └── debug_small.yaml
├── notebooks/                # Exploratory analysis
├── outputs/                  # Trained models (gitignored)
├── src/
│   ├── config.py             # TrainingConfig, TriageModelArtifact
│   ├── data/
│   │   └── datasets.py       # TriageDataset, data loading
│   ├── models/
│   │   └── triage_classifier.py  # TriageClassifier wrapper
│   ├── training/
│   │   └── train_triage_model.py # Training script
│   ├── eval/
│   │   └── evaluate_triage_model.py # Evaluation script
│   └── utils/
│       ├── logging.py        # MetricsLogger
│       └── seed.py           # Reproducibility
└── requirements.txt
```

---

## Quick Start

### 1. Install Dependencies

```bash
cd ml
pip install -r requirements.txt
```

### 2. Prepare Dataset

Create a CSV file with `text` and `label` columns:

```csv
text,label
"I've been feeling overwhelmed lately",medium
"I'm not sure what to do anymore",medium
"I feel hopeless and want to end it",high
"Thank you for listening to me",low
```

Labels must be one of: `low`, `medium`, `high`, `imminent`

### 3. Configure Experiment

Edit `experiments/configs/baseline_bert.yaml`:

```yaml
# Data paths (required)
train_path: "path/to/train.csv"
val_path: "path/to/val.csv"
test_path: "path/to/test.csv"

# Model
model_name_or_path: "distilbert-base-uncased"
num_labels: 4

# Training
batch_size: 16
num_epochs: 3
learning_rate: 2.0e-5

# Output
output_dir: "./outputs/my_experiment"
```

### 4. Train Model

```bash
# From project root
python -m ml.src.training.train_triage_model \
    --config ml/experiments/configs/baseline_bert.yaml

# With CLI overrides
python -m ml.src.training.train_triage_model \
    --config ml/experiments/configs/baseline_bert.yaml \
    --learning_rate 1e-5 \
    --num_epochs 5
```

### 5. Evaluate Model

```bash
python -m ml.src.eval.evaluate_triage_model \
    --checkpoint ./outputs/my_experiment/best_model \
    --test_path path/to/test.csv
```

### 6. Use in Backend

Set environment variables:

```bash
export TRIAGE_MODEL_BACKEND=neural
export NEURAL_MODEL_DIR=./ml/outputs/my_experiment/best_model
```

Or in `.env`:

```
TRIAGE_MODEL_BACKEND=neural
NEURAL_MODEL_DIR=./ml/outputs/my_experiment/best_model
```

---

## Configuration Reference

### TrainingConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name_or_path` | str | `distilbert-base-uncased` | HuggingFace model ID or path |
| `num_labels` | int | `4` | Number of risk classes |
| `max_seq_length` | int | `256` | Maximum token sequence length |
| `train_path` | str | - | Path to training CSV/JSON |
| `val_path` | str | - | Path to validation data |
| `test_path` | str | - | Path to test data |
| `batch_size` | int | `16` | Training batch size |
| `num_epochs` | int | `3` | Number of training epochs |
| `learning_rate` | float | `2e-5` | Initial learning rate |
| `weight_decay` | float | `0.01` | AdamW weight decay |
| `warmup_ratio` | float | `0.1` | Warmup steps as ratio |
| `output_dir` | str | `./outputs/default` | Model save directory |
| `seed` | int | `42` | Random seed |

---

## Model Artifact

After training, the output directory contains:

```
outputs/my_experiment/best_model/
├── config.json           # HuggingFace model config
├── model.safetensors     # Model weights
├── tokenizer.json        # Tokenizer
├── tokenizer_config.json
├── vocab.txt
├── artifact.json         # CrisisTriage metadata
└── special_tokens_map.json
```

The `artifact.json` contains metadata needed by the backend:

```json
{
  "artifact_dir": "/path/to/model",
  "model_name": "distilbert-base-uncased",
  "num_labels": 4,
  "label2id": {"low": 0, "medium": 1, "high": 2, "imminent": 3},
  "id2label": {"0": "low", "1": "medium", "2": "high", "3": "imminent"},
  "max_seq_length": 256,
  "created_at": "2024-01-15T10:30:00",
  "best_metric": 0.85,
  "best_metric_name": "f1_macro",
  "disclaimer": "RESEARCH/SIMULATION ONLY..."
}
```

---

## Backend Integration

The backend can use the trained model by setting:

```python
# In backend/app/config.py (via environment)
TRIAGE_MODEL_BACKEND=neural
NEURAL_MODEL_DIR=./ml/outputs/my_experiment/best_model
```

This causes `create_pipeline()` to instantiate `NeuralTriageModel` instead of `DummyTriageModel`.

### Inference Flow

```
Frontend → WebSocket/REST API
                ↓
         TriagePipeline.process_text_message()
                ↓
         NeuralTriageModel.predict_async()
                ↓
         TriageClassifier (HuggingFace)
                ↓
         Softmax → RiskLevel + Confidence
                ↓
         TriageResult → Frontend
```

---

## Evaluation Metrics

The evaluation script computes:

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| F1 (macro) | Unweighted mean of per-class F1 |
| F1 (weighted) | Weighted mean by support |
| Per-class Precision | True positives / predicted positives |
| Per-class Recall | True positives / actual positives |
| Confusion Matrix | Full NxN matrix |
| ROC-AUC | Area under ROC curve (one-vs-rest) |

Results are saved to `eval_metrics.json` in the checkpoint directory.

---

## Data Format

### CSV Format

```csv
text,label
"Message content here",low
"Another message",high
```

### JSON Format

```json
[
  {"text": "Message content", "label": "low"},
  {"text": "Another message", "label": "high"}
]
```

### JSONL Format

```jsonl
{"text": "Message content", "label": "low"}
{"text": "Another message", "label": "high"}
```

---

## Dependencies

Key packages (see `requirements.txt`):

- **torch** >= 2.1.0
- **transformers** >= 4.35.0
- **scikit-learn** >= 1.3.0
- **pyyaml** >= 6.0.0
- **tqdm** >= 4.66.0

---

## Privacy & Ethics

### Training Data Guidelines

1. **Never use real crisis hotline data** without proper IRB approval, consent, and data governance
2. Use synthetic or publicly available de-identified datasets
3. Ensure training data preprocessing removes all identifiers
4. Document data provenance in experiment configs

### Model Limitations

- Trained on limited data; may not generalize
- Text-only; does not capture tone, prosody, or context
- Predictions are probabilistic estimates, not clinical assessments
- Should only be used with human oversight in research settings

---

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or use gradient accumulation:

```yaml
batch_size: 8
gradient_accumulation_steps: 2
```

### Model Not Loading in Backend

1. Check that `artifact.json` exists in the model directory
2. Ensure `transformers` and `torch` are installed in the backend environment
3. Check logs for specific error messages

### Training Loss Not Decreasing

- Try lower learning rate (1e-5 instead of 2e-5)
- Ensure data labels are correct
- Check for class imbalance in training data
