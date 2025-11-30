# CrisisTriage AI — Neural Triage Classifier Training Blueprint

**Document Classification:** Internal Research Notes  
**Version:** 1.0  
**Audience:** ML Research Engineers, PhD-level practitioners  
**Scope:** Text-only classifier (v1) + Multimodal prosody-enhanced classifier (v2)

---

# 1. TRAINING PHILOSOPHY

## 1.1 Why Triage Classification is Hard

Crisis triage occupies a unique position in NLP:
- **Emotionally dense**: Same text can span multiple risk levels depending on context
- **Deliberately obscured**: Speakers use euphemism and coded language
- **Culturally heterogeneous**: Distress expression varies across backgrounds
- **Sparse high-risk signal**: IMMINENT cases are rare, creating severe class imbalance

## 1.2 Non-Negotiable Sensitivity to HIGH/IMMINENT

The cost matrix is radically asymmetric:

| Prediction | True Label | Consequence |
|------------|------------|-------------|
| LOW | IMMINENT | **Catastrophic** — missed crisis |
| HIGH | LOW | Over-caution — acceptable |

**Design principle**: A model with 85% accuracy but 95% IMMINENT recall beats one with 95% accuracy but 70% IMMINENT recall.

## 1.3 Ordinal Nature of Labels

```
LOW ←1→ MEDIUM ←2→ HIGH ←3→ IMMINENT
```

Predicting LOW when truth is IMMINENT (distance 3) is far worse than MEDIUM vs HIGH (distance 1).

## 1.4 Why Calibration > Accuracy

- Triage operators need meaningful confidence signals
- Threshold-based escalation requires calibrated probabilities
- Target: ECE < 0.05 after temperature scaling

---

# 2. DATASET STRATEGY

## 2.1 Target Sizes

| Stage | Minimum | Recommended |
|-------|---------|-------------|
| Prototype | 5,000 | 10,000 |
| Baseline | 20,000 | 50,000 |
| Production | 80,000 | 150,000 |
| Research | 200,000 | 500,000 |

## 2.2 Class Distribution (Safety-Weighted)

| Class | Natural | Target Training |
|-------|---------|-----------------|
| LOW | ~60% | 30% |
| MEDIUM | ~25% | 30% |
| HIGH | ~12% | 25% |
| IMMINENT | ~3% | 15% |

## 2.3 Data Sources

1. **LLM-Generated** (40%): GPT-4/Claude with persona conditioning
2. **Template Augmentation** (25%): Parameterized slot-filling
3. **Paraphrase Augmentation** (20%): Back-translation, LLM paraphrasing
4. **Adversarial Samples** (15%): Coded language, sarcasm, boundary cases

## 2.4 Critical Augmentations

- **Negation inversion**: "I don't want to hurt myself" ↔ "I want to hurt myself"
- **Typo injection**: "i cant do this anymor im scred"
- **Euphemism coverage**: "catch the bus", "unalive", "kms"
- **Sarcasm**: "Oh yeah, today was just GREAT"
- **Dialectal variants**: AAVE, British, regional patterns

## 2.5 Safety Constraints

- **NO real crisis hotline data** — ever
- **NO scraping suicide forums** — ethical violation
- All data must be synthetic with clear provenance

---

# 3. LABEL TAXONOMY

## 3.1 Definitions

| Level | Definition | Key Indicators |
|-------|------------|----------------|
| **LOW** | Stable, coping, seeking connection | Gratitude, forward planning |
| **MEDIUM** | Distressed but no danger to self | Life stressors, difficulty coping |
| **HIGH** | Suicidal ideation, hopelessness | Ideation, burden, isolation |
| **IMMINENT** | Means, plan, intent, active crisis | Timeline, means access, goodbye statements |

## 3.2 Borderline Rules

- **Any mention of self-harm** → escalate to HIGH
- **Ideation + (plan OR means OR intent)** → IMMINENT
- **Conflicting signals** → label based on most concerning statement

---

# 4. MODEL ARCHITECTURE

## 4.1 Recommended: DeBERTa-v3-small

| Model | Params | Speed | Quality |
|-------|--------|-------|---------|
| DistilBERT | 66M | Fastest | Good |
| **DeBERTa-v3-small** | 44M | Fast | **Best** |
| DeBERTa-v3-base | 184M | Slow | Excellent |

## 4.2 Hyperparameters

```yaml
learning_rate: 2.0e-5
warmup_ratio: 0.1
batch_size: 32
epochs: 8-10
max_seq_length: 256
weight_decay: 0.01
max_grad_norm: 1.0
label_smoothing: 0.05
fp16: true
```

## 4.3 Regularization Stack

1. Dropout (0.1)
2. Weight decay (0.01)
3. Label smoothing (0.05)
4. Gradient clipping (1.0)
5. Early stopping (patience=3)
6. Layerwise LR decay (0.95)

---

# 5. LOSS FUNCTION & METRICS

## 5.1 Recommended: Focal Loss

```python
class_weights = [0.5, 0.8, 1.5, 2.5]  # LOW, MEDIUM, HIGH, IMMINENT
focal_gamma = 2.0
```

Focal loss down-weights easy examples and focuses on hard cases.

## 5.2 Primary Metrics

| Metric | Target |
|--------|--------|
| Macro F1 | > 0.80 |
| HIGH Recall | > 0.90 |
| IMMINENT Recall | > 0.95 |
| ECE | < 0.05 |

## 5.3 Safety Metrics

- **FNSS** (False Negative Severity Score): Quadratic penalty for under-prediction
- **Risk Distance Error**: Mean ordinal distance between pred and true

---

# 6. TRAINING PIPELINE

## 6.1 Training Loop

- AMP/FP16 enabled
- Evaluation every epoch
- Early stopping on val_f1_macro (patience=3)
- Save best checkpoint by F1-macro, tiebreak by IMMINENT recall

## 6.2 Required Artifacts

```
outputs/experiment/
├── config.yaml
├── model/model.safetensors
├── tokenizer/
├── label2id.json
├── training_log.jsonl
├── eval_results.json
└── training_metadata.json
```

---

# 7. EVALUATION FRAMEWORK

## 7.1 Standard Metrics

- Confusion matrix (inspect HIGH→LOW, IMMINENT→LOW errors)
- Per-class P/R/F1

## 7.2 Domain-Specific Test Sets

| Test Set | Size | Purpose |
|----------|------|---------|
| Negation | 500 | "don't want" vs "want" |
| Euphemism | 500 | Coded language detection |
| Boundary | 500 | HIGH vs IMMINENT edge cases |
| Typo/Slang | 500 | Degraded text robustness |

## 7.3 Calibration

- ECE and reliability diagrams
- Temperature scaling post-training
- Per-class threshold tuning for target recall

---

# 8. INTERPRETABILITY

- **Integrated Gradients**: Token-level attribution scores
- **Attention Rollout**: Aggregated attention visualization
- **Risk Factor Mapping**: Map high-attribution tokens to categories (ideation, hopelessness, burden, means, intent)

---

# 9. DEPLOYMENT CONSTRAINTS

## 9.1 Latency Budget

| Component | Target |
|-----------|--------|
| Tokenization | 1ms |
| Model forward | 20ms |
| Total | <50ms |

## 9.2 Optimizations

- ONNX export for 2-3x speedup
- INT8 quantization for 4x size reduction
- Batch inference with dynamic batching

## 9.3 Fail-Safe Logic

```python
if confidence < 0.6:
    return "high", "low_confidence"  # Escalate when uncertain
if error:
    return "high", "error"  # Never fail to LOW
```

---

# 10. REPRODUCIBILITY

- Fixed seeds (random, numpy, torch, PYTHONHASHSEED)
- Deterministic dataloaders with worker seeding
- Dataset checksums stored in metadata
- Git commit SHA in training metadata
- Docker environment for exact reproduction

---

# 11. EXPERIMENT DIRECTORY LAYOUT

```
ml/
├── data/
│   ├── synthetic/          # Train/val/test CSVs
│   └── adversarial/        # Stress test sets
├── src/
│   ├── data/               # Datasets, augmentation
│   ├── models/             # Classifier, losses
│   ├── training/           # Training script
│   └── eval/               # Evaluation, stress tests
├── experiments/
│   ├── configs/            # YAML configs
│   ├── runs/               # Training runs (gitignored)
│   ├── ablations/          # Ablation studies
│   └── multimodal/         # v2 experiments
└── outputs/                # Production models
```

---

# 12. PRE-TRAINING CHECKLIST

- [ ] Verify dataset paths exist and are readable
- [ ] Check label distribution matches expected
- [ ] Validate no label leakage between splits
- [ ] Run tokenizer coverage report
- [ ] Verify GPU memory sufficient for batch size
- [ ] Confirm config hash matches expected
- [ ] Check adversarial test sets prepared
- [ ] Verify random seed is set

---

# 13. POST-TRAINING CHECKLIST

- [ ] Inspect confusion matrix for dangerous errors
- [ ] Verify IMMINENT recall > 0.92
- [ ] Check calibration curve and compute ECE
- [ ] Run all adversarial test sets
- [ ] Apply temperature scaling if ECE > 0.05
- [ ] Export ONNX model
- [ ] Save all artifacts with metadata
- [ ] Update MODEL_CARD.md with results
- [ ] Document failure cases

---

# 14. MULTIMODAL EXTENSION (v2)

## 14.1 Prosody Features

| Feature | Description |
|---------|-------------|
| pitch_mean, pitch_std, pitch_range | F0 statistics |
| energy_mean, energy_std | Volume dynamics |
| speech_rate | Syllables per second |
| pause_ratio | Silence proportion |
| jitter, shimmer | Voice quality |
| spectral_centroid | Timbral brightness |

Normalize per-speaker with z-scoring. Temporal pooling: mean, std, min, max over utterance.

## 14.2 Fusion Architectures

| Architecture | Description | Recommendation |
|--------------|-------------|----------------|
| **Early Fusion** | Prosody vector appended to CLS embedding | Simple, effective |
| **Late Fusion** | Two towers, concatenate before classifier | Modular, debuggable |
| **Attention Fusion** | Prosody-informed attention gating | Best for nuance |
| **MoE Fusion** | Expert heads with learned gating | Research-grade |

**Recommended for safety-critical**: Late Fusion with confidence-weighted combination.

## 14.3 Multitask Training

```
Loss = λ₁·risk_loss + λ₂·emotion_loss + λ₃·prosody_reconstruction
```

Benefits: Regularization, richer representations
Pitfalls: Optimization balance, task interference

## 14.4 Multimodal Calibration

- Per-modality temperature scaling
- Confidence-weighted fusion: `p_final = α·p_text + (1-α)·p_multimodal` where α depends on prosody availability
- Fallback to text-only when prosody unavailable

## 14.5 Multimodal Metrics

- **Cross-modal consistency**: Agreement between text-only and multimodal predictions
- **Prosody lift**: Performance gain from adding prosody
- **Uncertainty reduction**: Entropy decrease with multimodal input

---

# 15. DRAFT SUPER-CONFIG (See separate file)

A complete YAML config has been generated at:
`ml/experiments/configs/deberta_focal_prod.yaml`

Key settings:
- Model: `microsoft/deberta-v3-small`
- Loss: Focal (γ=2.0) with safety weights [0.5, 0.8, 1.5, 2.5]
- LR: 2e-5 with cosine schedule, 10% warmup
- Epochs: 10 with early stopping (patience=3)
- FP16 enabled, batch size 32

---

# 16. SUMMARY

This blueprint provides:
- ✅ Complete text-only model training guidelines
- ✅ Multimodal prosody fusion architecture
- ✅ Safety-first evaluation framework
- ✅ Reproducibility requirements
- ✅ Pre/post-training checklists
- ✅ Experiment organization structure

**Next steps:**
1. Generate synthetic training data (80k+ samples)
2. Train baseline with debug config
3. Scale to full config
4. Evaluate on adversarial sets
5. Calibrate and export

---

*This document follows research standards suitable for ACL/EMNLP/NeurIPS submission.*
