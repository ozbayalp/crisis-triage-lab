# Model Card: Neural Triage Classifier

> **Document Version:** 1.0  
> **Last Updated:** November 2024  
> **Status:** Research Prototype

---

## Model Overview

| Property | Value |
|----------|-------|
| **Model Name** | Neural Triage Classifier |
| **Version** | 0.1.0-research |
| **Type** | Text Classification (Sequence Classification) |
| **Framework** | PyTorch / Hugging Face Transformers |
| **License** | Research use only |

### Purpose

The Neural Triage Classifier is a research prototype designed to classify text inputs into qualitative risk bands for **simulation and research purposes only**. It demonstrates the technical feasibility of applying transformer-based NLP to triage-like classification tasks.

### Non-Goals

This model is explicitly **NOT** designed for:

- Clinical diagnosis or treatment decisions
- Real-time crisis intervention
- Deployment in healthcare settings
- Use as a medical device or clinical decision support tool
- Replacement of trained mental health professionals

---

## Intended Use

### Primary Use Cases

| Use Case | Description |
|----------|-------------|
| **Research** | Exploring triage behavior in simulated environments |
| **Pipeline Testing** | Validating end-to-end system architecture |
| **Benchmarking** | Comparing model architectures on synthetic tasks |
| **Education** | Understanding ML system design for sensitive domains |
| **Demonstration** | Showcasing privacy-first ML pipeline patterns |

### Out-of-Scope Uses

The following uses are **strictly prohibited**:

- Real crisis hotline support or triage
- Clinical decision-making or treatment planning
- Any context involving real patients or crisis callers
- Production deployment in healthcare environments
- Automated decision-making without human oversight
- Any use that could impact individual health outcomes

---

## Training Data

### Data Sources

The model is trained on **synthetic and/or publicly available text classification datasets**. These may include:

- Synthetically generated text samples with assigned risk labels
- Public sentiment/emotion classification datasets (repurposed for research)
- Manually curated examples for specific risk categories

### Data Exclusions

**The training data explicitly DOES NOT include:**

- Real crisis hotline transcripts
- Actual patient communications
- Protected health information (PHI)
- Any data from real mental health interventions
- Data from individuals experiencing actual crises

### Label Schema

| Label | Description | Training Distribution* |
|-------|-------------|------------------------|
| `LOW` | No immediate risk indicators | ~40% |
| `MEDIUM` | Mild distress signals, monitoring appropriate | ~30% |
| `HIGH` | Significant distress, attention required | ~20% |
| `IMMINENT` | Critical indicators, immediate attention | ~10% |

*Distribution is approximate and intentionally imbalanced to reflect research goals.

### Data Limitations

- **Synthetic nature**: Training data may not reflect the complexity of real crisis communications
- **Cultural bias**: Data may underrepresent non-Western expressions of distress
- **Linguistic coverage**: Primarily English; limited multilingual support
- **Temporal drift**: No ongoing updates to reflect evolving language patterns

---

## Model Architecture

### Base Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Neural Triage Classifier                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Input Text                                                    │
│       │                                                         │
│       ▼                                                         │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              Tokenizer (WordPiece)                      │   │
│   │              max_length: 512 tokens                     │   │
│   └─────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │         Pretrained Transformer Backbone                 │   │
│   │         (e.g., DistilBERT, 66M parameters)              │   │
│   └─────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              Classification Head                        │   │
│   │              Linear(768 → 4) + Softmax                  │   │
│   └─────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│   Output: P(low), P(medium), P(high), P(imminent)               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Technical Specifications

| Component | Specification |
|-----------|---------------|
| **Backbone** | DistilBERT-base-uncased (or similar) |
| **Hidden Size** | 768 |
| **Attention Heads** | 12 |
| **Parameters** | ~66M (backbone) + ~3K (head) |
| **Input Length** | 512 tokens maximum |
| **Output Classes** | 4 (LOW, MEDIUM, HIGH, IMMINENT) |
| **Activation** | Softmax (classification head) |

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Batch Size | 16–32 |
| Epochs | 3–5 |
| Warmup Steps | 10% of total |
| Weight Decay | 0.01 |
| Gradient Clipping | 1.0 |

---

## Evaluation

### Metrics

The model is evaluated using standard classification metrics:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions / total predictions |
| **Macro F1** | Unweighted average F1 across all classes |
| **Per-Class Precision** | TP / (TP + FP) for each class |
| **Per-Class Recall** | TP / (TP + FN) for each class |
| **Confusion Matrix** | Class-wise prediction distribution |

### Example Results (Placeholder)

> ⚠️ **Note**: The following numbers are **illustrative placeholders** based on synthetic validation data. They do not represent performance on real crisis communications.

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| LOW | 0.85* | 0.88* | 0.86* | 400* |
| MEDIUM | 0.72* | 0.68* | 0.70* | 300* |
| HIGH | 0.78* | 0.75* | 0.76* | 200* |
| IMMINENT | 0.82* | 0.80* | 0.81* | 100* |

| Overall Metric | Value |
|----------------|-------|
| **Accuracy** | 0.80* |
| **Macro F1** | 0.78* |
| **Weighted F1** | 0.79* |

*Values marked with asterisk (*) are placeholders for illustration.

### Evaluation Limitations

- Evaluation is performed on synthetic/held-out data from the same distribution
- **No evaluation on real crisis data has been performed**
- Cross-domain generalization is unknown and likely poor
- Adversarial robustness has not been tested

---

## Limitations

### Technical Limitations

| Limitation | Impact |
|------------|--------|
| **Domain Shift** | Model trained on synthetic data; real crisis language differs significantly |
| **Text-Only** | Does not incorporate prosodic/acoustic signals (separate pipeline component) |
| **English-Centric** | Limited or no support for other languages |
| **Context-Free** | Each input is processed independently; no conversation history |
| **Fixed Vocabulary** | Cannot adapt to evolving slang or crisis-specific terminology |

### Known Failure Modes

- **Sarcasm and irony**: May misclassify sarcastic expressions of distress
- **Coded language**: Unlikely to detect euphemisms or indirect expressions
- **Cultural variation**: Western-centric training data limits cross-cultural validity
- **Brevity**: Very short inputs may lack sufficient signal
- **Adversarial inputs**: Susceptible to intentional manipulation

### What This Model Cannot Do

- Predict actual risk of self-harm or suicide
- Replace clinical judgment or professional assessment
- Provide reliable outputs for real individuals in crisis
- Generalize to populations not represented in training data
- Guarantee safety-critical performance

---

## Ethical Considerations

### Potential Risks

| Risk | Mitigation |
|------|------------|
| **Misinterpretation as clinical tool** | Explicit disclaimers; research-only framing |
| **Overreliance on model outputs** | Human oversight required; no autonomous decisions |
| **Harm from false negatives** | Never deployed in real crisis contexts |
| **Harm from false positives** | No real-world consequences in research setting |
| **Dual use for surveillance** | Local-only processing; no data exfiltration |

### Regulatory Status

- **Not a medical device** under any jurisdiction
- **No FDA, CE, or equivalent clearance**
- **Not compliant** with clinical decision support regulations
- **Research use only** — not for production deployment

### Human Oversight Requirements

Any hypothetical future deployment (not recommended) would require:

1. Trained human reviewers for all outputs
2. Clinical validation studies with appropriate IRB oversight
3. Regulatory approval for intended use context
4. Ongoing monitoring and bias auditing
5. Clear user communication about system limitations

---

## Responsible AI Practices

### Bias Considerations

The model may exhibit biases stemming from:

- Training data composition and labeling decisions
- Pretrained backbone's existing biases
- Underrepresentation of certain demographics or expressions

No formal bias audit has been conducted.

### Transparency

- Model architecture and training process are documented
- Limitations are explicitly stated
- Intended use is clearly scoped to research

### Accountability

This model is a personal research project. It is not:

- A product of any commercial organization
- Endorsed by any healthcare institution
- Supported by any clinical expertise

---

## Citation

If referencing this work in academic contexts:

```
CrisisTriage AI: Neural Triage Classifier (Research Prototype)
Personal research project, 2024
https://github.com/[username]/CrisisTriage
```

---

## Contact & Ownership

| Field | Value |
|-------|-------|
| **Author** | [Your Name] |
| **Affiliation** | Personal Research Project |
| **Contact** | [Your Email] |
| **Repository** | [GitHub URL] |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | November 2024 | Initial model card |

---

> **Final Note**: This model exists to demonstrate ML engineering practices in sensitive domains. It is not, and should never be, used for real crisis intervention. If you or someone you know is in crisis, please contact your local emergency services or a crisis hotline.
