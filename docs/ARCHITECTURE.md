# CrisisTriage AI — System Architecture

> Detailed technical architecture for the real-time mental health triage platform.

## Table of Contents

1. [Overview](#overview)
2. [Design Principles](#design-principles)
3. [System Components](#system-components)
4. [Data Flow](#data-flow)
5. [Processing Pipeline](#processing-pipeline)
6. [Model Architecture](#model-architecture)
7. [WebSocket Protocol](#websocket-protocol)
8. [Privacy Architecture](#privacy-architecture)
9. [Scalability Considerations](#scalability-considerations)
10. [Future Extensions](#future-extensions)

---

## Overview

CrisisTriage AI is a real-time, privacy-first platform that performs multimodal triage on simulated mental health hotline conversations. The system ingests audio, extracts text and prosody features, and produces interpretable risk assessments with sub-second latency.

### Core Outputs

| Output | Type | Description |
|--------|------|-------------|
| Emotional State | Categorical | calm / anxious / distressed / panicked |
| Risk Level | Categorical | low / medium / high / imminent |
| Urgency Score | Continuous | 0–100 scale |
| Recommended Action | Categorical | continue / follow-up / escalate / intervene |
| Explanation | Structured | Feature attributions for interpretability |

---

## Design Principles

### 1. Privacy-First

- **Local inference only**: No raw audio or transcripts leave the deployment environment
- **Ephemeral by default**: Raw data is not persisted unless explicitly configured
- **Separation of concerns**: Raw data, features, and outputs are architecturally isolated

### 2. Real-Time

- **Sub-second latency**: From audio chunk to triage update
- **Streaming architecture**: WebSocket-based bidirectional communication
- **Chunked processing**: Audio processed in overlapping windows

### 3. Interpretable

- **Explainability built-in**: Every triage output includes feature attributions
- **Human-readable summaries**: Natural language explanations for operators
- **Audit trail**: Decision provenance for post-hoc analysis

### 4. Modular

- **Pluggable extractors**: Swap or add feature extractors independently
- **Model versioning**: A/B test different model versions
- **Extensible modalities**: Architecture supports adding video/text chat

---

## System Components

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                 FRONTEND                                      │
│                          (Next.js / React / TypeScript)                       │
├──────────────────────────────────────────────────────────────────────────────┤
│  • Dashboard UI           • WebSocket client         • State management       │
│  • Transcript panel       • Urgency chart           • Alert system            │
│  • Explainability view    • Session controls        • Audio capture (future)  │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ WebSocket / REST
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                               API GATEWAY                                     │
│                              (FastAPI / Python)                               │
├──────────────────────────────────────────────────────────────────────────────┤
│  • REST endpoints          • WebSocket handlers      • Request validation     │
│  • Session management      • Rate limiting           • Authentication (TBD)   │
│  • Health checks           • Metrics export          • Error handling         │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           PROCESSING PIPELINE                                 │
│                              (Python / Async)                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │   Audio     │    │    ASR      │    │   Feature   │    │   Triage    │   │
│  │   Buffer    │───▶│  (Whisper)  │───▶│ Extraction  │───▶│   Model     │   │
│  │             │    │             │    │             │    │             │   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│        │                   │                  │                  │           │
│        │                   │                  │                  │           │
│        ▼                   ▼                  ▼                  ▼           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │  Prosody    │    │ Transcript  │    │  Feature    │    │  Explain-   │   │
│  │  Extractor  │    │  Segments   │    │  Vectors    │    │  ability    │   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                       │
│                           (Redis / Optional DB)                               │
├──────────────────────────────────────────────────────────────────────────────┤
│  • Session state (ephemeral)     • Feature cache (optional)                  │
│  • Model registry                • Audit logs (optional)                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Session Lifecycle

```
1. Client creates session         POST /api/sessions → session_id
2. Client connects WebSocket      WS /ws/session/{id}
3. Client streams audio           Binary frames → backend
4. Backend processes chunks       Audio → ASR → Features → Triage
5. Backend streams updates        JSON frames → client
6. Client ends session            DELETE /api/sessions/{id}
7. Backend cleans up              Apply retention policies
```

### Audio Chunk Processing

```
┌─────────────────────────────────────────────────────────────────┐
│                     Audio Chunk (100-500ms)                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
        ┌───────────────┐               ┌───────────────┐
        │ Prosody Path  │               │   ASR Path    │
        ├───────────────┤               ├───────────────┤
        │ • Pitch (F0)  │               │ • Whisper     │
        │ • Energy/RMS  │               │ • Streaming   │
        │ • Jitter      │               │ • VAD         │
        │ • Speech rate │               └───────┬───────┘
        └───────┬───────┘                       │
                │                               ▼
                │                       ┌───────────────┐
                │                       │ Text Features │
                │                       ├───────────────┤
                │                       │ • Embeddings  │
                │                       │ • Keywords    │
                │                       │ • Intent      │
                │                       └───────┬───────┘
                │                               │
                └───────────────┬───────────────┘
                                ▼
                        ┌───────────────┐
                        │ Feature Fusion│
                        │ [prosody|text]│
                        └───────┬───────┘
                                ▼
                        ┌───────────────┐
                        │ Triage Model  │
                        │ + Explanation │
                        └───────┬───────┘
                                ▼
                        ┌───────────────┐
                        │ WebSocket Out │
                        └───────────────┘
```

---

## Processing Pipeline

### Audio Buffer

- **Input**: PCM audio chunks from WebSocket (binary frames)
- **Buffer size**: 500ms–2s configurable window
- **Overlap**: 50% for continuity
- **Output**: Audio segments for parallel processing

### ASR (Whisper)

- **Model**: OpenAI Whisper (base/small/medium) running locally
- **Mode**: Streaming with Voice Activity Detection
- **Output**: Transcript segments with timestamps and confidence

### Prosody Extractor

- **Library**: Librosa + optional openSMILE
- **Features extracted**:
  - Pitch (F0): mean, std, range, contour
  - Energy: RMS, dynamics
  - Temporal: speech rate, pause ratio, pause duration
  - Quality: jitter, shimmer (voice instability indicators)

### Text Feature Extractor

- **Embeddings**: Sentence-BERT or similar transformer encoder
- **Keywords**: Risk-related lexicon matching
- **Intent**: Simple intent classification (question, statement, distress cue)

### Triage Model

- **Architecture**: Multimodal fusion network (see Model Architecture)
- **Heads**: 4 parallel classification/regression heads
- **Explainability**: Integrated Gradients or SHAP on feature vectors

---

## Model Architecture

```
                    Input Layer
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
┌───────────────┐                 ┌───────────────┐
│ Prosody Enc.  │                 │  Text Enc.    │
│ (MLP/LSTM)    │                 │ (Transformer) │
│ [batch, Dp]   │                 │ [batch, Dt]   │
└───────┬───────┘                 └───────┬───────┘
        │                                 │
        └─────────────┬───────────────────┘
                      ▼
              ┌───────────────┐
              │ Fusion Layer  │
              │ (Concat+MLP)  │
              │ [batch, Df]   │
              └───────┬───────┘
                      │
        ┌─────────────┼─────────────┬─────────────┐
        ▼             ▼             ▼             ▼
   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
   │Emotion  │  │ Risk    │  │Urgency  │  │ Action  │
   │ Head    │  │ Head    │  │ Head    │  │ Head    │
   │(4-class)│  │(4-class)│  │(regress)│  │(4-class)│
   └─────────┘  └─────────┘  └─────────┘  └─────────┘
```

### Loss Function

Multi-task weighted loss:

```
L = λ_emo * CE(emotion) + λ_risk * CE(risk) + λ_urg * MSE(urgency) + λ_act * CE(action)
```

---

## WebSocket Protocol

### Client → Server

| Frame Type | Content | Description |
|------------|---------|-------------|
| Binary | PCM audio | Raw audio chunk (16kHz, mono, 16-bit) |
| Text/JSON | `{"type": "control", "action": "pause"}` | Control commands |

### Server → Client

| Message Type | Content | Description |
|--------------|---------|-------------|
| `connected` | Session confirmation | Sent on connection |
| `transcript` | `{text, is_final, timestamps}` | ASR output |
| `triage` | `{scores, explanation}` | Triage assessment |
| `alert` | `{level, message}` | High-risk alert |
| `status` | `{message}` | System status |

---

## Privacy Architecture

### Data Classification

| Data Type | Sensitivity | Default Retention |
|-----------|-------------|-------------------|
| Raw audio | HIGH | Ephemeral (memory only) |
| Transcripts | HIGH | Ephemeral (configurable) |
| Prosody features | MEDIUM | Ephemeral (configurable) |
| Triage outputs | LOW | Session-scoped (configurable) |

### Privacy Controls

```python
# Environment-based privacy configuration
PERSIST_RAW_AUDIO=false      # Never persist raw audio by default
PERSIST_TRANSCRIPTS=false    # Opt-in transcript persistence
PERSIST_FEATURES=false       # Opt-in feature persistence
SESSION_TTL_SECONDS=0        # Delete immediately after session
```

### Isolation Boundaries

```
┌─────────────────────────────────────────────────────────────────┐
│                       TRUST BOUNDARY                            │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐     │
│  │ Raw Audio     │   │ Features      │   │ Outputs       │     │
│  │ (ephemeral)   │──▶│ (derived)     │──▶│ (anonymized)  │     │
│  └───────────────┘   └───────────────┘   └───────────────┘     │
│         │                                                       │
│         ▼                                                       │
│  [Deleted after processing]                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Scalability Considerations

### Horizontal Scaling

- **Stateless API**: Session state in Redis enables multi-worker deployment
- **Model serving**: Can extract to dedicated inference service (e.g., Triton)
- **Load balancing**: WebSocket sticky sessions via session ID hashing

### Vertical Scaling

- **GPU inference**: Whisper + triage model on CUDA
- **Batching**: Accumulate audio chunks for batch ASR inference
- **Model optimization**: ONNX/TensorRT for production

### Bottlenecks

| Component | Bottleneck | Mitigation |
|-----------|------------|------------|
| Whisper ASR | Compute-bound | Use faster-whisper, GPU, or smaller model |
| Feature extraction | I/O-bound | Async parallel processing |
| WebSocket connections | Memory | Connection pooling, Redis pub/sub |

---

## Future Extensions

### Planned

- [ ] Browser-based audio capture (WebRTC)
- [ ] Text chat modality (parallel input stream)
- [ ] Model fine-tuning pipeline
- [ ] A/B testing framework for models

### Potential

- Video modality (facial expressions, gestures)
- Multi-party conversation support
- Integration with ticketing/CRM systems
- Federated learning for privacy-preserving training
