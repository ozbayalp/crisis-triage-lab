# System Card: CrisisTriage AI

> **Document Version:** 1.0  
> **Last Updated:** November 2024  
> **Status:** Research Prototype

---

## System Overview

**CrisisTriage AI** is a research prototype demonstrating real-time, privacy-first triage simulation over text and audio inputs. It combines speech processing, acoustic analysis, and neural text classification into an end-to-end pipeline designed for research and educational purposes.

### System Purpose

| Purpose | Description |
|---------|-------------|
| **Primary** | Demonstrate privacy-preserving ML pipeline architecture |
| **Secondary** | Explore real-time multimodal (text + audio) processing |
| **Tertiary** | Provide a testbed for triage-like classification research |

### System Non-Purpose

This system is **explicitly not designed for**:

- Real crisis intervention or support
- Clinical use in any healthcare setting
- Production deployment to end users
- Any context involving individuals in actual distress

---

## Key Components

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CrisisTriage AI System                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐                          ┌─────────────────┐          │
│  │    Frontend     │◄────── WebSocket ───────►│     Backend     │          │
│  │   (Next.js)     │◄──────── REST ──────────►│    (FastAPI)    │          │
│  └─────────────────┘                          └────────┬────────┘          │
│         │                                              │                    │
│         │                                              ▼                    │
│  ┌──────┴──────┐                          ┌────────────────────────┐       │
│  │ Live Triage │                          │    TriagePipeline      │       │
│  │  Dashboard  │                          │    (Orchestrator)      │       │
│  └─────────────┘                          └────────────┬───────────┘       │
│         │                                              │                    │
│  ┌──────┴──────┐                    ┌──────────────────┼──────────────┐    │
│  │  Analytics  │                    │                  │              │    │
│  │  Dashboard  │                    ▼                  ▼              ▼    │
│  └─────────────┘              ┌──────────┐      ┌──────────┐   ┌──────────┐│
│                               │Transcribe│      │ Prosody  │   │  Triage  ││
│                               │ Service  │      │ Service  │   │  Model   ││
│                               └──────────┘      └──────────┘   └──────────┘│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Summary

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | Next.js, React, TailwindCSS | User interface and visualization |
| **Backend** | FastAPI, Python 3.11+ | API, WebSocket, orchestration |
| **Transcription** | OpenAI Whisper (local) | Speech-to-text conversion |
| **Prosody** | Librosa | Acoustic feature extraction |
| **Triage Model** | PyTorch, Transformers | Risk classification |
| **Analytics** | In-memory store | Metrics aggregation |

---

## Data Flow

### Audio Processing Path

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Browser    │    │  WebSocket  │    │   Whisper   │    │   Librosa   │
│  Microphone │───►│   Server    │───►│   (Local)   │───►│   (Local)   │
└─────────────┘    └─────────────┘    └──────┬──────┘    └──────┬──────┘
                                             │                   │
      PCM 16-bit, 16kHz                 Transcript          Prosody
                                             │              Features
                                             ▼                   │
                                      ┌─────────────┐            │
                                      │   Triage    │◄───────────┘
                                      │    Model    │
                                      └──────┬──────┘
                                             │
                                             ▼
                                      ┌─────────────┐
                                      │   Triage    │
                                      │   Result    │
                                      └─────────────┘
```

### Text Processing Path

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    User     │    │  REST API   │    │   Triage    │    │   Triage    │
│    Input    │───►│  Endpoint   │───►│    Model    │───►│   Result    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### Data Persistence

| Data Type | Default Behavior | Configurable |
|-----------|------------------|--------------|
| Raw audio | **Not persisted** | Yes |
| Transcripts | **Not persisted** | Yes |
| Prosody features | **Not persisted** | Yes |
| Triage results | Session-scoped only | Yes |
| Analytics events | In-memory, bounded | Yes |
| Text snippets | **Disabled by default** | Yes |

---

## Privacy & Security

### Privacy-First Design Principles

1. **Local Inference**: All ML processing runs locally; no external API calls
2. **Ephemeral by Default**: Raw data is not persisted unless explicitly configured
3. **Anonymized Logging**: Sensitive content is redacted from logs by default
4. **Bounded Analytics**: Event history is capped and automatically rotated
5. **No PII Collection**: System does not collect personally identifiable information

### Configuration Flags

| Flag | Default | Effect |
|------|---------|--------|
| `ANONYMIZE_LOGS` | `true` | Redact session IDs and content from logs |
| `STORE_RAW_TRANSCRIPTS` | `false` | Persist transcripts to disk |
| `STORE_AUDIO` | `false` | Persist raw audio files |
| `ENABLE_ANALYTICS` | `true` | Enable metrics collection |
| `STORE_ANALYTICS_TEXT_SNIPPETS` | `false` | Store text in analytics events |

### Security Considerations

| Aspect | Implementation |
|--------|----------------|
| **Network** | Local-only by default; CORS configured for localhost |
| **Authentication** | None (research prototype) |
| **Data at Rest** | Ephemeral by default; no encryption (none needed) |
| **Data in Transit** | WebSocket/HTTP (HTTPS recommended for any deployment) |

---

## Deployment & Configuration

### Intended Environments

| Environment | Suitability |
|-------------|-------------|
| Local development machine | ✅ Primary use case |
| Controlled lab environment | ✅ Appropriate |
| Internal research server | ⚠️ With access controls |
| Public internet | ❌ Not recommended |
| Production healthcare | ❌ Prohibited |

### Configuration

The system is configured via environment variables:

```bash
# Service backends
TRANSCRIPTION_BACKEND=whisper    # dummy | whisper
PROSODY_BACKEND=librosa          # dummy | librosa
TRIAGE_MODEL_BACKEND=dummy       # dummy | neural

# Whisper configuration
WHISPER_MODEL_NAME=base          # tiny | base | small | medium | large

# Privacy controls
ANONYMIZE_LOGS=true
STORE_RAW_TRANSCRIPTS=false
STORE_AUDIO=false

# Analytics
ENABLE_ANALYTICS=true
STORE_ANALYTICS_TEXT_SNIPPETS=false
ANALYTICS_MAX_EVENTS=10000
```

### Deployment Options

```bash
# Local development
cd backend && uvicorn main:app --reload
cd frontend && npm run dev

# Docker Compose
docker-compose up --build
```

---

## Intended Use / Non-Use

### Intended Uses

| Use | Description |
|-----|-------------|
| **Research** | Studying triage behavior, pipeline architecture, and ML integration |
| **Education** | Learning about privacy-first ML system design |
| **Demonstration** | Showcasing real-time multimodal processing capabilities |
| **Benchmarking** | Comparing model architectures on synthetic tasks |
| **Development** | Testing and extending the system for research purposes |

### Prohibited Uses

| Use | Reason |
|-----|--------|
| **Real crisis support** | Not clinically validated; could cause harm |
| **Healthcare deployment** | No regulatory clearance; not a medical device |
| **User-facing service** | No safety guarantees; inappropriate for real users |
| **Clinical research** | Would require IRB approval and proper protocols |
| **Commercial use** | Research-only license; not production-ready |

---

## System Capabilities & Limitations

### Capabilities

| Capability | Description |
|------------|-------------|
| Real-time audio processing | Sub-second transcription and analysis |
| Multimodal analysis | Combined text and acoustic features |
| Privacy-preserving | All processing runs locally |
| Configurable | Service backends and privacy settings |
| Observable | Analytics and metrics endpoints |
| Testable | Comprehensive test suite |

### Limitations

| Limitation | Impact |
|------------|--------|
| **Synthetic training data** | Unknown real-world performance |
| **English-only** | No multilingual support |
| **Single-speaker** | Not designed for multi-party conversations |
| **No conversation context** | Each input processed independently |
| **Research prototype** | Not hardened for production use |
| **No clinical validation** | Cannot be used for real assessments |

---

## Performance Characteristics

### Latency (Typical)

| Component | Latency |
|-----------|---------|
| Transcription (Whisper base) | 200–500ms per chunk |
| Prosody extraction | 50–100ms |
| Triage inference | 30–80ms |
| **End-to-end (audio)** | 300–700ms |
| **End-to-end (text)** | 50–100ms |

### Resource Requirements

| Resource | Requirement |
|----------|-------------|
| CPU | Modern multi-core (4+ cores recommended) |
| RAM | 4GB minimum, 8GB recommended |
| GPU | Optional (speeds up Whisper) |
| Disk | 2GB for models and dependencies |

---

## Future Directions

The following are potential research directions, not planned features:

| Direction | Description |
|-----------|-------------|
| **Prosody-aware models** | Integrate acoustic features into neural classifier |
| **Multilingual support** | Extend to non-English languages |
| **Conversation context** | Maintain state across multiple turns |
| **Formal evaluation** | Benchmark on standard datasets |
| **Federated learning** | Privacy-preserving distributed training |
| **Explainability** | Enhanced feature attribution and explanations |

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Technical architecture details |
| [PRIVACY.md](PRIVACY.md) | Privacy and data handling policies |
| [MODEL_CARD.md](MODEL_CARD.md) | Neural triage classifier details |
| [SAFETY_LIMITATIONS.md](SAFETY_LIMITATIONS.md) | Safety and misuse documentation |

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
| 1.0 | November 2024 | Initial system card |

---

> **Reminder**: This system is a research prototype. It must not be used for real crisis intervention, clinical decision-making, or any context involving individuals in actual distress. If you or someone you know is experiencing a mental health crisis, please contact your local emergency services or a crisis hotline.
