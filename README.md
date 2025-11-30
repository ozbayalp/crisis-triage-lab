# CrisisTriage AI

> **Privacy-first, research-grade real-time triage for mental health hotline conversations.**

---

**IMPORTANT: NOT FOR CLINICAL USE** - This is a research prototype and simulation tool only. It has no clinical validity and must not be used for real crisis intervention, diagnosis, or treatment decisions. See [SAFETY_LIMITATIONS.md](docs/SAFETY_LIMITATIONS.md) for details.

---

## Vision

CrisisTriage AI is an AI-powered platform that performs real-time emotional and risk assessment during simulated mental health hotline calls. It combines:

- **Voice signals**: Prosody analysis (pitch, speech rate, pauses, energy)
- **Text signals**: Transcript analysis (intent, risk language, semantic content)

To produce actionable triage outputs:

| Output | Description |
|--------|-------------|
| **Emotional State** | calm / anxious / distressed / panicked |
| **Risk Level** | low / medium / high / imminent |
| **Urgency Score** | 0-100 interpretable scale |
| **Recommended Action** | keep listening / ask follow-up / escalate to human |

## Privacy-First Design

- **All inference runs locally** - no raw audio or transcripts sent to external APIs
- **Ephemeral by default** - raw data is not persisted unless explicitly configured
- **Separation of concerns** - raw data, features, and model outputs are isolated

## Architecture Overview

```
┌─────────────────┐     WebSocket      ┌─────────────────┐
│  React/Next.js  │◄──────────────────►│  FastAPI        │
│  Dashboard      │                    │  Backend        │
└─────────────────┘                    └────────┬────────┘
                                                │
                                    ┌───────────┴───────────┐
                                    │   Processing Pipeline │
                                    ├───────────────────────┤
                                    │ • Whisper ASR (local) │
                                    │ • Prosody extraction  │
                                    │ • Text features       │
                                    │ • Triage model        │
                                    └───────────────────────┘
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for detailed system design.

## Repository Structure

```
TriageSoftware/
├── backend/          # FastAPI service (API, streaming, inference)
├── ml/               # ML training, experiments, notebooks
├── frontend/         # Next.js dashboard
├── infra/            # Docker, deployment configs
├── docs/             # Architecture, privacy, model cards
└── data/             # Local data (gitignored)
```

## Quickstart

### Prerequisites

- Python 3.11+
- Node.js 20+
- Docker & Docker Compose (optional, for containerized dev)

### Local Development

```bash
# 1. Clone and enter repo
cd TriageSoftware

# 2. Backend setup
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload

# 3. Frontend setup (new terminal)
cd frontend
npm install
npm run dev
```

### Docker Compose

```bash
docker-compose up --build
```

## Running Tests

### Backend Tests

```bash
cd backend
pip install -r requirements-dev.txt
pytest
```

**Test Coverage:**

| Test File | Coverage |
|-----------|----------|
| `test_pipeline_dummy.py` | Core pipeline orchestration |
| `test_history_store.py` | Analytics storage |
| `test_api_endpoints.py` | REST API |
| `test_neural_model_smoke.py` | ML model integration |

**Skip slow tests:**

```bash
pytest -m "not slow"
```

**Run with coverage:**

```bash
pytest --cov=app --cov-report=html
```

### Frontend Tests

Frontend tests are planned for future implementation.

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | System design and component details |
| [Privacy](docs/PRIVACY.md) | Data handling and retention policies |
| [Model Card](docs/MODEL_CARD.md) | Neural triage classifier capabilities and limitations |
| [System Card](docs/SYSTEM_CARD.md) | Full system overview, intended use, and constraints |
| [Safety & Limitations](docs/SAFETY_LIMITATIONS.md) | **Critical** — Safety warnings, risks, and prohibited uses |

## Status

**Active development** - Core pipeline, analytics, ML training, telephony integration, and testing complete.

### Completed

- Real-time text and audio processing pipeline
- Whisper transcription and Librosa prosody extraction
- Neural triage classifier with training/evaluation stack
- Live triage dashboard and analytics dashboard
- Privacy-first configuration and ephemeral defaults
- Backend test suite with pytest (73+ tests)
- Phone call integration infrastructure (Twilio-ready)
- Dark/light mode UI with Vercel Design System

### Future Work

- Frontend tests
- Prosody-aware neural models
- Multilingual support
- Formal evaluation on benchmark datasets
- Twilio production integration

## License

Research use only. Not licensed for production, clinical, or commercial use.

---

> **If you are in crisis**: Please contact your local emergency services or a crisis hotline. This software cannot provide support. In the US, call or text **988** for the Suicide & Crisis Lifeline.
