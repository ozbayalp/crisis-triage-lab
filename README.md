# CrisisTriage AI

> **Privacy-first, research-grade real-time triage for mental health hotline conversations.**

---

**IMPORTANT: NOT FOR CLINICAL USE** - This is a research prototype and simulation tool only. It has no clinical validity and must not be used for real crisis intervention, diagnosis, or treatment decisions. See [SAFETY_LIMITATIONS.md](docs/SAFETY_LIMITATIONS.md) for details.

---

## Project Summary for Reviewers

- **Full-Stack Development**: React/Next.js frontend with FastAPI backend, WebSocket streaming
- **Machine Learning Integration**: Custom neural classifier (DistilBERT-based), local Whisper ASR
- **Real-Time Processing**: Live audio capture, silence detection, streaming transcription
- **Privacy-Conscious Architecture**: Ephemeral data handling, no external API dependencies
- **Professional UI/UX**: Vercel Design System, responsive design, dark/light mode
- **Comprehensive Testing**: 73+ backend tests with pytest, integration tests
- **Production Patterns**: CORS handling, error boundaries, graceful degradation

The system runs entirely locally - no cloud services, API keys, or external dependencies required for core functionality.

---

## Tech Stack Overview

| Layer | Technologies |
|-------|--------------|
| **Frontend** | Next.js 14, React 18, TypeScript, TailwindCSS |
| **Backend** | FastAPI, Python 3.11+, Pydantic, WebSockets |
| **ML/AI** | PyTorch, Transformers (DistilBERT), OpenAI Whisper |
| **Audio** | Web Audio API, PCM streaming, silence detection |
| **Testing** | pytest, React Testing Library (planned) |
| **Deployment** | Docker, Docker Compose |

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

## How You Can Run This Project Locally

Welcome! This section is designed to help recruiters, hiring managers, and technical reviewers quickly get the project running on their local machine. The entire system runs locally with no external API keys required.

### Prerequisites

Before you begin, ensure you have:

- **Docker Desktop** (includes Docker Compose) - [Download here](https://www.docker.com/products/docker-desktop/)
- **Git** for cloning the repository
- ~4GB of free disk space (for ML models)

That's it! Docker handles all other dependencies.

### Quick Start (Recommended)

The fastest way to run everything:

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/TriageSoftware.git
cd TriageSoftware

# 2. Start with Docker Compose
docker-compose up --build
```

Wait 2-3 minutes for initial model downloads, then open:
- **Frontend Dashboard**: http://localhost:3000
- **Backend API Docs**: http://localhost:8000/docs

### Manual Setup (Alternative)

If you prefer running without Docker:

```bash
# 1. Clone and enter repo
git clone https://github.com/yourusername/TriageSoftware.git
cd TriageSoftware

# 2. Backend setup
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
python -m uvicorn main:app --host 0.0.0.0 --port 8000

# 3. Frontend setup (new terminal)
cd frontend
npm install
cp .env.local.example .env.local  # If exists
npm run dev
```

### Testing the System

Once running, try these features:

#### 1. Text Message Triage
- Navigate to **Live Triage** page
- Click "Start Session"
- Type a message in the text input, e.g.:
  - *"I've been feeling overwhelmed lately and can't sleep"*
  - *"Everything is fine, just checking in"*
- Click "Send" to see real-time triage results

#### 2. Microphone Audio Triage
- On the **Live Triage** page, click "Start Microphone"
- Speak naturally for 5+ seconds
- Pause to let the system process (uses silence detection)
- View the live transcript and triage assessment

#### 3. Analytics Dashboard
- Navigate to **Analytics** page
- View aggregated statistics from your test sessions
- Use the "Test Scenarios" section to copy pre-made test messages

#### 4. Sessions History
- Navigate to **Sessions** page
- View all triage sessions grouped by session ID
- Explore individual session details

### Demo Test Messages

Try these messages to see different risk assessments:

| Risk Level | Example Message |
|------------|-----------------|
| **Low** | *"I'm doing okay today, just wanted to talk to someone"* |
| **Medium** | *"I've been feeling really anxious and overwhelmed with work"* |
| **High** | *"I don't know if I can keep going like this anymore"* |

### What Runs Locally

Everything runs on your machine:
- **Whisper ASR** - Speech-to-text transcription (no cloud API)
- **Neural Triage Model** - DistilBERT-based classifier (pre-trained weights included)
- **All audio processing** - Captured and processed locally
- **No data leaves your machine** - Privacy by design

### No API Keys Required

The core functionality requires zero external API keys. Optional integrations (like Twilio for phone calls) are disabled by default and only needed for telephony features.

### Note on Telephony Integration

Phone call integration (Twilio) is **disabled by default**. To enable it for development:
1. Set `ENABLE_TELEPHONY_INTEGRATION=true` in `backend/.env`
2. Configure Twilio credentials (requires Twilio account)

This feature is optional and not required to evaluate the core system.

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

## Why This Project Cannot Be Publicly Deployed

This system is intentionally designed for **local execution only** and is not deployed to any public-facing servers. This decision reflects careful consideration of ethical, legal, and safety concerns inherent to AI systems that appear to assess mental health crises.

### Ethical Concerns

- **Vulnerable Population**: People in mental health crises are extremely vulnerable. Presenting an AI system that appears to assess crisis severity could lead users to over-rely on automated outputs instead of seeking human support.
- **Model Limitations**: The neural model, while technically functional, has not been validated on representative clinical populations. Its outputs should not be interpreted as clinically meaningful assessments.
- **Potential for Harm**: Incorrect triage (false negatives especially) could delay critical intervention for someone in genuine crisis.

### Legal Implications

- **Not a Medical Device**: This system is not FDA-approved, CE-marked, or certified by any regulatory body. Deploying it publicly could constitute offering unlicensed medical advice.
- **Liability Exposure**: Operating a public-facing crisis assessment tool without proper licensure, clinical oversight, and liability insurance would expose operators to significant legal risk.
- **Jurisdictional Complexity**: Mental health regulations vary significantly across jurisdictions. A public deployment would need to comply with healthcare laws in every region where users might access it.

### Privacy Risks

- **Sensitive Data**: Users might submit deeply personal content about mental health struggles, suicidal ideation, or trauma. Public deployment would create obligations around data protection (HIPAA, GDPR, etc.) that this research prototype is not designed to meet.
- **No Clinical Data Governance**: The system lacks the audit trails, access controls, and data retention policies required for handling protected health information.

### Why Local Usage Avoids These Risks

When run locally:
- Users are developers/reviewers evaluating engineering skills, not individuals seeking crisis support
- No real crisis data enters the system
- All processing stays on the user's machine
- The "research simulation" context is clear and explicit
- No public-facing endpoint exists that could be mistakenly used for real crisis intervention

### Safety Measures Implemented

This repository includes multiple layers of safety communication:
- Prominent "NOT FOR CLINICAL USE" warnings in README and UI
- Safety disclaimer banner on every page of the dashboard
- [SAFETY_LIMITATIONS.md](docs/SAFETY_LIMITATIONS.md) documenting known limitations
- [MODEL_CARD.md](docs/MODEL_CARD.md) with explicit capability boundaries
- Crisis hotline information displayed in the UI (988 Lifeline reference)

### Intentional Design Decisions

- **No public deployment scripts** - Infrastructure configs are for local Docker only
- **No production environment variables** - No cloud service integrations configured
- **Ephemeral data by default** - Nothing persists to reduce accidental data retention
- **Telephony disabled by default** - Phone integration requires explicit opt-in

This approach aligns with responsible AI development practices: demonstrating technical capability while acknowledging the boundaries between research prototypes and production systems that affect human welfare.

---

## License

Research use only. Not licensed for production, clinical, or commercial use.

---

> **If you are in crisis**: Please contact your local emergency services or a crisis hotline. This software cannot provide support. In the US, call or text **988** for the Suicide & Crisis Lifeline.
