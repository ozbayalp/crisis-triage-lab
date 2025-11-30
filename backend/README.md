# CrisisTriage AI - Backend

FastAPI backend for real-time mental health triage.

> **IMPORTANT SAFETY NOTICE**  
> This is a RESEARCH AND SIMULATION tool only.  
> NOT a medical device. NOT suitable for real crisis intervention.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload --port 8000

# View API docs
open http://localhost:8000/docs
```

## Running Tests

### Prerequisites

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

### Run All Tests

```bash
pytest
```

### Run with Coverage

```bash
pytest --cov=app --cov-report=html
open htmlcov/index.html
```

### Run Specific Test Files

```bash
# Pipeline tests
pytest tests/test_pipeline_dummy.py -v

# History store tests
pytest tests/test_history_store.py -v

# API tests
pytest tests/test_api_endpoints.py -v

# Neural model smoke tests
pytest tests/test_neural_model_smoke.py -v
```

### Skip Slow Tests

Some tests (e.g., neural model integration) are marked as slow:

```bash
# Skip slow tests
pytest -m "not slow"

# Run only slow tests
pytest -m slow
```

### Test Markers

| Marker | Description |
|--------|-------------|
| `slow` | Tests that may take longer (e.g., model loading) |
| `integration` | Tests requiring external dependencies |

## Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── test_pipeline_dummy.py   # Core pipeline tests
├── test_history_store.py    # Analytics store tests
├── test_api_endpoints.py    # REST API tests
└── test_neural_model_smoke.py  # ML model tests
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_ENV` | `development` | Environment mode |
| `APP_LOG_LEVEL` | `INFO` | Logging level |
| `TRANSCRIPTION_BACKEND` | `dummy` | `dummy` or `whisper` |
| `PROSODY_BACKEND` | `dummy` | `dummy` or `librosa` |
| `TRIAGE_MODEL_BACKEND` | `dummy` | `dummy` or `neural` |
| `ENABLE_ANALYTICS` | `true` | Enable analytics API |

See `.env.example` for full list.

## API Endpoints

### Health

- `GET /` - Root health check
- `GET /api/health` - Detailed health status

### Sessions

- `POST /api/sessions` - Create new session
- `GET /api/sessions/{id}` - Get session status
- `DELETE /api/sessions/{id}` - End session

### Triage

- `POST /api/sessions/{id}/triage` - Process text
- `GET /api/sessions/{id}/triage/latest` - Get latest result

### Analytics

- `GET /api/analytics/summary` - Aggregated stats
- `GET /api/analytics/recent` - Recent events
- `DELETE /api/analytics/clear` - Clear data

### WebSocket

- `WS /ws/session/{id}` - Real-time streaming

## Architecture

```
app/
├── api/
│   ├── routes.py      # REST endpoints
│   ├── websocket.py   # WebSocket handler
│   └── schemas.py     # Pydantic models
├── core/
│   ├── pipeline.py    # Triage orchestrator
│   ├── history_store.py  # Analytics storage
│   └── types.py       # Domain types
├── services/
│   ├── transcription.py  # Speech-to-text
│   ├── prosody.py        # Acoustic features
│   └── triage_model.py   # Risk classification
└── config.py          # Settings
```

## Development

### Code Style

```bash
# Format code
black app tests

# Sort imports
isort app tests

# Lint
flake8 app tests
```

### Type Checking

```bash
mypy app
```
