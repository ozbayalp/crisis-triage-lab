"""
CrisisTriage AI - Test Configuration and Fixtures

Shared fixtures for all test modules.
"""

import os
import sys
from datetime import datetime
from typing import Generator

import pytest
from fastapi.testclient import TestClient

# Ensure backend package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Settings
from app.core.pipeline import TriagePipeline
from app.core.history_store import InMemoryTriageHistoryStore, create_history_store
from app.core.types import (
    RiskLevel,
    EmotionalState,
    InputModality,
    TriageEvent,
)
from app.services.transcription import DummyTranscriptionService
from app.services.prosody import DummyProsodyExtractor
from app.services.triage_model import DummyTriageModel


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests requiring external dependencies")


# =============================================================================
# Settings Fixtures
# =============================================================================

@pytest.fixture
def test_settings() -> Settings:
    """
    Create test settings with safe defaults.
    
    Analytics enabled, but text snippets disabled for privacy.
    """
    return Settings(
        app_env="testing",
        app_debug=True,
        app_log_level="WARNING",  # Reduce noise in tests
        transcription_backend="dummy",
        prosody_backend="dummy",
        triage_model_backend="dummy",
        enable_analytics=True,
        store_analytics_text_snippets=False,
        analytics_max_events=100,
        anonymize_logs=True,
        store_raw_transcripts=False,
        store_audio=False,
    )


@pytest.fixture
def test_settings_analytics_disabled() -> Settings:
    """Settings with analytics disabled."""
    return Settings(
        app_env="testing",
        app_debug=True,
        app_log_level="WARNING",
        enable_analytics=False,
    )


@pytest.fixture
def test_settings_with_snippets() -> Settings:
    """Settings with text snippet storage enabled."""
    return Settings(
        app_env="testing",
        app_debug=True,
        app_log_level="WARNING",
        enable_analytics=True,
        store_analytics_text_snippets=True,
        analytics_max_events=100,
    )


# =============================================================================
# Service Fixtures
# =============================================================================

@pytest.fixture
def dummy_transcription() -> DummyTranscriptionService:
    """Create a dummy transcription service."""
    return DummyTranscriptionService()


@pytest.fixture
def dummy_prosody() -> DummyProsodyExtractor:
    """Create a dummy prosody extractor."""
    return DummyProsodyExtractor()


@pytest.fixture
def dummy_model() -> DummyTriageModel:
    """Create a dummy triage model."""
    return DummyTriageModel()


# =============================================================================
# History Store Fixtures
# =============================================================================

@pytest.fixture
def history_store() -> InMemoryTriageHistoryStore:
    """Create a fresh in-memory history store."""
    return InMemoryTriageHistoryStore(
        max_events=100,
        store_text_snippets=False,
        max_snippets_per_risk=3,
    )


@pytest.fixture
def history_store_with_snippets() -> InMemoryTriageHistoryStore:
    """Create a history store that stores text snippets."""
    return InMemoryTriageHistoryStore(
        max_events=100,
        store_text_snippets=True,
        max_snippets_per_risk=3,
    )


# =============================================================================
# Pipeline Fixtures
# =============================================================================

@pytest.fixture
def pipeline(
    test_settings: Settings,
    dummy_transcription: DummyTranscriptionService,
    dummy_prosody: DummyProsodyExtractor,
    dummy_model: DummyTriageModel,
    history_store: InMemoryTriageHistoryStore,
) -> TriagePipeline:
    """
    Create a test pipeline with dummy services.
    
    This pipeline is fully functional but uses dummy implementations
    that don't require ML models or external dependencies.
    """
    return TriagePipeline(
        transcription=dummy_transcription,
        prosody=dummy_prosody,
        model=dummy_model,
        settings=test_settings,
        history_store=history_store,
    )


@pytest.fixture
def pipeline_no_analytics(
    test_settings_analytics_disabled: Settings,
    dummy_transcription: DummyTranscriptionService,
    dummy_prosody: DummyProsodyExtractor,
    dummy_model: DummyTriageModel,
) -> TriagePipeline:
    """Create a test pipeline with analytics disabled."""
    return TriagePipeline(
        transcription=dummy_transcription,
        prosody=dummy_prosody,
        model=dummy_model,
        settings=test_settings_analytics_disabled,
        history_store=None,
    )


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_triage_events() -> list[TriageEvent]:
    """Create sample triage events for testing."""
    base_time = datetime.utcnow()
    return [
        TriageEvent(
            timestamp=base_time,
            session_id="sess1234",
            risk_level=RiskLevel.LOW,
            emotional_state=EmotionalState.CALM,
            urgency_score=15,
            confidence=0.85,
            modality=InputModality.TEXT,
            processing_time_ms=45.2,
            text_snippet=None,
        ),
        TriageEvent(
            timestamp=base_time,
            session_id="sess5678",
            risk_level=RiskLevel.MEDIUM,
            emotional_state=EmotionalState.ANXIOUS,
            urgency_score=45,
            confidence=0.72,
            modality=InputModality.TEXT,
            processing_time_ms=52.1,
            text_snippet=None,
        ),
        TriageEvent(
            timestamp=base_time,
            session_id="sess9012",
            risk_level=RiskLevel.HIGH,
            emotional_state=EmotionalState.DISTRESSED,
            urgency_score=78,
            confidence=0.91,
            modality=InputModality.AUDIO,
            processing_time_ms=120.5,
            text_snippet=None,
        ),
    ]


@pytest.fixture
def low_risk_messages() -> list[str]:
    """Sample low-risk messages for testing."""
    return [
        "Thank you for listening, I feel better now.",
        "I'm doing okay today, just wanted to check in.",
        "Things are improving, I talked to a friend.",
        "I appreciate your help, I'm managing well.",
    ]


@pytest.fixture
def high_risk_messages() -> list[str]:
    """Sample high-risk messages for testing.
    
    These must match HIGH_RISK_KEYWORDS in DummyTriageModel:
    suicide, kill myself, end my life, want to die, no reason to live,
    better off dead, can't go on, hurt myself, self-harm, overdose
    """
    return [
        "I want to kill myself.",
        "I want to end my life.",
        "I feel like I want to die.",
        "I'm thinking about suicide.",
    ]


@pytest.fixture
def fake_audio_bytes() -> bytes:
    """Generate fake audio bytes for testing."""
    # PCM 16-bit, 16kHz, mono - 0.5 seconds of silence
    # 16000 samples/sec * 0.5 sec * 2 bytes/sample = 16000 bytes
    return b'\x00\x00' * 8000


# =============================================================================
# FastAPI App Fixture
# =============================================================================

@pytest.fixture
def app_with_analytics():
    """Create a FastAPI app instance with analytics enabled."""
    # Import here to avoid circular imports
    from main import create_app
    
    # Override settings for testing
    os.environ["ENABLE_ANALYTICS"] = "true"
    os.environ["APP_ENV"] = "testing"
    os.environ["APP_LOG_LEVEL"] = "WARNING"
    
    app = create_app()
    return app


@pytest.fixture
def client(app_with_analytics) -> Generator[TestClient, None, None]:
    """Create a test client for the FastAPI app."""
    with TestClient(app_with_analytics) as c:
        yield c
