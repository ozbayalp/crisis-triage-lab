"""
CrisisTriage AI - Core Package

Contains the central orchestration logic and domain types:
- pipeline: Triage processing orchestrator
- types: Internal domain types and type aliases
- history_store: Analytics event storage
"""

from .types import (
    SessionId,
    TriageResult,
    ProsodyFeatures,
    FeatureVector,
    AudioChunk,
    TriageEvent,
    TriageAnalytics,
    RiskLevelStats,
    InputModality,
)
from .pipeline import TriagePipeline, create_pipeline
from .history_store import (
    TriageHistoryStore,
    InMemoryTriageHistoryStore,
    create_history_store,
)

__all__ = [
    # Pipeline
    "TriagePipeline",
    "create_pipeline",
    # Types
    "SessionId",
    "TriageResult",
    "ProsodyFeatures",
    "FeatureVector",
    "AudioChunk",
    # Analytics
    "TriageEvent",
    "TriageAnalytics",
    "RiskLevelStats",
    "InputModality",
    "TriageHistoryStore",
    "InMemoryTriageHistoryStore",
    "create_history_store",
]
