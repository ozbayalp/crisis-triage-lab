"""
CrisisTriage AI - Services Package

Contains service interfaces and implementations for:
- Transcription (ASR)
- Prosody feature extraction
- Triage model inference

Design Pattern:
    Each service defines a Protocol (interface) and one or more implementations.
    The pipeline is configured with concrete implementations at startup,
    enabling dependency injection and easy testing/swapping of components.
"""

from .transcription import (
    TranscriptionService,
    DummyTranscriptionService,
    WhisperTranscriptionService,
)
from .prosody import (
    ProsodyExtractor,
    DummyProsodyExtractor,
    LibrosaProsodyExtractor,
)
from .triage_model import (
    TriageModel,
    DummyTriageModel,
    NeuralTriageModel,
)

__all__ = [
    # Transcription
    "TranscriptionService",
    "DummyTranscriptionService",
    "WhisperTranscriptionService",
    # Prosody
    "ProsodyExtractor",
    "DummyProsodyExtractor",
    "LibrosaProsodyExtractor",
    # Triage
    "TriageModel",
    "DummyTriageModel",
    "NeuralTriageModel",
]
