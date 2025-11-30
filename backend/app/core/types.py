"""
CrisisTriage AI - Core Domain Types

Internal type definitions for the triage pipeline. These are domain objects
used within the core and service layers, independent of API serialization.

Design Notes:
- These types are the "lingua franca" between pipeline components.
- API layer converts these to/from Pydantic schemas for external communication.
- Using dataclasses for simplicity and immutability where appropriate.
- Enums match the API schema enums for consistency but are defined here
  to avoid circular imports and maintain domain independence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, NewType, Optional, Any
import time


# =============================================================================
# Type Aliases
# =============================================================================

SessionId = NewType("SessionId", str)
"""Unique identifier for a triage session. Opaque string (typically UUID4)."""

AudioChunk = NewType("AudioChunk", bytes)
"""Raw audio bytes. Expected format: PCM 16-bit, 16kHz, mono."""

FeatureVector = NewType("FeatureVector", List[float])
"""Dense numeric feature vector for model input."""


# =============================================================================
# Enums (Domain-level, mirroring API enums)
# =============================================================================

class EmotionalState(str, Enum):
    """Detected emotional state of the speaker."""
    CALM = "calm"
    ANXIOUS = "anxious"
    DISTRESSED = "distressed"
    PANICKED = "panicked"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    """Assessed risk level for the speaker."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    IMMINENT = "imminent"
    UNKNOWN = "unknown"


class RecommendedAction(str, Enum):
    """Recommended action for the operator based on triage assessment."""
    CONTINUE_LISTENING = "continue_listening"
    ASK_FOLLOWUP = "ask_followup"
    ESCALATE_TO_HUMAN = "escalate_to_human"
    IMMEDIATE_INTERVENTION = "immediate_intervention"


# =============================================================================
# Prosody Features
# =============================================================================

@dataclass(frozen=True)
class ProsodyFeatures:
    """
    Acoustic/prosodic features extracted from audio.
    
    These features capture voice characteristics that correlate with
    emotional state and distress levels, independent of linguistic content.
    
    All values are normalized to reasonable ranges for model consumption.
    None values indicate the feature could not be extracted (e.g., silence).
    
    Attributes:
        speech_rate: Syllables per second (typical range: 2-8)
        pitch_mean: Mean fundamental frequency in Hz (typical: 85-255 Hz)
        pitch_std: Standard deviation of pitch (higher = more variation)
        pitch_range: Max - min pitch in Hz
        energy_mean: Mean RMS energy (normalized 0-1)
        energy_std: Variation in energy
        pause_ratio: Fraction of audio that is silence (0-1)
        pause_count: Number of detected pauses
        jitter: Pitch perturbation (voice instability indicator)
        shimmer: Amplitude perturbation (voice instability indicator)
        voice_quality_score: Aggregate voice quality metric (0-1)
        duration_seconds: Duration of the audio segment
    """
    speech_rate: Optional[float] = None
    pitch_mean: Optional[float] = None
    pitch_std: Optional[float] = None
    pitch_range: Optional[float] = None
    energy_mean: Optional[float] = None
    energy_std: Optional[float] = None
    pause_ratio: Optional[float] = None
    pause_count: Optional[int] = None
    jitter: Optional[float] = None
    shimmer: Optional[float] = None
    voice_quality_score: Optional[float] = None
    duration_seconds: Optional[float] = None

    def to_vector(self) -> FeatureVector:
        """
        Convert to dense feature vector for model input.
        
        Missing values are replaced with sentinel (-1.0) that models
        should be trained to handle, or use feature-specific defaults.
        """
        sentinel = -1.0
        return FeatureVector([
            self.speech_rate if self.speech_rate is not None else sentinel,
            self.pitch_mean if self.pitch_mean is not None else sentinel,
            self.pitch_std if self.pitch_std is not None else sentinel,
            self.pitch_range if self.pitch_range is not None else sentinel,
            self.energy_mean if self.energy_mean is not None else sentinel,
            self.energy_std if self.energy_std is not None else sentinel,
            self.pause_ratio if self.pause_ratio is not None else sentinel,
            float(self.pause_count) if self.pause_count is not None else sentinel,
            self.jitter if self.jitter is not None else sentinel,
            self.shimmer if self.shimmer is not None else sentinel,
            self.voice_quality_score if self.voice_quality_score is not None else sentinel,
            self.duration_seconds if self.duration_seconds is not None else sentinel,
        ])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "speech_rate": self.speech_rate,
            "pitch_mean": self.pitch_mean,
            "pitch_std": self.pitch_std,
            "pitch_range": self.pitch_range,
            "energy_mean": self.energy_mean,
            "energy_std": self.energy_std,
            "pause_ratio": self.pause_ratio,
            "pause_count": self.pause_count,
            "jitter": self.jitter,
            "shimmer": self.shimmer,
            "voice_quality_score": self.voice_quality_score,
            "duration_seconds": self.duration_seconds,
        }


# =============================================================================
# Feature Explanation (for interpretability)
# =============================================================================

@dataclass(frozen=True)
class FeatureContribution:
    """A single feature's contribution to a prediction."""
    feature_name: str
    contribution: float  # Signed contribution (+ increases risk, - decreases)
    value: Optional[Any] = None  # The actual feature value
    category: str = "unknown"  # "text" | "prosody" | "combined"


@dataclass
class TriageExplanation:
    """
    Explanation of which features drove a triage decision.
    
    Supports interpretability requirements by providing feature-level
    attribution for model predictions.
    """
    top_contributors: List[FeatureContribution] = field(default_factory=list)
    summary: str = ""
    method: str = "none"  # "shap" | "integrated_gradients" | "attention" | "none"
    confidence_interval: Optional[tuple[float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "text_features": [
                {"feature": c.feature_name, "contribution": c.contribution, "value": c.value}
                for c in self.top_contributors if c.category == "text"
            ],
            "prosody_features": [
                {"feature": c.feature_name, "contribution": c.contribution, "value": c.value}
                for c in self.top_contributors if c.category == "prosody"
            ],
            "summary": self.summary,
        }


# =============================================================================
# Triage Result (Core Domain Object)
# =============================================================================

@dataclass
class TriageResult:
    """
    Complete triage assessment result.
    
    This is the primary output of the triage pipeline, containing all
    classification outputs, confidence scores, and explanations.
    
    Attributes:
        session_id: The session this result belongs to
        emotional_state: Detected emotional state
        risk_level: Assessed risk level
        urgency_score: Continuous urgency score (0-100)
        recommended_action: Suggested next action for operator
        confidence: Model confidence in the assessment (0-1)
        explanation: Feature attribution for interpretability
        timestamp_ms: Milliseconds since session start
        transcript_segment: The text that was analyzed (if applicable)
        prosody_features: Prosody features used (if applicable)
        processing_time_ms: Pipeline latency for this result
        model_version: Version identifier of the triage model
    """
    session_id: SessionId
    emotional_state: EmotionalState
    risk_level: RiskLevel
    urgency_score: int  # 0-100
    recommended_action: RecommendedAction
    confidence: float  # 0-1
    explanation: TriageExplanation
    timestamp_ms: int
    transcript_segment: Optional[str] = None
    prosody_features: Optional[ProsodyFeatures] = None
    processing_time_ms: Optional[float] = None
    model_version: str = "dummy-v0.0.1"

    def __post_init__(self):
        """Validate constraints."""
        if not 0 <= self.urgency_score <= 100:
            raise ValueError(f"urgency_score must be 0-100, got {self.urgency_score}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be 0-1, got {self.confidence}")

    @classmethod
    def create_unknown(cls, session_id: SessionId, reason: str = "Processing failed") -> "TriageResult":
        """Factory for creating an 'unknown' result when processing fails."""
        return cls(
            session_id=session_id,
            emotional_state=EmotionalState.UNKNOWN,
            risk_level=RiskLevel.UNKNOWN,
            urgency_score=0,
            recommended_action=RecommendedAction.CONTINUE_LISTENING,
            confidence=0.0,
            explanation=TriageExplanation(summary=reason),
            timestamp_ms=int(time.time() * 1000),
        )


# =============================================================================
# Transcription Result
# =============================================================================

@dataclass
class TranscriptionResult:
    """
    Result from the transcription service.
    
    Supports both streaming (partial) and finalized transcriptions.
    """
    text: str
    is_final: bool = True
    confidence: float = 1.0
    start_time_ms: Optional[int] = None
    end_time_ms: Optional[int] = None
    language: str = "en"
    
    # For word-level timestamps (optional, model-dependent)
    word_timestamps: Optional[List[Dict[str, Any]]] = None


# =============================================================================
# Pipeline Context (for passing state through pipeline stages)
# =============================================================================

@dataclass
class PipelineContext:
    """
    Context object passed through pipeline stages.
    
    Carries session information, timing, and accumulated results
    as data flows through the pipeline.
    """
    session_id: SessionId
    request_id: str  # Unique ID for this processing request
    start_time: datetime = field(default_factory=datetime.utcnow)
    
    # Accumulated data
    raw_audio: Optional[AudioChunk] = None
    transcription: Optional[TranscriptionResult] = None
    prosody: Optional[ProsodyFeatures] = None
    
    # Metadata
    source: str = "unknown"  # "websocket" | "rest" | "batch"
    client_timestamp_ms: Optional[int] = None
    
    def elapsed_ms(self) -> float:
        """Calculate elapsed time since context creation."""
        delta = datetime.utcnow() - self.start_time
        return delta.total_seconds() * 1000


# =============================================================================
# Input Modality
# =============================================================================

class InputModality(str, Enum):
    """Source modality of the triage input."""
    TEXT = "text"
    AUDIO = "audio"
    PHONE_CALL = "phone_call"
    MIXED = "mixed"


# =============================================================================
# Analytics Types
# =============================================================================

@dataclass
class TriageEvent:
    """
    A single triage event for analytics.
    
    This is a privacy-aware representation of a triage result,
    storing only the metadata needed for aggregate analytics.
    
    IMPORTANT: Raw text/transcript is only stored if 
    STORE_ANALYTICS_TEXT_SNIPPETS=True in settings.
    """
    timestamp: datetime
    session_id: str  # Truncated/anonymized session ID
    risk_level: RiskLevel
    emotional_state: EmotionalState
    urgency_score: int
    confidence: float
    modality: InputModality
    processing_time_ms: Optional[float] = None
    
    # Only populated if STORE_ANALYTICS_TEXT_SNIPPETS=True
    text_snippet: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "risk_level": self.risk_level.value,
            "emotional_state": self.emotional_state.value,
            "urgency_score": self.urgency_score,
            "confidence": self.confidence,
            "modality": self.modality.value,
            "processing_time_ms": self.processing_time_ms,
            "text_snippet": self.text_snippet,
        }
    
    @classmethod
    def from_triage_result(
        cls,
        result: "TriageResult",
        modality: InputModality,
        store_text: bool = False,
    ) -> "TriageEvent":
        """Create a TriageEvent from a TriageResult."""
        # Anonymize session ID (keep first 8 chars only)
        anon_session = result.session_id[:8] if len(result.session_id) > 8 else result.session_id
        
        # Truncate text snippet if allowed
        text_snippet = None
        if store_text and result.transcript_segment:
            text_snippet = result.transcript_segment[:100]
            if len(result.transcript_segment) > 100:
                text_snippet += "..."
        
        return cls(
            timestamp=datetime.utcnow(),
            session_id=anon_session,
            risk_level=result.risk_level,
            emotional_state=result.emotional_state,
            urgency_score=result.urgency_score,
            confidence=result.confidence,
            modality=modality,
            processing_time_ms=result.processing_time_ms,
            text_snippet=text_snippet,
        )


@dataclass
class RiskLevelStats:
    """Statistics for a single risk level."""
    count: int = 0
    percentage: float = 0.0
    avg_urgency: float = 0.0
    avg_confidence: float = 0.0
    example_snippets: List[str] = field(default_factory=list)


@dataclass
class TriageAnalytics:
    """
    Aggregated analytics across triage events.
    
    Provides summary statistics for dashboards and evaluation.
    """
    # Totals
    total_events: int = 0
    events_last_hour: int = 0
    events_last_24h: int = 0
    
    # Risk level distribution
    risk_counts: Dict[str, int] = field(default_factory=dict)
    risk_percentages: Dict[str, float] = field(default_factory=dict)
    
    # Emotional state distribution
    emotion_counts: Dict[str, int] = field(default_factory=dict)
    emotion_percentages: Dict[str, float] = field(default_factory=dict)
    
    # Modality distribution
    modality_counts: Dict[str, int] = field(default_factory=dict)
    
    # Averages
    avg_urgency_score: float = 0.0
    avg_confidence: float = 0.0
    avg_processing_time_ms: float = 0.0
    
    # Per-risk stats (including example snippets if privacy allows)
    risk_level_stats: Dict[str, RiskLevelStats] = field(default_factory=dict)
    
    # Unique sessions observed
    unique_sessions: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "total_events": self.total_events,
            "events_last_hour": self.events_last_hour,
            "events_last_24h": self.events_last_24h,
            "risk_counts": self.risk_counts,
            "risk_percentages": self.risk_percentages,
            "emotion_counts": self.emotion_counts,
            "emotion_percentages": self.emotion_percentages,
            "modality_counts": self.modality_counts,
            "avg_urgency_score": self.avg_urgency_score,
            "avg_confidence": self.avg_confidence,
            "avg_processing_time_ms": self.avg_processing_time_ms,
            "risk_level_stats": {
                k: {
                    "count": v.count,
                    "percentage": v.percentage,
                    "avg_urgency": v.avg_urgency,
                    "avg_confidence": v.avg_confidence,
                    "example_snippets": v.example_snippets,
                }
                for k, v in self.risk_level_stats.items()
            },
            "unique_sessions": self.unique_sessions,
        }
