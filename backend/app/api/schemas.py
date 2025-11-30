"""
CrisisTriage AI - API Schemas

Pydantic models for request/response validation.
These define the contract between frontend and backend.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


# ===========================================
# Enums
# ===========================================

class EmotionalState(str, Enum):
    """Detected emotional state of the caller."""
    CALM = "calm"
    ANXIOUS = "anxious"
    DISTRESSED = "distressed"
    PANICKED = "panicked"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    """Assessed risk level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    IMMINENT = "imminent"
    UNKNOWN = "unknown"


class RecommendedAction(str, Enum):
    """Recommended action for the operator."""
    CONTINUE_LISTENING = "continue_listening"
    ASK_FOLLOWUP = "ask_followup"
    ESCALATE_TO_HUMAN = "escalate_to_human"
    IMMEDIATE_INTERVENTION = "immediate_intervention"


# ===========================================
# Session Schemas
# ===========================================

class SessionCreateRequest(BaseModel):
    """Request to create a new triage session."""
    
    # Optional metadata (no PII should be included)
    metadata: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional session metadata (e.g., simulation ID, test run ID)"
    )
    
    # Audio configuration
    sample_rate: int = Field(
        default=16000,
        description="Audio sample rate in Hz"
    )
    channels: int = Field(
        default=1,
        description="Number of audio channels (1 = mono)"
    )


class SessionCreateResponse(BaseModel):
    """Response after creating a session."""
    
    session_id: str = Field(description="Unique session identifier (UUID)")
    websocket_url: str = Field(description="WebSocket URL for streaming")
    status: str = Field(description="Session status")


class SessionStatus(BaseModel):
    """Current status of a session."""
    
    session_id: str
    status: str = Field(description="active | paused | ended")
    created_at: datetime
    duration_seconds: float = Field(description="Session duration so far")
    latest_triage: Optional["TriageSnapshot"] = None


# ===========================================
# Triage Request Schemas
# ===========================================

class TriageRequest(BaseModel):
    """Request to process text through the triage pipeline."""
    
    text: str = Field(
        description="Text message to analyze",
        min_length=1,
        max_length=10000,
    )
    
    # Optional context for multi-turn conversations
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional conversation context (e.g., previous turns)"
    )


# ===========================================
# Triage Response Schemas
# ===========================================

class FeatureExplanation(BaseModel):
    """Explanation of which features drove a triage decision."""
    
    # Top contributing text features
    text_features: List[Dict[str, Any]] = Field(
        default=[],
        description="Text-based feature contributions (e.g., keywords, intent)"
    )
    
    # Top contributing prosody features
    prosody_features: List[Dict[str, Any]] = Field(
        default=[],
        description="Voice-based feature contributions (e.g., pitch variance, speech rate)"
    )
    
    # Human-readable summary
    summary: str = Field(
        default="",
        description="Brief natural language explanation"
    )


class TriageSnapshot(BaseModel):
    """
    A point-in-time triage assessment.
    
    This is the core output of the triage model, representing
    the system's assessment at a given moment.
    """
    
    # Core triage outputs
    emotional_state: EmotionalState = Field(
        description="Detected emotional state"
    )
    risk_level: RiskLevel = Field(
        description="Assessed risk level"
    )
    urgency_score: int = Field(
        ge=0, le=100,
        description="Urgency score from 0 (low) to 100 (critical)"
    )
    recommended_action: RecommendedAction = Field(
        description="Recommended action for the operator"
    )
    
    # Explainability
    explanation: FeatureExplanation = Field(
        default_factory=FeatureExplanation,
        description="Feature attribution for this assessment"
    )
    
    # Confidence
    confidence: float = Field(
        ge=0.0, le=1.0,
        default=0.0,
        description="Model confidence in this assessment"
    )
    
    # Timing
    timestamp_ms: int = Field(
        description="Timestamp in milliseconds since session start"
    )


class TranscriptSegment(BaseModel):
    """A segment of transcribed speech."""
    
    text: str = Field(description="Transcribed text")
    is_final: bool = Field(
        default=False,
        description="Whether this is a finalized transcript (vs. interim)"
    )
    start_ms: int = Field(description="Start time in ms")
    end_ms: int = Field(description="End time in ms")
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


# ===========================================
# Health Schemas
# ===========================================

class HealthResponse(BaseModel):
    """System health status."""
    
    status: str = Field(description="Overall status: healthy | degraded | unhealthy")
    components: Dict[str, str] = Field(
        description="Status of individual components"
    )
    version: str = Field(default="0.1.0")


# ===========================================
# WebSocket Message Schemas
# ===========================================

class WSMessage(BaseModel):
    """Base WebSocket message."""
    type: str
    data: Optional[Dict[str, Any]] = None


class WSTriageUpdate(BaseModel):
    """WebSocket triage update message."""
    type: str = "triage"
    data: TriageSnapshot


class WSTranscriptUpdate(BaseModel):
    """WebSocket transcript update message."""
    type: str = "transcript"
    data: TranscriptSegment


class WSAlert(BaseModel):
    """WebSocket alert message for high-risk situations."""
    type: str = "alert"
    level: str = Field(description="warning | high | critical")
    message: str
    timestamp_ms: int


# ===========================================
# Input Modality
# ===========================================

class InputModality(str, Enum):
    """Source modality of the triage input."""
    TEXT = "text"
    AUDIO = "audio"
    MIXED = "mixed"


# ===========================================
# Analytics Schemas
# ===========================================

class TriageEventSchema(BaseModel):
    """
    A single triage event for analytics.
    
    Privacy: text_snippet is only populated if STORE_ANALYTICS_TEXT_SNIPPETS=True.
    """
    timestamp: datetime
    session_id: str = Field(description="Anonymized session ID (truncated)")
    risk_level: RiskLevel
    emotional_state: EmotionalState
    urgency_score: int = Field(ge=0, le=100)
    confidence: float = Field(ge=0.0, le=1.0)
    modality: InputModality
    processing_time_ms: Optional[float] = None
    text_snippet: Optional[str] = Field(
        default=None,
        description="Truncated text snippet (only if privacy allows)"
    )

    class Config:
        from_attributes = True


class RiskLevelStatsSchema(BaseModel):
    """Statistics for a single risk level."""
    count: int
    percentage: float
    avg_urgency: float
    avg_confidence: float
    example_snippets: List[str] = Field(
        default_factory=list,
        description="Example snippets (only if privacy allows)"
    )


class TriageAnalyticsSchema(BaseModel):
    """
    Aggregated analytics across triage events.
    
    Provides summary statistics for dashboards and evaluation.
    All data is derived from simulated/research data only.
    """
    # Totals
    total_events: int
    events_last_hour: int
    events_last_24h: int
    
    # Risk level distribution
    risk_counts: Dict[str, int]
    risk_percentages: Dict[str, float]
    
    # Emotional state distribution
    emotion_counts: Dict[str, int]
    emotion_percentages: Dict[str, float]
    
    # Modality distribution
    modality_counts: Dict[str, int]
    
    # Averages
    avg_urgency_score: float
    avg_confidence: float
    avg_processing_time_ms: float
    
    # Per-risk stats
    risk_level_stats: Dict[str, RiskLevelStatsSchema]
    
    # Sessions
    unique_sessions: int

    class Config:
        from_attributes = True


class AnalyticsDisabledResponse(BaseModel):
    """Response when analytics is disabled."""
    message: str = "Analytics is disabled in this deployment"
    enabled: bool = False
