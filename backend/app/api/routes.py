"""
CrisisTriage AI - REST API Routes

Endpoints for session management, triage queries, and system health.
Real-time streaming is handled separately via WebSocket.

Architecture:
    All triage operations flow through the TriagePipeline, accessed via
    dependency injection from app.state. This ensures:
    - Single source of truth for triage logic
    - Consistent privacy policy enforcement
    - Centralized logging and metrics
"""

from fastapi import APIRouter, HTTPException, Request, status, Depends, Query
from typing import Optional, List, Union
from uuid import uuid4
import logging

from app.config import Settings
from app.core.pipeline import TriagePipeline
from app.core.types import TriageResult as DomainTriageResult
from app.core.history_store import TriageHistoryStore

from .schemas import (
    SessionCreateRequest,
    SessionCreateResponse,
    SessionStatus,
    TriageSnapshot,
    TriageRequest,
    HealthResponse,
    FeatureExplanation,
    TriageEventSchema,
    TriageAnalyticsSchema,
    RiskLevelStatsSchema,
    AnalyticsDisabledResponse,
    InputModality as SchemaInputModality,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["api"])


# =============================================================================
# Dependencies
# =============================================================================

def get_pipeline(request: Request) -> TriagePipeline:
    """Dependency to get the triage pipeline from app state."""
    return request.app.state.pipeline


def get_settings(request: Request) -> Settings:
    """Dependency to get settings from app state."""
    return request.app.state.settings


def get_history_store(request: Request) -> Optional[TriageHistoryStore]:
    """Dependency to get the history store from pipeline."""
    pipeline = request.app.state.pipeline
    return getattr(pipeline, '_history_store', None)


# =============================================================================
# Converters (Domain -> API Schema)
# =============================================================================

def domain_to_schema(result: DomainTriageResult) -> TriageSnapshot:
    """
    Convert domain TriageResult to API TriageSnapshot schema.
    
    This conversion layer isolates the API schema from internal domain types,
    allowing them to evolve independently.
    """
    return TriageSnapshot(
        emotional_state=result.emotional_state.value,
        risk_level=result.risk_level.value,
        urgency_score=result.urgency_score,
        recommended_action=result.recommended_action.value,
        explanation=FeatureExplanation(
            text_features=result.explanation.to_dict().get("text_features", []),
            prosody_features=result.explanation.to_dict().get("prosody_features", []),
            summary=result.explanation.summary,
        ),
        confidence=result.confidence,
        timestamp_ms=result.timestamp_ms,
    )


# =============================================================================
# Health & Status
# =============================================================================

@router.get("/health", response_model=HealthResponse)
async def health_check(
    pipeline: TriagePipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_settings),
):
    """
    System health check.
    
    Returns status of all critical components:
    - API server
    - Pipeline status
    - Model availability
    """
    # Check pipeline services
    components = {
        "api": "operational",
        "pipeline": "operational",
        "transcription_model": pipeline._transcription.model_id,
        "prosody_extractor": pipeline._prosody.extractor_id,
        "triage_model": pipeline._model.model_id,
        "redis": "not_configured",  # TODO: Check Redis when implemented
    }
    
    return HealthResponse(
        status="healthy",
        components=components,
    )


# =============================================================================
# Session Management
# =============================================================================

@router.post("/sessions", response_model=SessionCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    request: SessionCreateRequest,
    pipeline: TriagePipeline = Depends(get_pipeline),
):
    """
    Create a new triage session.
    
    A session represents one hotline conversation. After creation,
    connect to the WebSocket endpoint to stream audio and receive
    real-time triage updates.
    
    Privacy note: Session IDs are random UUIDs with no PII.
    """
    session_id = str(uuid4())
    
    logger.info("Session created: %s", session_id[:8] + "...")
    
    # TODO: Initialize session state in Redis
    # Future: pipeline.register_session(session_id, request.metadata)
    
    return SessionCreateResponse(
        session_id=session_id,
        websocket_url=f"/ws/session/{session_id}",
        status="created",
    )


@router.get("/sessions/{session_id}", response_model=SessionStatus)
async def get_session_status(session_id: str):
    """
    Get current status of a session.
    
    Returns the latest triage snapshot and session metadata.
    """
    # TODO: Fetch from session store (Redis)
    
    # Placeholder response
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Session {session_id} not found",
    )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def end_session(
    session_id: str,
    settings: Settings = Depends(get_settings),
):
    """
    End and clean up a session.
    
    Privacy behavior (based on config):
    - Raw audio: Deleted immediately (default)
    - Transcripts: Deleted or retained based on PERSIST_TRANSCRIPTS
    - Features: Deleted or retained based on PERSIST_FEATURES
    - Triage outputs: Deleted or retained based on SESSION_TTL_SECONDS
    """
    logger.info("Session ended: %s", session_id[:8] + "...")
    
    # TODO: Implement session cleanup in Redis
    # TODO: Apply privacy policies based on settings:
    #   if not settings.persist_transcripts:
    #       await session_store.delete_transcripts(session_id)
    #   if not settings.persist_features:
    #       await session_store.delete_features(session_id)
    pass


# =============================================================================
# Triage Operations
# =============================================================================

@router.post("/sessions/{session_id}/triage", response_model=TriageSnapshot)
async def process_triage(
    session_id: str,
    request: TriageRequest,
    pipeline: TriagePipeline = Depends(get_pipeline),
):
    """
    Process a text message through the triage pipeline.
    
    This is the primary REST endpoint for text-based triage. For real-time
    audio streaming, use the WebSocket endpoint instead.
    
    The pipeline will:
    1. Analyze the text for risk indicators
    2. Run the triage model
    3. Return assessment with explanation
    
    Privacy:
    - Text is processed in memory
    - Logging respects anonymize_logs setting
    - Persistence respects store_raw_transcripts setting
    """
    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text message cannot be empty",
        )
    
    # Process through pipeline
    result = await pipeline.process_text_message(
        session_id=session_id,
        message=request.text,
        source="rest",
    )
    
    # Convert domain result to API schema
    return domain_to_schema(result)


@router.get("/sessions/{session_id}/triage/latest", response_model=TriageSnapshot)
async def get_latest_triage(session_id: str):
    """
    Get the latest triage assessment for a session.
    
    For real-time updates, use the WebSocket endpoint instead.
    This endpoint is useful for reconnection scenarios.
    """
    # TODO: Fetch latest triage from session state (Redis)
    # Example:
    #   latest = await session_store.get_latest_triage(session_id)
    #   if latest:
    #       return domain_to_schema(latest)
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"No triage data found for session {session_id}",
    )


@router.get("/sessions/{session_id}/triage/history")
async def get_triage_history(
    session_id: str,
    limit: int = 100,
    offset: int = 0,
    settings: Settings = Depends(get_settings),
):
    """
    Get historical triage assessments for a session.
    
    Returns time-series data for dashboard visualization.
    Only available if persistence is enabled.
    """
    if not settings.persist_features:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Triage history requires persistence to be enabled",
        )
    
    # TODO: Implement history retrieval from persistence layer
    # Example:
    #   history = await session_store.get_triage_history(
    #       session_id, limit=limit, offset=offset
    #   )
    #   return [domain_to_schema(r) for r in history]
    
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Triage history not yet implemented",
    )


# =============================================================================
# Analytics Endpoints
# =============================================================================

@router.get(
    "/analytics/summary",
    response_model=Union[TriageAnalyticsSchema, AnalyticsDisabledResponse],
    tags=["analytics"],
)
async def get_analytics_summary(
    settings: Settings = Depends(get_settings),
    history_store: Optional[TriageHistoryStore] = Depends(get_history_store),
):
    """
    Get aggregated analytics across all triage events.
    
    Returns distribution of risk levels, emotions, modalities, and averages.
    
    Privacy:
    - Only aggregate statistics are returned
    - Text snippets only included if STORE_ANALYTICS_TEXT_SNIPPETS=True
    - All data is from simulated/research sessions only
    
    NOTE: This is for RESEARCH AND SIMULATION ONLY.
    """
    if not settings.enable_analytics:
        return AnalyticsDisabledResponse()
    
    if history_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Analytics store not initialized",
        )
    
    analytics = await history_store.get_aggregate_stats()
    
    # Convert domain types to schema
    risk_level_stats = {}
    for risk_key, stats in analytics.risk_level_stats.items():
        risk_level_stats[risk_key] = RiskLevelStatsSchema(
            count=stats.count,
            percentage=stats.percentage,
            avg_urgency=stats.avg_urgency,
            avg_confidence=stats.avg_confidence,
            example_snippets=stats.example_snippets if settings.store_analytics_text_snippets else [],
        )
    
    return TriageAnalyticsSchema(
        total_events=analytics.total_events,
        events_last_hour=analytics.events_last_hour,
        events_last_24h=analytics.events_last_24h,
        risk_counts=analytics.risk_counts,
        risk_percentages=analytics.risk_percentages,
        emotion_counts=analytics.emotion_counts,
        emotion_percentages=analytics.emotion_percentages,
        modality_counts=analytics.modality_counts,
        avg_urgency_score=analytics.avg_urgency_score,
        avg_confidence=analytics.avg_confidence,
        avg_processing_time_ms=analytics.avg_processing_time_ms,
        risk_level_stats=risk_level_stats,
        unique_sessions=analytics.unique_sessions,
    )


@router.get(
    "/analytics/recent",
    response_model=List[TriageEventSchema],
    tags=["analytics"],
)
async def get_recent_events(
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum events to return"),
    settings: Settings = Depends(get_settings),
    history_store: Optional[TriageHistoryStore] = Depends(get_history_store),
):
    """
    Get recent triage events.
    
    Returns the most recent triage events in reverse chronological order.
    
    Privacy:
    - Session IDs are anonymized (truncated)
    - Text snippets only included if STORE_ANALYTICS_TEXT_SNIPPETS=True
    - All data is from simulated/research sessions only
    
    NOTE: This is for RESEARCH AND SIMULATION ONLY.
    """
    if not settings.enable_analytics:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Analytics is disabled in this deployment",
        )
    
    if history_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Analytics store not initialized",
        )
    
    events = await history_store.get_recent_events(limit=limit)
    
    # Convert domain events to schema, respecting privacy
    return [
        TriageEventSchema(
            timestamp=event.timestamp,
            session_id=event.session_id,
            risk_level=event.risk_level.value,
            emotional_state=event.emotional_state.value,
            urgency_score=event.urgency_score,
            confidence=event.confidence,
            modality=event.modality.value,
            processing_time_ms=event.processing_time_ms,
            text_snippet=event.text_snippet if settings.store_analytics_text_snippets else None,
        )
        for event in events
    ]


@router.delete(
    "/analytics/clear",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["analytics"],
)
async def clear_analytics(
    settings: Settings = Depends(get_settings),
    history_store: Optional[TriageHistoryStore] = Depends(get_history_store),
):
    """
    Clear all analytics data.
    
    Use with caution - this permanently deletes all stored events.
    Useful for resetting between test runs.
    """
    if not settings.enable_analytics:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Analytics is disabled in this deployment",
        )
    
    if history_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Analytics store not initialized",
        )
    
    await history_store.clear()
    logger.info("Analytics data cleared")
