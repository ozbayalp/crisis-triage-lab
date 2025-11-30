"""
CrisisTriage AI - Triage History Store

Stores triage events for analytics and evaluation.

Privacy Notes:
    - Events are stored with anonymized session IDs
    - Text snippets are only stored if STORE_ANALYTICS_TEXT_SNIPPETS=True
    - In-memory store is bounded to prevent memory issues
    - All data is ephemeral (lost on restart unless using Redis backend)

IMPORTANT SAFETY NOTICE:
    This is for RESEARCH AND SIMULATION ONLY.
    Not a medical device. Not suitable for real crisis intervention.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections import defaultdict
from datetime import datetime, timedelta
from threading import Lock
from typing import Dict, List, Optional, Protocol, runtime_checkable

from app.config import Settings
from app.core.types import (
    EmotionalState,
    InputModality,
    RiskLevel,
    RiskLevelStats,
    TriageAnalytics,
    TriageEvent,
    TriageResult,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Protocol
# =============================================================================

@runtime_checkable
class TriageHistoryStore(Protocol):
    """
    Protocol for triage history storage.
    
    Implementations must be thread-safe and handle bounded storage.
    """
    
    @abstractmethod
    async def record_event(self, event: TriageEvent) -> None:
        """Record a triage event."""
        ...
    
    @abstractmethod
    async def get_recent_events(self, limit: int = 100) -> List[TriageEvent]:
        """Get the most recent events."""
        ...
    
    @abstractmethod
    async def get_aggregate_stats(self) -> TriageAnalytics:
        """Get aggregated analytics across all stored events."""
        ...
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all stored events."""
        ...


# =============================================================================
# In-Memory Implementation
# =============================================================================

class InMemoryTriageHistoryStore:
    """
    In-memory implementation of TriageHistoryStore.
    
    Uses a bounded deque-like structure to store events. Thread-safe.
    
    Features:
        - Bounded storage (configurable max events)
        - Fast aggregate computation
        - Example snippets per risk level (if privacy allows)
    """
    
    def __init__(
        self,
        max_events: int = 10000,
        store_text_snippets: bool = False,
        max_snippets_per_risk: int = 3,
    ):
        """
        Initialize the in-memory store.
        
        Args:
            max_events: Maximum number of events to store
            store_text_snippets: Whether to store text snippets
            max_snippets_per_risk: Max example snippets per risk level
        """
        self._max_events = max_events
        self._store_text = store_text_snippets
        self._max_snippets_per_risk = max_snippets_per_risk
        
        # Thread-safe storage
        self._lock = Lock()
        self._events: List[TriageEvent] = []
        
        # Indices for fast lookup
        self._session_ids: set = set()
        self._snippets_by_risk: Dict[RiskLevel, List[str]] = defaultdict(list)
        
        logger.info(
            "InMemoryTriageHistoryStore initialized: max_events=%d, store_text=%s",
            max_events, store_text_snippets
        )
    
    async def record_event(self, event: TriageEvent) -> None:
        """Record a triage event."""
        with self._lock:
            # Add event
            self._events.append(event)
            self._session_ids.add(event.session_id)
            
            # Store example snippet if allowed
            if self._store_text and event.text_snippet:
                risk_snippets = self._snippets_by_risk[event.risk_level]
                if len(risk_snippets) < self._max_snippets_per_risk:
                    risk_snippets.append(event.text_snippet)
            
            # Enforce bounds
            if len(self._events) > self._max_events:
                # Remove oldest events
                excess = len(self._events) - self._max_events
                self._events = self._events[excess:]
                
                logger.debug(
                    "Trimmed %d old events from history store", excess
                )
    
    async def get_recent_events(self, limit: int = 100) -> List[TriageEvent]:
        """Get the most recent events."""
        with self._lock:
            # Return newest first
            return list(reversed(self._events[-limit:]))
    
    async def get_aggregate_stats(self) -> TriageAnalytics:
        """Get aggregated analytics across all stored events."""
        with self._lock:
            if not self._events:
                return TriageAnalytics()
            
            now = datetime.utcnow()
            one_hour_ago = now - timedelta(hours=1)
            one_day_ago = now - timedelta(hours=24)
            
            # Initialize counters
            risk_counts: Dict[str, int] = defaultdict(int)
            emotion_counts: Dict[str, int] = defaultdict(int)
            modality_counts: Dict[str, int] = defaultdict(int)
            
            # Per-risk aggregates
            risk_urgency_sums: Dict[str, float] = defaultdict(float)
            risk_confidence_sums: Dict[str, float] = defaultdict(float)
            
            # Global aggregates
            total_urgency = 0.0
            total_confidence = 0.0
            total_processing_time = 0.0
            processing_time_count = 0
            
            events_last_hour = 0
            events_last_24h = 0
            
            for event in self._events:
                # Risk level
                risk_key = event.risk_level.value
                risk_counts[risk_key] += 1
                risk_urgency_sums[risk_key] += event.urgency_score
                risk_confidence_sums[risk_key] += event.confidence
                
                # Emotion
                emotion_counts[event.emotional_state.value] += 1
                
                # Modality
                modality_counts[event.modality.value] += 1
                
                # Global
                total_urgency += event.urgency_score
                total_confidence += event.confidence
                if event.processing_time_ms:
                    total_processing_time += event.processing_time_ms
                    processing_time_count += 1
                
                # Time windows
                if event.timestamp >= one_hour_ago:
                    events_last_hour += 1
                if event.timestamp >= one_day_ago:
                    events_last_24h += 1
            
            total_events = len(self._events)
            
            # Calculate percentages
            risk_percentages = {
                k: (v / total_events) * 100 for k, v in risk_counts.items()
            }
            emotion_percentages = {
                k: (v / total_events) * 100 for k, v in emotion_counts.items()
            }
            
            # Calculate per-risk stats
            risk_level_stats = {}
            for risk_key, count in risk_counts.items():
                avg_urgency = risk_urgency_sums[risk_key] / count if count > 0 else 0
                avg_confidence = risk_confidence_sums[risk_key] / count if count > 0 else 0
                
                # Get example snippets
                risk_level = RiskLevel(risk_key)
                snippets = self._snippets_by_risk.get(risk_level, [])[:self._max_snippets_per_risk]
                
                risk_level_stats[risk_key] = RiskLevelStats(
                    count=count,
                    percentage=(count / total_events) * 100,
                    avg_urgency=avg_urgency,
                    avg_confidence=avg_confidence,
                    example_snippets=snippets,
                )
            
            return TriageAnalytics(
                total_events=total_events,
                events_last_hour=events_last_hour,
                events_last_24h=events_last_24h,
                risk_counts=dict(risk_counts),
                risk_percentages=risk_percentages,
                emotion_counts=dict(emotion_counts),
                emotion_percentages=emotion_percentages,
                modality_counts=dict(modality_counts),
                avg_urgency_score=total_urgency / total_events if total_events > 0 else 0,
                avg_confidence=total_confidence / total_events if total_events > 0 else 0,
                avg_processing_time_ms=(
                    total_processing_time / processing_time_count
                    if processing_time_count > 0 else 0
                ),
                risk_level_stats=risk_level_stats,
                unique_sessions=len(self._session_ids),
            )
    
    async def clear(self) -> None:
        """Clear all stored events."""
        with self._lock:
            self._events.clear()
            self._session_ids.clear()
            self._snippets_by_risk.clear()
            logger.info("Triage history store cleared")


# =============================================================================
# Factory Function
# =============================================================================

def create_history_store(settings: Settings) -> TriageHistoryStore:
    """
    Create a triage history store based on settings.
    
    Currently only supports in-memory storage.
    Future: Add Redis-backed implementation.
    
    Args:
        settings: Application settings
        
    Returns:
        Configured TriageHistoryStore instance
    """
    if not settings.enable_analytics:
        logger.info("Analytics disabled, using no-op history store")
        return NoOpTriageHistoryStore()
    
    logger.info(
        "Creating InMemoryTriageHistoryStore: max_events=%d, store_text=%s",
        settings.analytics_max_events,
        settings.store_analytics_text_snippets,
    )
    
    return InMemoryTriageHistoryStore(
        max_events=settings.analytics_max_events,
        store_text_snippets=settings.store_analytics_text_snippets,
        max_snippets_per_risk=settings.analytics_example_snippets_per_risk,
    )


# =============================================================================
# No-Op Implementation (when analytics disabled)
# =============================================================================

class NoOpTriageHistoryStore:
    """
    No-op implementation when analytics is disabled.
    """
    
    async def record_event(self, event: TriageEvent) -> None:
        pass
    
    async def get_recent_events(self, limit: int = 100) -> List[TriageEvent]:
        return []
    
    async def get_aggregate_stats(self) -> TriageAnalytics:
        return TriageAnalytics()
    
    async def clear(self) -> None:
        pass
