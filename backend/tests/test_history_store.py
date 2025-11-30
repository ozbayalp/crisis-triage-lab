"""
CrisisTriage AI - History Store Tests

Tests for the InMemoryTriageHistoryStore and analytics functionality.
These tests verify:
- Event recording and retrieval
- Aggregate statistics computation
- Bounded buffer behavior
- Privacy-aware text snippet handling

Run with: pytest tests/test_history_store.py -v
"""

import pytest
from datetime import datetime, timedelta
from app.core.history_store import InMemoryTriageHistoryStore, create_history_store
from app.core.types import (
    RiskLevel,
    EmotionalState,
    InputModality,
    TriageEvent,
    TriageAnalytics,
)
from app.config import Settings


class TestEventRecording:
    """Tests for recording events to the history store."""

    @pytest.mark.asyncio
    async def test_record_single_event(self, history_store: InMemoryTriageHistoryStore):
        """Should successfully record a single event."""
        event = TriageEvent(
            timestamp=datetime.utcnow(),
            session_id="test1234",
            risk_level=RiskLevel.LOW,
            emotional_state=EmotionalState.CALM,
            urgency_score=20,
            confidence=0.85,
            modality=InputModality.TEXT,
        )
        
        await history_store.record_event(event)
        
        events = await history_store.get_recent_events(limit=10)
        assert len(events) == 1
        assert events[0].session_id == "test1234"
        assert events[0].risk_level == RiskLevel.LOW

    @pytest.mark.asyncio
    async def test_record_multiple_events(
        self, history_store: InMemoryTriageHistoryStore, sample_triage_events: list[TriageEvent]
    ):
        """Should record multiple events."""
        for event in sample_triage_events:
            await history_store.record_event(event)
        
        events = await history_store.get_recent_events(limit=10)
        assert len(events) == len(sample_triage_events)

    @pytest.mark.asyncio
    async def test_record_event_with_all_fields(self, history_store: InMemoryTriageHistoryStore):
        """Should correctly store all event fields."""
        event = TriageEvent(
            timestamp=datetime.utcnow(),
            session_id="full1234",
            risk_level=RiskLevel.HIGH,
            emotional_state=EmotionalState.DISTRESSED,
            urgency_score=78,
            confidence=0.92,
            modality=InputModality.AUDIO,
            processing_time_ms=125.5,
            text_snippet=None,
        )
        
        await history_store.record_event(event)
        
        events = await history_store.get_recent_events(limit=1)
        stored = events[0]
        
        assert stored.session_id == "full1234"
        assert stored.risk_level == RiskLevel.HIGH
        assert stored.emotional_state == EmotionalState.DISTRESSED
        assert stored.urgency_score == 78
        assert stored.confidence == 0.92
        assert stored.modality == InputModality.AUDIO
        assert stored.processing_time_ms == 125.5


class TestEventRetrieval:
    """Tests for retrieving events from the history store."""

    @pytest.mark.asyncio
    async def test_get_recent_events_empty_store(self, history_store: InMemoryTriageHistoryStore):
        """Empty store should return empty list."""
        events = await history_store.get_recent_events(limit=10)
        assert events == []

    @pytest.mark.asyncio
    async def test_get_recent_events_respects_limit(
        self, history_store: InMemoryTriageHistoryStore, sample_triage_events: list[TriageEvent]
    ):
        """Should respect the limit parameter."""
        for event in sample_triage_events:
            await history_store.record_event(event)
        
        events = await history_store.get_recent_events(limit=2)
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_get_recent_events_order(self, history_store: InMemoryTriageHistoryStore):
        """Events should be returned in reverse chronological order (newest first)."""
        now = datetime.utcnow()
        
        # Record events with different timestamps
        for i in range(3):
            event = TriageEvent(
                timestamp=now + timedelta(seconds=i),
                session_id=f"order{i}",
                risk_level=RiskLevel.LOW,
                emotional_state=EmotionalState.CALM,
                urgency_score=10 + i * 10,
                confidence=0.8,
                modality=InputModality.TEXT,
            )
            await history_store.record_event(event)
        
        events = await history_store.get_recent_events(limit=10)
        
        # Newest should be first (highest urgency score in this case)
        assert events[0].session_id == "order2"
        assert events[1].session_id == "order1"
        assert events[2].session_id == "order0"


class TestAggregateStatistics:
    """Tests for aggregate statistics computation."""

    @pytest.mark.asyncio
    async def test_aggregate_stats_empty_store(self, history_store: InMemoryTriageHistoryStore):
        """Empty store should return zeroed analytics."""
        stats = await history_store.get_aggregate_stats()
        
        assert stats.total_events == 0
        assert stats.events_last_hour == 0
        assert stats.unique_sessions == 0

    @pytest.mark.asyncio
    async def test_aggregate_stats_total_events(
        self, history_store: InMemoryTriageHistoryStore, sample_triage_events: list[TriageEvent]
    ):
        """Should correctly count total events."""
        for event in sample_triage_events:
            await history_store.record_event(event)
        
        stats = await history_store.get_aggregate_stats()
        assert stats.total_events == len(sample_triage_events)

    @pytest.mark.asyncio
    async def test_aggregate_stats_risk_counts(self, history_store: InMemoryTriageHistoryStore):
        """Should correctly count events by risk level."""
        events = [
            TriageEvent(
                timestamp=datetime.utcnow(),
                session_id="r1",
                risk_level=RiskLevel.LOW,
                emotional_state=EmotionalState.CALM,
                urgency_score=10,
                confidence=0.8,
                modality=InputModality.TEXT,
            ),
            TriageEvent(
                timestamp=datetime.utcnow(),
                session_id="r2",
                risk_level=RiskLevel.LOW,
                emotional_state=EmotionalState.CALM,
                urgency_score=15,
                confidence=0.8,
                modality=InputModality.TEXT,
            ),
            TriageEvent(
                timestamp=datetime.utcnow(),
                session_id="r3",
                risk_level=RiskLevel.HIGH,
                emotional_state=EmotionalState.DISTRESSED,
                urgency_score=75,
                confidence=0.8,
                modality=InputModality.TEXT,
            ),
        ]
        
        for event in events:
            await history_store.record_event(event)
        
        stats = await history_store.get_aggregate_stats()
        
        assert stats.risk_counts.get("low", 0) == 2
        assert stats.risk_counts.get("high", 0) == 1
        assert stats.risk_counts.get("medium", 0) == 0

    @pytest.mark.asyncio
    async def test_aggregate_stats_emotion_counts(self, history_store: InMemoryTriageHistoryStore):
        """Should correctly count events by emotional state."""
        events = [
            TriageEvent(
                timestamp=datetime.utcnow(),
                session_id="e1",
                risk_level=RiskLevel.LOW,
                emotional_state=EmotionalState.CALM,
                urgency_score=10,
                confidence=0.8,
                modality=InputModality.TEXT,
            ),
            TriageEvent(
                timestamp=datetime.utcnow(),
                session_id="e2",
                risk_level=RiskLevel.MEDIUM,
                emotional_state=EmotionalState.ANXIOUS,
                urgency_score=40,
                confidence=0.8,
                modality=InputModality.TEXT,
            ),
            TriageEvent(
                timestamp=datetime.utcnow(),
                session_id="e3",
                risk_level=RiskLevel.MEDIUM,
                emotional_state=EmotionalState.ANXIOUS,
                urgency_score=45,
                confidence=0.8,
                modality=InputModality.TEXT,
            ),
        ]
        
        for event in events:
            await history_store.record_event(event)
        
        stats = await history_store.get_aggregate_stats()
        
        assert stats.emotion_counts.get("calm", 0) == 1
        assert stats.emotion_counts.get("anxious", 0) == 2

    @pytest.mark.asyncio
    async def test_aggregate_stats_modality_counts(self, history_store: InMemoryTriageHistoryStore):
        """Should correctly count events by modality."""
        events = [
            TriageEvent(
                timestamp=datetime.utcnow(),
                session_id="m1",
                risk_level=RiskLevel.LOW,
                emotional_state=EmotionalState.CALM,
                urgency_score=10,
                confidence=0.8,
                modality=InputModality.TEXT,
            ),
            TriageEvent(
                timestamp=datetime.utcnow(),
                session_id="m2",
                risk_level=RiskLevel.LOW,
                emotional_state=EmotionalState.CALM,
                urgency_score=15,
                confidence=0.8,
                modality=InputModality.AUDIO,
            ),
        ]
        
        for event in events:
            await history_store.record_event(event)
        
        stats = await history_store.get_aggregate_stats()
        
        assert stats.modality_counts.get("text", 0) == 1
        assert stats.modality_counts.get("audio", 0) == 1

    @pytest.mark.asyncio
    async def test_aggregate_stats_averages(self, history_store: InMemoryTriageHistoryStore):
        """Should correctly compute averages."""
        events = [
            TriageEvent(
                timestamp=datetime.utcnow(),
                session_id="a1",
                risk_level=RiskLevel.LOW,
                emotional_state=EmotionalState.CALM,
                urgency_score=20,
                confidence=0.80,
                modality=InputModality.TEXT,
                processing_time_ms=100.0,
            ),
            TriageEvent(
                timestamp=datetime.utcnow(),
                session_id="a2",
                risk_level=RiskLevel.HIGH,
                emotional_state=EmotionalState.DISTRESSED,
                urgency_score=80,
                confidence=0.90,
                modality=InputModality.TEXT,
                processing_time_ms=200.0,
            ),
        ]
        
        for event in events:
            await history_store.record_event(event)
        
        stats = await history_store.get_aggregate_stats()
        
        assert stats.avg_urgency_score == pytest.approx(50.0)  # (20 + 80) / 2
        assert stats.avg_confidence == pytest.approx(0.85)  # (0.80 + 0.90) / 2
        assert stats.avg_processing_time_ms == pytest.approx(150.0)  # (100 + 200) / 2

    @pytest.mark.asyncio
    async def test_aggregate_stats_unique_sessions(self, history_store: InMemoryTriageHistoryStore):
        """Should correctly count unique sessions."""
        events = [
            TriageEvent(
                timestamp=datetime.utcnow(),
                session_id="session-A",
                risk_level=RiskLevel.LOW,
                emotional_state=EmotionalState.CALM,
                urgency_score=10,
                confidence=0.8,
                modality=InputModality.TEXT,
            ),
            TriageEvent(
                timestamp=datetime.utcnow(),
                session_id="session-A",  # Same session
                risk_level=RiskLevel.MEDIUM,
                emotional_state=EmotionalState.ANXIOUS,
                urgency_score=40,
                confidence=0.8,
                modality=InputModality.TEXT,
            ),
            TriageEvent(
                timestamp=datetime.utcnow(),
                session_id="session-B",  # Different session
                risk_level=RiskLevel.LOW,
                emotional_state=EmotionalState.CALM,
                urgency_score=15,
                confidence=0.8,
                modality=InputModality.TEXT,
            ),
        ]
        
        for event in events:
            await history_store.record_event(event)
        
        stats = await history_store.get_aggregate_stats()
        assert stats.unique_sessions == 2

    @pytest.mark.asyncio
    async def test_aggregate_stats_risk_percentages(self, history_store: InMemoryTriageHistoryStore):
        """Should correctly compute risk level percentages."""
        # Create 4 events: 2 LOW, 1 MEDIUM, 1 HIGH
        risk_levels = [RiskLevel.LOW, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]
        
        for i, risk in enumerate(risk_levels):
            event = TriageEvent(
                timestamp=datetime.utcnow(),
                session_id=f"pct{i}",
                risk_level=risk,
                emotional_state=EmotionalState.CALM,
                urgency_score=10,
                confidence=0.8,
                modality=InputModality.TEXT,
            )
            await history_store.record_event(event)
        
        stats = await history_store.get_aggregate_stats()
        
        assert stats.risk_percentages.get("low", 0) == 50.0  # 2/4 * 100
        assert stats.risk_percentages.get("medium", 0) == 25.0  # 1/4 * 100
        assert stats.risk_percentages.get("high", 0) == 25.0  # 1/4 * 100


class TestBoundedBuffer:
    """Tests for bounded buffer behavior."""

    @pytest.mark.asyncio
    async def test_bounded_buffer_enforces_max(self):
        """Should enforce max_events limit."""
        store = InMemoryTriageHistoryStore(max_events=5, store_text_snippets=False)
        
        # Record more events than max
        for i in range(10):
            event = TriageEvent(
                timestamp=datetime.utcnow(),
                session_id=f"bounded{i}",
                risk_level=RiskLevel.LOW,
                emotional_state=EmotionalState.CALM,
                urgency_score=i * 10,
                confidence=0.8,
                modality=InputModality.TEXT,
            )
            await store.record_event(event)
        
        events = await store.get_recent_events(limit=100)
        
        # Should only have max_events
        assert len(events) == 5
        
        # Should have the newest events (5-9)
        session_ids = [e.session_id for e in events]
        assert "bounded9" in session_ids
        assert "bounded5" in session_ids
        assert "bounded0" not in session_ids  # Old event should be gone


class TestTextSnippets:
    """Tests for text snippet handling."""

    @pytest.mark.asyncio
    async def test_snippets_not_stored_when_disabled(self, history_store: InMemoryTriageHistoryStore):
        """Text snippets should not be stored when disabled."""
        event = TriageEvent(
            timestamp=datetime.utcnow(),
            session_id="nosnip",
            risk_level=RiskLevel.LOW,
            emotional_state=EmotionalState.CALM,
            urgency_score=10,
            confidence=0.8,
            modality=InputModality.TEXT,
            text_snippet="This should not be stored",
        )
        
        await history_store.record_event(event)
        
        stats = await history_store.get_aggregate_stats()
        # No example snippets should be stored
        for risk_stats in stats.risk_level_stats.values():
            assert len(risk_stats.example_snippets) == 0

    @pytest.mark.asyncio
    async def test_snippets_stored_when_enabled(
        self, history_store_with_snippets: InMemoryTriageHistoryStore
    ):
        """Text snippets should be stored when enabled."""
        event = TriageEvent(
            timestamp=datetime.utcnow(),
            session_id="yessnip",
            risk_level=RiskLevel.HIGH,
            emotional_state=EmotionalState.DISTRESSED,
            urgency_score=75,
            confidence=0.8,
            modality=InputModality.TEXT,
            text_snippet="This IS a test snippet",
        )
        
        await history_store_with_snippets.record_event(event)
        
        stats = await history_store_with_snippets.get_aggregate_stats()
        high_stats = stats.risk_level_stats.get("high")
        assert high_stats is not None
        assert len(high_stats.example_snippets) == 1
        assert "test snippet" in high_stats.example_snippets[0]


class TestClearStore:
    """Tests for clearing the store."""

    @pytest.mark.asyncio
    async def test_clear_removes_all_events(
        self, history_store: InMemoryTriageHistoryStore, sample_triage_events: list[TriageEvent]
    ):
        """Clear should remove all events."""
        for event in sample_triage_events:
            await history_store.record_event(event)
        
        # Verify events exist
        events_before = await history_store.get_recent_events(limit=100)
        assert len(events_before) > 0
        
        # Clear
        await history_store.clear()
        
        # Verify empty
        events_after = await history_store.get_recent_events(limit=100)
        assert len(events_after) == 0

    @pytest.mark.asyncio
    async def test_clear_resets_statistics(
        self, history_store: InMemoryTriageHistoryStore, sample_triage_events: list[TriageEvent]
    ):
        """Clear should reset aggregate statistics."""
        for event in sample_triage_events:
            await history_store.record_event(event)
        
        await history_store.clear()
        
        stats = await history_store.get_aggregate_stats()
        assert stats.total_events == 0
        assert stats.unique_sessions == 0


class TestFactoryFunction:
    """Tests for the create_history_store factory."""

    def test_create_with_analytics_enabled(self, test_settings: Settings):
        """Should create store when analytics enabled."""
        store = create_history_store(test_settings)
        assert store is not None
        assert isinstance(store, InMemoryTriageHistoryStore)

    def test_create_with_analytics_disabled(self, test_settings_analytics_disabled: Settings):
        """Should create no-op store when analytics disabled."""
        store = create_history_store(test_settings_analytics_disabled)
        # Should be a no-op store (not InMemoryTriageHistoryStore)
        assert store is not None
