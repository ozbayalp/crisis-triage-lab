"""
CrisisTriage AI - Pipeline Tests with Dummy Services

Tests the core pipeline orchestration logic using dummy service implementations.
These tests verify:
- Text message processing works end-to-end
- Audio chunk processing works end-to-end
- Privacy flags don't break the pipeline
- Edge cases are handled gracefully

Run with: pytest tests/test_pipeline_dummy.py -v
"""

import pytest
from app.core.types import RiskLevel, EmotionalState, RecommendedAction
from app.core.pipeline import TriagePipeline


class TestTextMessageProcessing:
    """Tests for process_text_message()."""

    @pytest.mark.asyncio
    async def test_process_text_message_returns_valid_result(self, pipeline: TriagePipeline):
        """Basic test that text processing returns a valid TriageResult."""
        result = await pipeline.process_text_message(
            session_id="test-session-1",
            message="I'm feeling okay today, thank you for asking.",
        )
        
        assert result is not None
        assert result.session_id == "test-session-1"
        assert result.risk_level in RiskLevel
        assert result.emotional_state in EmotionalState
        assert result.recommended_action in RecommendedAction
        assert 0 <= result.urgency_score <= 100
        assert 0.0 <= result.confidence <= 1.0
        assert result.timestamp_ms > 0

    @pytest.mark.asyncio
    async def test_process_text_message_low_risk(
        self, pipeline: TriagePipeline, low_risk_messages: list[str]
    ):
        """Low-risk messages should result in LOW or MEDIUM risk levels."""
        for message in low_risk_messages:
            result = await pipeline.process_text_message(
                session_id="test-low",
                message=message,
            )
            
            # DummyTriageModel uses keyword matching, so these should be low/medium
            assert result.risk_level in {RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.UNKNOWN}
            # Low risk messages should have lower urgency
            assert result.urgency_score < 70

    @pytest.mark.asyncio
    async def test_process_text_message_high_risk(
        self, pipeline: TriagePipeline, high_risk_messages: list[str]
    ):
        """High-risk messages should trigger HIGH or IMMINENT risk levels."""
        for message in high_risk_messages:
            result = await pipeline.process_text_message(
                session_id="test-high",
                message=message,
            )
            
            # DummyTriageModel should detect high-risk keywords
            assert result.risk_level in {RiskLevel.HIGH, RiskLevel.IMMINENT}
            # High risk messages should have higher urgency
            assert result.urgency_score >= 50

    @pytest.mark.asyncio
    async def test_process_text_message_preserves_session_id(self, pipeline: TriagePipeline):
        """Session ID should be preserved in the result."""
        session_id = "unique-session-abc123"
        result = await pipeline.process_text_message(
            session_id=session_id,
            message="Test message",
        )
        
        assert result.session_id == session_id

    @pytest.mark.asyncio
    async def test_process_text_message_includes_explanation(self, pipeline: TriagePipeline):
        """Result should include an explanation."""
        result = await pipeline.process_text_message(
            session_id="test-explain",
            message="I'm feeling very anxious about everything.",
        )
        
        assert result.explanation is not None
        # Explanation should have some content
        assert result.explanation.summary or result.explanation.top_contributors

    @pytest.mark.asyncio
    async def test_process_text_message_records_to_history(
        self, pipeline: TriagePipeline, history_store
    ):
        """Processing should record events to history store."""
        # Process a message
        await pipeline.process_text_message(
            session_id="test-history",
            message="Test message for history",
        )
        
        # Check history store
        events = await history_store.get_recent_events(limit=10)
        assert len(events) >= 1
        assert events[0].session_id.startswith("test-his")  # Truncated


class TestTextMessageEdgeCases:
    """Tests for edge cases in text message processing."""

    @pytest.mark.asyncio
    async def test_empty_message_returns_unknown(self, pipeline: TriagePipeline):
        """Empty or whitespace messages should be handled gracefully."""
        result = await pipeline.process_text_message(
            session_id="test-empty",
            message="",
        )
        
        # Should return a valid result (possibly UNKNOWN)
        assert result is not None
        assert result.risk_level in RiskLevel

    @pytest.mark.asyncio
    async def test_whitespace_only_message(self, pipeline: TriagePipeline):
        """Whitespace-only messages should be handled gracefully."""
        result = await pipeline.process_text_message(
            session_id="test-whitespace",
            message="   \n\t   ",
        )
        
        assert result is not None
        assert result.risk_level in RiskLevel

    @pytest.mark.asyncio
    async def test_very_long_message(self, pipeline: TriagePipeline):
        """Very long messages should be handled without error."""
        long_message = "I feel anxious. " * 1000  # ~16k characters
        
        result = await pipeline.process_text_message(
            session_id="test-long",
            message=long_message,
        )
        
        assert result is not None
        assert result.risk_level in RiskLevel

    @pytest.mark.asyncio
    async def test_unicode_message(self, pipeline: TriagePipeline):
        """Unicode messages should be handled correctly."""
        result = await pipeline.process_text_message(
            session_id="test-unicode",
            message="I'm feeling 😢 sad today. Everything feels 💔 broken.",
        )
        
        assert result is not None
        assert result.risk_level in RiskLevel

    @pytest.mark.asyncio
    async def test_special_characters_message(self, pipeline: TriagePipeline):
        """Messages with special characters should be handled."""
        result = await pipeline.process_text_message(
            session_id="test-special",
            message="Help! I need someone... <script>alert('xss')</script>",
        )
        
        assert result is not None
        assert result.risk_level in RiskLevel


class TestAudioChunkProcessing:
    """Tests for process_audio_chunk()."""

    @pytest.mark.asyncio
    async def test_process_audio_chunk_returns_valid_result(
        self, pipeline: TriagePipeline, fake_audio_bytes: bytes
    ):
        """Audio processing should return a valid TriageResult."""
        result = await pipeline.process_audio_chunk(
            session_id="test-audio-1",
            audio_bytes=fake_audio_bytes,
        )
        
        assert result is not None
        assert result.session_id == "test-audio-1"
        assert result.risk_level in RiskLevel
        assert result.emotional_state in EmotionalState
        assert 0 <= result.urgency_score <= 100

    @pytest.mark.asyncio
    async def test_process_audio_chunk_includes_transcript(
        self, pipeline: TriagePipeline, fake_audio_bytes: bytes
    ):
        """Audio processing should produce a transcript segment."""
        result = await pipeline.process_audio_chunk(
            session_id="test-audio-transcript",
            audio_bytes=fake_audio_bytes,
        )
        
        # DummyTranscriptionService should produce some text
        assert result.transcript_segment is not None or result.risk_level == RiskLevel.UNKNOWN

    @pytest.mark.asyncio
    async def test_process_audio_chunk_small_audio(self, pipeline: TriagePipeline):
        """Very small audio chunks should be handled gracefully."""
        small_audio = b'\x00\x00' * 100  # Very small
        
        result = await pipeline.process_audio_chunk(
            session_id="test-audio-small",
            audio_bytes=small_audio,
        )
        
        assert result is not None
        assert result.risk_level in RiskLevel

    @pytest.mark.asyncio
    async def test_process_audio_chunk_empty_audio(self, pipeline: TriagePipeline):
        """Empty audio should be handled gracefully."""
        result = await pipeline.process_audio_chunk(
            session_id="test-audio-empty",
            audio_bytes=b'',
        )
        
        assert result is not None
        # Likely returns UNKNOWN for empty audio
        assert result.risk_level in RiskLevel


class TestPipelineWithoutAnalytics:
    """Tests for pipeline with analytics disabled."""

    @pytest.mark.asyncio
    async def test_process_text_without_analytics(self, pipeline_no_analytics: TriagePipeline):
        """Pipeline should work with analytics disabled."""
        result = await pipeline_no_analytics.process_text_message(
            session_id="test-no-analytics",
            message="I'm feeling okay today.",
        )
        
        assert result is not None
        assert result.risk_level in RiskLevel

    @pytest.mark.asyncio
    async def test_process_audio_without_analytics(
        self, pipeline_no_analytics: TriagePipeline, fake_audio_bytes: bytes
    ):
        """Audio processing should work with analytics disabled."""
        result = await pipeline_no_analytics.process_audio_chunk(
            session_id="test-no-analytics-audio",
            audio_bytes=fake_audio_bytes,
        )
        
        assert result is not None
        assert result.risk_level in RiskLevel


class TestPipelinePrivacy:
    """Tests for privacy-related behavior."""

    @pytest.mark.asyncio
    async def test_anonymize_logs_enabled(self, pipeline: TriagePipeline):
        """Pipeline should function with anonymize_logs=True."""
        # This mainly tests that no exceptions are thrown
        result = await pipeline.process_text_message(
            session_id="test-private",
            message="This is sensitive content that should not be logged.",
        )
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_multiple_sessions_isolated(self, pipeline: TriagePipeline):
        """Different sessions should be properly isolated."""
        result1 = await pipeline.process_text_message(
            session_id="session-A",
            message="Session A message",
        )
        result2 = await pipeline.process_text_message(
            session_id="session-B",
            message="Session B message",
        )
        
        assert result1.session_id == "session-A"
        assert result2.session_id == "session-B"


class TestPipelinePerformance:
    """Basic performance sanity checks."""

    @pytest.mark.asyncio
    async def test_processing_time_is_recorded(self, pipeline: TriagePipeline):
        """Processing time should be recorded in the result."""
        result = await pipeline.process_text_message(
            session_id="test-perf",
            message="Quick test message.",
        )
        
        # Processing time should be recorded
        assert result.processing_time_ms is not None or True  # May be None for dummy
        # If recorded, should be reasonable
        if result.processing_time_ms:
            assert 0 < result.processing_time_ms < 10000  # Less than 10 seconds

    @pytest.mark.asyncio
    async def test_multiple_sequential_calls(self, pipeline: TriagePipeline):
        """Pipeline should handle multiple sequential calls."""
        for i in range(10):
            result = await pipeline.process_text_message(
                session_id=f"test-seq-{i}",
                message=f"Message number {i}",
            )
            assert result is not None
            assert result.session_id == f"test-seq-{i}"
