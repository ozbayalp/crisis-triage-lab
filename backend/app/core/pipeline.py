"""
CrisisTriage AI - Triage Pipeline Orchestrator

Central orchestration layer that coordinates all triage processing.
This is the single entry point for both REST and WebSocket triage flows.

Architecture:
    The pipeline follows a staged processing model:
    
    1. INPUT STAGE: Receive text or audio from API layer
    2. TRANSCRIPTION STAGE: Convert audio to text (if audio input)
    3. FEATURE EXTRACTION STAGE: Extract prosody features (if audio)
    4. INFERENCE STAGE: Run triage model on text + features
    5. OUTPUT STAGE: Return structured TriageResult
    
    Each stage is handled by a pluggable service, enabling:
    - Easy testing with dummy implementations
    - Swapping models without changing API code
    - A/B testing different model versions
    - Graceful degradation if a service fails

Design Principles:
    - Stateless: No persistent state in the pipeline itself
    - Async-first: All processing is async for WebSocket compatibility
    - Observable: Structured logging at each stage
    - Fail-safe: Errors return safe default results, never crash
    - Privacy-aware: Respects config flags for data handling

Usage:
    from app.core.pipeline import TriagePipeline
    from app.services import DummyTranscriptionService, DummyProsodyExtractor, DummyTriageModel
    from app.config import get_settings
    
    pipeline = TriagePipeline(
        transcription=DummyTranscriptionService(),
        prosody=DummyProsodyExtractor(),
        model=DummyTriageModel(),
        settings=get_settings(),
    )
    
    result = await pipeline.process_audio_chunk(session_id, audio_bytes)
    result = await pipeline.process_text_message(session_id, "I'm feeling overwhelmed")
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Optional, Callable, Any
from dataclasses import dataclass

from app.config import Settings
from app.core.types import (
    SessionId,
    AudioChunk,
    TriageResult,
    ProsodyFeatures,
    TranscriptionResult,
    PipelineContext,
    EmotionalState,
    RiskLevel,
    RecommendedAction,
    TriageExplanation,
    InputModality,
    TriageEvent,
)
from app.services.transcription import TranscriptionService, TranscriptionError
from app.services.prosody import ProsodyExtractor, ProsodyExtractionError
from app.services.triage_model import TriageModel, TriageModelError

# Type hint only import to avoid circular dependency
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app.core.history_store import TriageHistoryStore

logger = logging.getLogger(__name__)


# =============================================================================
# Pipeline Metrics (for observability)
# =============================================================================

@dataclass
class PipelineMetrics:
    """Metrics for a single pipeline execution."""
    request_id: str
    session_id: str
    input_type: str  # "text" | "audio"
    transcription_ms: Optional[float] = None
    prosody_ms: Optional[float] = None
    inference_ms: Optional[float] = None
    total_ms: Optional[float] = None
    success: bool = True
    error_stage: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "session_id": self.session_id[:8] + "...",  # Truncate for logs
            "input_type": self.input_type,
            "transcription_ms": round(self.transcription_ms, 2) if self.transcription_ms else None,
            "prosody_ms": round(self.prosody_ms, 2) if self.prosody_ms else None,
            "inference_ms": round(self.inference_ms, 2) if self.inference_ms else None,
            "total_ms": round(self.total_ms, 2) if self.total_ms else None,
            "success": self.success,
            "error_stage": self.error_stage,
        }


# =============================================================================
# Pipeline Hooks (for extensibility)
# =============================================================================

PipelineHook = Callable[[PipelineContext, TriageResult], None]
"""Hook function called after successful pipeline execution."""


# =============================================================================
# Triage Pipeline
# =============================================================================

class TriagePipeline:
    """
    Central orchestrator for triage processing.
    
    Coordinates transcription, prosody extraction, and model inference
    into a unified processing flow. Handles both text and audio inputs.
    
    Attributes:
        transcription: Service for speech-to-text conversion
        prosody: Service for acoustic feature extraction
        model: Triage model for risk/emotion assessment
        settings: Application configuration
    """

    def __init__(
        self,
        transcription: TranscriptionService,
        prosody: ProsodyExtractor,
        model: TriageModel,
        settings: Settings,
        history_store: Optional["TriageHistoryStore"] = None,
    ):
        """
        Initialize the triage pipeline.
        
        Args:
            transcription: Transcription service implementation
            prosody: Prosody extraction service implementation
            model: Triage model implementation
            settings: Application settings (includes privacy config)
            history_store: Optional history store for analytics
        """
        self._transcription = transcription
        self._prosody = prosody
        self._model = model
        self._settings = settings
        self._history_store = history_store
        
        # Hooks for extensibility (e.g., persistence, alerts)
        self._post_hooks: list[PipelineHook] = []
        
        # Metrics callback (for monitoring systems)
        self._metrics_callback: Optional[Callable[[PipelineMetrics], None]] = None
        
        logger.info(
            "TriagePipeline initialized: transcription=%s, prosody=%s, model=%s, analytics=%s",
            transcription.model_id,
            prosody.extractor_id,
            model.model_id,
            "enabled" if history_store else "disabled",
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def process_text_message(
        self,
        session_id: str,
        message: str,
        source: str = "rest",
    ) -> TriageResult:
        """
        Process a text message through the triage pipeline.
        
        This is the primary entry point for text-only input (e.g., chat).
        Prosody features are not available for text input.
        
        Args:
            session_id: Unique session identifier
            message: Text message to analyze
            source: Origin of the request ("rest", "websocket", "batch")
            
        Returns:
            TriageResult with assessment outputs
            
        Note:
            Even if the message is empty or inference fails, this method
            returns a valid TriageResult (with UNKNOWN states).
        """
        request_id = self._generate_request_id()
        start_time = time.time()
        
        ctx = PipelineContext(
            session_id=SessionId(session_id),
            request_id=request_id,
            source=source,
        )
        
        metrics = PipelineMetrics(
            request_id=request_id,
            session_id=session_id,
            input_type="text",
        )
        
        self._log_input(ctx, message=message)
        
        try:
            # For text-only, skip transcription and prosody
            result = await self._run_inference(ctx, message, prosody=None)
            
            metrics.inference_ms = ctx.elapsed_ms()
            metrics.total_ms = (time.time() - start_time) * 1000
            
            # Execute post-hooks (e.g., persistence, alerting, analytics)
            await self._execute_hooks(ctx, result, modality=InputModality.TEXT)
            
            self._log_output(ctx, result, metrics)
            
            return result
            
        except Exception as e:
            metrics.success = False
            metrics.error_stage = "inference"
            metrics.error_message = str(e)
            metrics.total_ms = (time.time() - start_time) * 1000
            
            logger.error(
                "Pipeline error [%s]: %s",
                request_id,
                str(e),
                exc_info=True,
            )
            
            return TriageResult.create_unknown(
                SessionId(session_id),
                reason=f"Processing failed: {type(e).__name__}",
            )
        finally:
            self._emit_metrics(metrics)

    async def process_audio_chunk(
        self,
        session_id: str,
        audio_bytes: bytes,
        source: str = "websocket",
    ) -> TriageResult:
        """
        Process an audio chunk through the full triage pipeline.
        
        This runs the complete pipeline:
        1. Transcribe audio to text
        2. Extract prosody features from audio
        3. Run triage model on text + prosody
        
        Args:
            session_id: Unique session identifier
            audio_bytes: Raw audio (expected: PCM 16-bit, 16kHz, mono)
            source: Origin of the request
            
        Returns:
            TriageResult with assessment outputs
            
        Note:
            Audio is processed in memory and NOT persisted by default.
            Set STORE_AUDIO=true in config to enable audio storage.
        """
        request_id = self._generate_request_id()
        start_time = time.time()
        
        ctx = PipelineContext(
            session_id=SessionId(session_id),
            request_id=request_id,
            source=source,
            raw_audio=AudioChunk(audio_bytes),
        )
        
        metrics = PipelineMetrics(
            request_id=request_id,
            session_id=session_id,
            input_type="audio",
        )
        
        self._log_input(ctx, audio_bytes=audio_bytes)
        
        try:
            # Stage 1: Transcription
            transcription_start = time.time()
            transcription = await self._run_transcription(ctx, audio_bytes)
            metrics.transcription_ms = (time.time() - transcription_start) * 1000
            ctx.transcription = transcription
            
            # Stage 2: Prosody extraction (parallel with transcription in future)
            prosody_start = time.time()
            prosody = await self._run_prosody_extraction(ctx, audio_bytes)
            metrics.prosody_ms = (time.time() - prosody_start) * 1000
            ctx.prosody = prosody
            
            # Stage 3: Inference
            inference_start = time.time()
            result = await self._run_inference(ctx, transcription.text, prosody)
            metrics.inference_ms = (time.time() - inference_start) * 1000
            
            # Attach transcript to result for frontend display
            result.transcript_segment = transcription.text
            result.prosody_features = prosody
            
            metrics.total_ms = (time.time() - start_time) * 1000
            
            # Execute post-hooks (analytics with AUDIO modality)
            await self._execute_hooks(ctx, result, modality=InputModality.AUDIO)
            
            self._log_output(ctx, result, metrics)
            
            return result
            
        except TranscriptionError as e:
            metrics.success = False
            metrics.error_stage = "transcription"
            metrics.error_message = str(e)
            logger.warning("Transcription failed [%s]: %s", request_id, e)
            return TriageResult.create_unknown(
                SessionId(session_id),
                reason="Transcription failed",
            )
            
        except ProsodyExtractionError as e:
            # Prosody failure is non-fatal; continue without prosody
            metrics.prosody_ms = None
            logger.warning("Prosody extraction failed [%s], continuing: %s", request_id, e)
            
            # Still run inference, just without prosody
            result = await self._run_inference(ctx, ctx.transcription.text, prosody=None)
            metrics.total_ms = (time.time() - start_time) * 1000
            return result
            
        except Exception as e:
            metrics.success = False
            metrics.error_stage = "unknown"
            metrics.error_message = str(e)
            metrics.total_ms = (time.time() - start_time) * 1000
            
            logger.error(
                "Pipeline error [%s]: %s",
                request_id,
                str(e),
                exc_info=True,
            )
            
            return TriageResult.create_unknown(
                SessionId(session_id),
                reason=f"Processing failed: {type(e).__name__}",
            )
        finally:
            self._emit_metrics(metrics)

    # -------------------------------------------------------------------------
    # Pipeline Stages (Internal)
    # -------------------------------------------------------------------------

    async def _run_transcription(
        self,
        ctx: PipelineContext,
        audio_bytes: bytes,
    ) -> TranscriptionResult:
        """Run transcription stage."""
        logger.debug("[%s] Starting transcription (%d bytes)", ctx.request_id, len(audio_bytes))
        
        result = await self._transcription.transcribe_async(AudioChunk(audio_bytes))
        
        # Privacy: conditionally log transcript
        if self._settings.store_raw_transcripts:
            logger.debug("[%s] Transcript: %s", ctx.request_id, result.text[:100])
        else:
            logger.debug(
                "[%s] Transcript: [REDACTED, %d chars]",
                ctx.request_id,
                len(result.text),
            )
        
        return result

    async def _run_prosody_extraction(
        self,
        ctx: PipelineContext,
        audio_bytes: bytes,
    ) -> ProsodyFeatures:
        """Run prosody extraction stage."""
        logger.debug("[%s] Starting prosody extraction", ctx.request_id)
        
        features = await self._prosody.extract_async(AudioChunk(audio_bytes))
        
        logger.debug(
            "[%s] Prosody: pitch_mean=%.1f, speech_rate=%.2f, energy=%.2f",
            ctx.request_id,
            features.pitch_mean or 0,
            features.speech_rate or 0,
            features.energy_mean or 0,
        )
        
        return features

    async def _run_inference(
        self,
        ctx: PipelineContext,
        text: str,
        prosody: Optional[ProsodyFeatures],
    ) -> TriageResult:
        """Run triage model inference stage."""
        logger.debug("[%s] Starting inference", ctx.request_id)
        
        result = await self._model.predict_async(
            session_id=ctx.session_id,
            text=text,
            prosody=prosody,
            context=None,  # TODO: Add conversation history
        )
        
        # Attach processing metadata
        result.processing_time_ms = ctx.elapsed_ms()
        
        return result

    # -------------------------------------------------------------------------
    # Hooks and Observability
    # -------------------------------------------------------------------------

    def register_hook(self, hook: PipelineHook) -> None:
        """
        Register a post-processing hook.
        
        Hooks are called after successful pipeline execution with the
        context and result. Use for:
        - Persistence to database/cache
        - Alert triggering for high-risk results
        - Metric emission
        - Audit logging
        
        Args:
            hook: Callable that receives (PipelineContext, TriageResult)
        """
        self._post_hooks.append(hook)
        logger.info("Registered pipeline hook: %s", hook.__name__ if hasattr(hook, '__name__') else str(hook))

    def set_metrics_callback(self, callback: Callable[[PipelineMetrics], None]) -> None:
        """
        Set callback for metrics emission.
        
        Called after every pipeline execution (success or failure).
        Use for integration with monitoring systems (Prometheus, Datadog, etc.)
        """
        self._metrics_callback = callback

    async def _execute_hooks(
        self,
        ctx: PipelineContext,
        result: TriageResult,
        modality: InputModality = InputModality.TEXT,
    ) -> None:
        """Execute all registered post-hooks and record to history store."""
        # Record to history store if enabled
        if self._history_store and self._settings.enable_analytics:
            try:
                event = TriageEvent.from_triage_result(
                    result,
                    modality=modality,
                    store_text=self._settings.store_analytics_text_snippets,
                )
                await self._history_store.record_event(event)
            except Exception as e:
                logger.warning("Failed to record event to history store: %s", e)
        
        # Execute custom hooks
        for hook in self._post_hooks:
            try:
                # Support both sync and async hooks
                hook_result = hook(ctx, result)
                if hasattr(hook_result, '__await__'):
                    await hook_result
            except Exception as e:
                logger.error(
                    "Hook execution failed [%s]: %s",
                    hook.__name__ if hasattr(hook, '__name__') else "unknown",
                    str(e),
                )

    def _emit_metrics(self, metrics: PipelineMetrics) -> None:
        """Emit metrics to callback if configured."""
        if self._metrics_callback:
            try:
                self._metrics_callback(metrics)
            except Exception as e:
                logger.warning("Metrics emission failed: %s", e)

    # -------------------------------------------------------------------------
    # Logging (Privacy-Aware)
    # -------------------------------------------------------------------------

    def _log_input(
        self,
        ctx: PipelineContext,
        message: Optional[str] = None,
        audio_bytes: Optional[bytes] = None,
    ) -> None:
        """Log pipeline input with privacy considerations."""
        if self._settings.anonymize_logs:
            # Minimal logging in anonymized mode
            if audio_bytes:
                logger.info(
                    "[%s] Processing audio: session=%s, bytes=%d",
                    ctx.request_id,
                    ctx.session_id[:8] + "...",
                    len(audio_bytes),
                )
            else:
                logger.info(
                    "[%s] Processing text: session=%s, chars=%d",
                    ctx.request_id,
                    ctx.session_id[:8] + "...",
                    len(message) if message else 0,
                )
        else:
            # Detailed logging (for development)
            if audio_bytes:
                logger.info(
                    "[%s] Processing audio: session=%s, bytes=%d, source=%s",
                    ctx.request_id,
                    ctx.session_id,
                    len(audio_bytes),
                    ctx.source,
                )
            elif message:
                # Still don't log full message content by default
                preview = message[:50] + "..." if len(message) > 50 else message
                logger.info(
                    "[%s] Processing text: session=%s, preview='%s'",
                    ctx.request_id,
                    ctx.session_id,
                    preview,
                )

    def _log_output(
        self,
        ctx: PipelineContext,
        result: TriageResult,
        metrics: PipelineMetrics,
    ) -> None:
        """Log pipeline output."""
        # Always log risk level and timing (non-sensitive)
        logger.info(
            "[%s] Result: risk=%s, urgency=%d, action=%s, total_ms=%.1f",
            ctx.request_id,
            result.risk_level.value,
            result.urgency_score,
            result.recommended_action.value,
            metrics.total_ms or 0,
        )
        
        # Log high-risk results at WARNING level for visibility
        if result.risk_level in (RiskLevel.HIGH, RiskLevel.IMMINENT):
            logger.warning(
                "[%s] HIGH RISK detected: session=%s, risk=%s, urgency=%d",
                ctx.request_id,
                ctx.session_id[:8] + "...",
                result.risk_level.value,
                result.urgency_score,
            )

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _generate_request_id(self) -> str:
        """Generate unique request ID for tracking."""
        return f"req_{uuid.uuid4().hex[:12]}"

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def startup(self) -> None:
        """
        Initialize pipeline resources.
        
        Called during application startup. Use for:
        - Model warmup (load weights, run dummy inference)
        - Connection pool initialization
        - Health check of services
        """
        logger.info("Pipeline startup: warming up model...")
        self._model.warmup()
        logger.info("Pipeline startup complete")

    async def shutdown(self) -> None:
        """
        Clean up pipeline resources.
        
        Called during application shutdown. Use for:
        - Flush pending metrics
        - Close connections
        - Save any cached state
        """
        logger.info("Pipeline shutdown: cleaning up...")
        # TODO: Add cleanup logic as needed
        logger.info("Pipeline shutdown complete")


# =============================================================================
# Factory Function
# =============================================================================

def create_pipeline(
    settings: Settings,
    history_store: Optional["TriageHistoryStore"] = None,
) -> TriagePipeline:
    """
    Factory function to create a configured TriagePipeline.
    
    Selects service implementations based on settings:
    - transcription_backend: "dummy" | "whisper"
    - prosody_backend: "dummy" | "librosa"  
    - triage_model_backend: "dummy" | "neural"
    
    All audio processing is done LOCALLY - no data is sent to external APIs.
    
    IMPORTANT SAFETY NOTICE:
        This pipeline is for RESEARCH AND SIMULATION ONLY.
        Not a medical device. Not suitable for real crisis intervention.
    
    Args:
        settings: Application settings
        history_store: Optional history store for analytics (default: create from settings)
        
    Returns:
        Configured TriagePipeline instance
    """
    from app.services.transcription import DummyTranscriptionService
    from app.services.prosody import DummyProsodyExtractor
    from app.services.triage_model import DummyTriageModel
    from app.core.history_store import create_history_store
    
    # --- History Store ---
    if history_store is None and settings.enable_analytics:
        history_store = create_history_store(settings)
    
    # --- Transcription Service ---
    transcription_backend = getattr(settings, 'transcription_backend', 'dummy').lower()
    
    if transcription_backend == "whisper":
        try:
            from app.services.transcription import WhisperTranscriptionService
            
            logger.info(
                "Initializing WhisperTranscriptionService (model=%s, lang=%s)",
                settings.whisper_model_name,
                settings.whisper_language,
            )
            transcription = WhisperTranscriptionService(
                model_name=settings.whisper_model_name,
                language=settings.whisper_language,
                store_raw_transcripts=settings.store_raw_transcripts,
                anonymize_logs=settings.anonymize_logs,
            )
            logger.info("WhisperTranscriptionService loaded successfully")
            
        except ImportError as e:
            logger.error(
                "Failed to load WhisperTranscriptionService: %s. "
                "Install openai-whisper. Falling back to DummyTranscriptionService.",
                str(e)
            )
            transcription = DummyTranscriptionService()
            
        except Exception as e:
            logger.error(
                "Failed to initialize WhisperTranscriptionService: %s. "
                "Falling back to DummyTranscriptionService.",
                str(e)
            )
            transcription = DummyTranscriptionService()
    else:
        logger.info("Using DummyTranscriptionService (placeholder)")
        transcription = DummyTranscriptionService()
    
    # --- Prosody Extractor ---
    prosody_backend = getattr(settings, 'prosody_backend', 'dummy').lower()
    
    if prosody_backend == "librosa":
        try:
            from app.services.prosody import LibrosaProsodyExtractor
            
            logger.info("Initializing LibrosaProsodyExtractor")
            prosody = LibrosaProsodyExtractor(
                store_prosody_features=settings.store_prosody_features,
                anonymize_logs=settings.anonymize_logs,
            )
            logger.info("LibrosaProsodyExtractor loaded successfully")
            
        except ImportError as e:
            logger.error(
                "Failed to load LibrosaProsodyExtractor: %s. "
                "Install librosa. Falling back to DummyProsodyExtractor.",
                str(e)
            )
            prosody = DummyProsodyExtractor()
            
        except Exception as e:
            logger.error(
                "Failed to initialize LibrosaProsodyExtractor: %s. "
                "Falling back to DummyProsodyExtractor.",
                str(e)
            )
            prosody = DummyProsodyExtractor()
    else:
        logger.info("Using DummyProsodyExtractor (placeholder)")
        prosody = DummyProsodyExtractor()
    
    # --- Triage Model ---
    model_backend = getattr(settings, 'triage_model_backend', 'dummy').lower()
    
    if model_backend == "neural":
        try:
            from app.services.triage_model import NeuralTriageModel
            
            neural_model_dir = getattr(settings, 'neural_model_dir', None)
            if not neural_model_dir:
                raise ValueError(
                    "neural_model_dir must be set when triage_model_backend='neural'"
                )
            
            logger.info("Initializing NeuralTriageModel from %s", neural_model_dir)
            model = NeuralTriageModel(
                model_dir=neural_model_dir,
                settings=settings,
            )
            logger.info("NeuralTriageModel loaded successfully")
            
        except ImportError as e:
            logger.error(
                "Failed to load NeuralTriageModel: %s. "
                "Ensure transformers and torch are installed. "
                "Falling back to DummyTriageModel.",
                str(e)
            )
            model = DummyTriageModel()
            
        except FileNotFoundError as e:
            logger.error(
                "Neural model artifact not found: %s. "
                "Train a model first or set triage_model_backend='dummy'. "
                "Falling back to DummyTriageModel.",
                str(e)
            )
            model = DummyTriageModel()
            
        except Exception as e:
            logger.error(
                "Failed to initialize NeuralTriageModel: %s. "
                "Falling back to DummyTriageModel.",
                str(e)
            )
            model = DummyTriageModel()
    else:
        logger.info("Using DummyTriageModel (heuristic-based)")
        model = DummyTriageModel()
    
    # Log pipeline configuration summary
    logger.info(
        "Pipeline configured: transcription=%s, prosody=%s, model=%s, analytics=%s",
        type(transcription).__name__,
        type(prosody).__name__,
        type(model).__name__,
        type(history_store).__name__ if history_store else "disabled",
    )
    
    return TriagePipeline(
        transcription=transcription,
        prosody=prosody,
        model=model,
        settings=settings,
        history_store=history_store,
    )
