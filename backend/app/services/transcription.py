"""
CrisisTriage AI - Transcription Service

Provides speech-to-text functionality for the triage pipeline.

Architecture:
    - Protocol defines the interface for all transcription implementations
    - DummyTranscriptionService: Placeholder for development/testing
    - WhisperTranscriptionService: Production implementation (TODO)

Integration Notes:
    Real Whisper integration should consider:
    1. Model size selection (tiny/base/small/medium/large) based on latency/accuracy tradeoffs
    2. Streaming vs batch: Whisper is primarily batch-oriented, but can be adapted
       for streaming using chunked processing with overlapping windows
    3. GPU acceleration: Use faster-whisper or whisper.cpp for production
    4. VAD (Voice Activity Detection): Pre-filter silence for efficiency
    5. Language detection: Whisper supports multilingual; may need to force English
    
Privacy Considerations:
    - Raw audio should never be logged in production
    - Transcripts may contain PII; handle according to STORE_RAW_TRANSCRIPTS config
    - Consider on-device processing for maximum privacy
"""

from __future__ import annotations

import hashlib
import logging
from abc import abstractmethod
from typing import Protocol, Optional, runtime_checkable

from app.core.types import TranscriptionResult, AudioChunk

logger = logging.getLogger(__name__)


# =============================================================================
# Protocol (Interface)
# =============================================================================

@runtime_checkable
class TranscriptionService(Protocol):
    """
    Protocol for speech-to-text transcription services.
    
    Implementations must provide both synchronous and async transcription.
    The pipeline will call the appropriate method based on context.
    
    Design Notes:
        - Using Protocol for structural subtyping (duck typing with type hints)
        - Implementations are free to batch/stream internally
        - Audio format expectation: PCM 16-bit, 16kHz, mono
    """

    @abstractmethod
    def transcribe(self, audio: AudioChunk) -> TranscriptionResult:
        """
        Transcribe audio to text (synchronous).
        
        Args:
            audio: Raw audio bytes (PCM 16-bit, 16kHz, mono)
            
        Returns:
            TranscriptionResult with text and metadata
            
        Raises:
            TranscriptionError: If transcription fails
        """
        ...

    @abstractmethod
    async def transcribe_async(self, audio: AudioChunk) -> TranscriptionResult:
        """
        Transcribe audio to text (asynchronous).
        
        Preferred for API handlers to avoid blocking the event loop.
        
        Args:
            audio: Raw audio bytes (PCM 16-bit, 16kHz, mono)
            
        Returns:
            TranscriptionResult with text and metadata
        """
        ...

    @abstractmethod
    def transcribe_streaming(
        self, 
        audio_chunks: list[AudioChunk],
        overlap_ms: int = 200,
    ) -> list[TranscriptionResult]:
        """
        Transcribe a sequence of audio chunks with overlap handling.
        
        For real-time streaming, audio arrives in chunks. This method
        handles the complexity of overlapping windows to avoid cutting
        words at chunk boundaries.
        
        Args:
            audio_chunks: List of sequential audio chunks
            overlap_ms: Overlap between chunks in milliseconds
            
        Returns:
            List of TranscriptionResults, one per logical segment
        """
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return identifier for the underlying model (for logging/tracking)."""
        ...


# =============================================================================
# Exceptions
# =============================================================================

class TranscriptionError(Exception):
    """Raised when transcription fails."""
    
    def __init__(self, message: str, audio_hash: Optional[str] = None):
        super().__init__(message)
        self.audio_hash = audio_hash  # For debugging without storing raw audio


# =============================================================================
# Dummy Implementation (Development/Testing)
# =============================================================================

class DummyTranscriptionService:
    """
    Placeholder transcription service for development and testing.
    
    Returns deterministic dummy transcriptions based on audio length.
    Useful for:
        - Testing pipeline integration without GPU/model dependencies
        - Benchmarking pipeline overhead independent of ASR latency
        - Development when real audio is not available
    
    WARNING: This is NOT suitable for production use.
    """

    def __init__(self, simulated_latency_ms: float = 50.0):
        """
        Initialize dummy transcription service.
        
        Args:
            simulated_latency_ms: Artificial delay to simulate processing time
        """
        self._simulated_latency_ms = simulated_latency_ms
        self._call_count = 0
        
        # Canned responses for variety in testing - realistic conversational paragraphs
        self._canned_responses = [
            # Low risk - coping well
            "Thank you so much for taking my call. I've been working through some things lately and I think I'm actually making progress. My therapist suggested I reach out when I feel overwhelmed, and right now I just needed to hear another voice. The breathing exercises have been helping a lot, and I've been journaling every night like she suggested.",
            
            # Low-medium risk - mild distress
            "I've been feeling really overwhelmed lately with everything going on. Work has been incredibly stressful, and I haven't been sleeping well. My partner and I have been arguing more than usual, and I just feel like I'm constantly walking on eggshells. I know it's probably just a rough patch, but some days it's hard to see the light at the end of the tunnel.",
            
            # Medium risk - significant distress
            "Things have been really difficult at work and home for the past few months. I lost my job three weeks ago and I haven't told my family yet because I'm ashamed. Every day I wake up and apply to jobs but nothing comes through. The bills are piling up and I don't know how much longer I can keep this up. I feel like I'm failing everyone who depends on me.",
            
            # Medium-high risk - hopelessness emerging
            "I'm not sure what to do anymore. I've tried everything - therapy, medication, exercise, all the things they tell you to do. But nothing seems to help for very long. Sometimes I wonder if things will ever get better or if this is just how my life is going to be forever. I'm exhausted from fighting this feeling every single day.",
            
            # High risk - active crisis indicators
            "I just need someone to talk to because I don't know how much longer I can keep going like this. I haven't eaten in two days and I can't stop crying. My friends have stopped calling because I always cancel on them. I feel completely alone even when people are around me. Sometimes I think everyone would be better off without me dragging them down.",
            
            # High risk - sleep disturbance and isolation
            "It's been really hard to sleep recently. I lie awake for hours thinking about everything I've done wrong and everyone I've disappointed. When I do fall asleep, I have nightmares and wake up more tired than before. I've stopped going out because I can't face people anymore. My apartment is a mess and I just don't have the energy to care.",
            
            # High risk - feeling misunderstood, burden
            "I feel like no one understands what I'm going through, not even my family. They keep telling me to just cheer up or think positive, but they don't get it. It's not that simple. Sometimes I feel like I'm just a burden on everyone around me. They'd probably be relieved if they didn't have to worry about me anymore.",
            
            # De-escalation - finding support
            "Thank you for listening to me tonight. I really needed this. I've been holding everything in for so long because I didn't want to bother anyone. But talking about it actually helps. I think I'm going to call my sister tomorrow - we used to be close and maybe it's time to reconnect. I forgot what it felt like to have someone really listen.",
        ]

    @property
    def model_id(self) -> str:
        return "dummy-transcription-v0.0.1"

    def transcribe(self, audio: AudioChunk) -> TranscriptionResult:
        """
        Return a dummy transcription based on audio characteristics.
        
        The response is deterministic based on audio hash to enable
        reproducible testing.
        """
        self._call_count += 1
        
        # Use audio hash to select response (deterministic)
        audio_hash = hashlib.md5(audio).hexdigest()
        response_idx = int(audio_hash[:8], 16) % len(self._canned_responses)
        
        # Estimate duration from byte length (16kHz, 16-bit mono = 32000 bytes/sec)
        estimated_duration_ms = len(audio) / 32.0
        
        logger.debug(
            "DummyTranscription: processed %d bytes (est. %.0fms), call #%d",
            len(audio),
            estimated_duration_ms,
            self._call_count,
        )
        
        return TranscriptionResult(
            text=self._canned_responses[response_idx],
            is_final=True,
            confidence=0.95,  # Dummy confidence
            start_time_ms=0,
            end_time_ms=int(estimated_duration_ms),
            language="en",
        )

    async def transcribe_async(self, audio: AudioChunk) -> TranscriptionResult:
        """Async wrapper around sync transcription."""
        import asyncio
        
        # Simulate processing latency
        await asyncio.sleep(self._simulated_latency_ms / 1000.0)
        
        return self.transcribe(audio)

    def transcribe_streaming(
        self,
        audio_chunks: list[AudioChunk],
        overlap_ms: int = 200,
    ) -> list[TranscriptionResult]:
        """
        Process multiple chunks, returning one result per chunk.
        
        In a real implementation, this would handle:
        - Buffering audio for optimal chunk sizes
        - Overlap processing to avoid word boundary issues
        - Streaming output with partial results
        """
        results = []
        cumulative_time_ms = 0
        
        for chunk in audio_chunks:
            result = self.transcribe(chunk)
            
            # Adjust timestamps for cumulative position
            chunk_duration = len(chunk) / 32.0  # ms
            result = TranscriptionResult(
                text=result.text,
                is_final=True,
                confidence=result.confidence,
                start_time_ms=int(cumulative_time_ms),
                end_time_ms=int(cumulative_time_ms + chunk_duration),
                language=result.language,
            )
            results.append(result)
            cumulative_time_ms += chunk_duration - overlap_ms
            
        return results


# =============================================================================
# Whisper Implementation
# =============================================================================

class WhisperTranscriptionService:
    """
    Production transcription service using OpenAI Whisper.
    
    All audio processing is done LOCALLY - no audio is sent to external APIs.
    
    IMPORTANT SAFETY NOTICE:
        This service is for RESEARCH AND SIMULATION ONLY.
        Not a medical device. Not suitable for real crisis intervention.
    
    Privacy Notes:
        - Audio is processed in-memory only
        - Raw audio is not persisted unless explicitly configured
        - Transcripts are not logged unless store_raw_transcripts=True
    
    Audio Format:
        Expects PCM 16-bit, 16kHz, mono audio bytes.
    """
    
    def __init__(
        self,
        model_name: str = "base",
        language: str = "en",
        device: str = "auto",
        store_raw_transcripts: bool = False,
        anonymize_logs: bool = True,
    ):
        """
        Initialize Whisper transcription service.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            language: Language code to force (e.g., "en")
            device: Device for inference ("auto", "cpu", "cuda")
            store_raw_transcripts: Whether transcripts may be logged/stored
            anonymize_logs: Whether to scrub sensitive info from logs
        """
        self._model_name = model_name
        self._language = language
        self._store_raw_transcripts = store_raw_transcripts
        self._anonymize_logs = anonymize_logs
        self._model = None
        self._device = device
        
        # Load model on init
        self._load_model()
    
    def _load_model(self) -> None:
        """Load Whisper model into memory."""
        try:
            import whisper
            import torch
            
            # Resolve device
            if self._device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self._device
            
            logger.info(
                "Loading Whisper model '%s' on device '%s'...",
                self._model_name, device
            )
            
            self._model = whisper.load_model(self._model_name, device=device)
            self._torch = torch
            
            logger.info("Whisper model loaded successfully")
            
        except ImportError as e:
            raise ImportError(
                "WhisperTranscriptionService requires 'openai-whisper'. "
                "Install with: pip install openai-whisper"
            ) from e
    
    @property
    def model_id(self) -> str:
        return f"whisper-{self._model_name}"
    
    def transcribe(self, audio: AudioChunk) -> TranscriptionResult:
        """
        Transcribe audio to text.
        
        Args:
            audio: Raw audio bytes (PCM 16-bit, 16kHz, mono)
            
        Returns:
            TranscriptionResult with transcribed text
        """
        import numpy as np
        import time
        
        start_time = time.time()
        
        # Validate audio
        if len(audio) < 100:
            logger.warning("Audio chunk too short for transcription: %d bytes", len(audio))
            return TranscriptionResult(
                text="",
                is_final=True,
                confidence=0.0,
                start_time_ms=0,
                end_time_ms=0,
                language=self._language,
            )
        
        try:
            # Convert bytes to numpy array (PCM 16-bit to float32 normalized)
            audio_np = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Calculate duration
            duration_ms = len(audio) / 32.0  # 16kHz, 16-bit = 32 bytes/ms
            
            # Transcribe with Whisper
            result = self._model.transcribe(
                audio_np,
                language=self._language,
                fp16=self._torch.cuda.is_available(),
                verbose=False,
            )
            
            text = result["text"].strip()
            
            # Calculate confidence from segment probabilities
            confidence = self._extract_confidence(result)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Privacy-aware logging
            if self._store_raw_transcripts:
                logger.debug(
                    "Whisper transcribed: '%s' (conf=%.2f, %.0fms)",
                    text[:100] + "..." if len(text) > 100 else text,
                    confidence,
                    processing_time
                )
            else:
                logger.debug(
                    "Whisper transcribed: %d chars (conf=%.2f, %.0fms)",
                    len(text), confidence, processing_time
                )
            
            return TranscriptionResult(
                text=text,
                is_final=True,
                confidence=confidence,
                start_time_ms=0,
                end_time_ms=int(duration_ms),
                language=result.get("language", self._language),
            )
            
        except Exception as e:
            logger.error("Whisper transcription failed: %s", str(e))
            return TranscriptionResult(
                text="",
                is_final=True,
                confidence=0.0,
                start_time_ms=0,
                end_time_ms=0,
                language=self._language,
            )
    
    def _extract_confidence(self, result: dict) -> float:
        """
        Extract confidence score from Whisper result.
        
        Uses average of segment-level no-speech probabilities
        inverted to get speech confidence.
        """
        try:
            segments = result.get("segments", [])
            if not segments:
                return 0.5  # Default if no segments
            
            # Average the inverse of no_speech_prob
            confidences = []
            for seg in segments:
                no_speech = seg.get("no_speech_prob", 0.5)
                confidences.append(1.0 - no_speech)
            
            return sum(confidences) / len(confidences)
            
        except Exception:
            return 0.5
    
    async def transcribe_async(self, audio: AudioChunk) -> TranscriptionResult:
        """Async wrapper around sync transcription."""
        import asyncio
        
        # Run in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.transcribe, audio)
    
    def transcribe_streaming(
        self,
        audio_chunks: list[AudioChunk],
        overlap_ms: int = 200,
    ) -> list[TranscriptionResult]:
        """
        Transcribe multiple audio chunks with overlap handling.
        
        For now, this just transcribes each chunk independently.
        A more sophisticated implementation would handle word boundaries.
        """
        results = []
        cumulative_time_ms = 0
        
        for chunk in audio_chunks:
            result = self.transcribe(chunk)
            
            # Adjust timestamps
            chunk_duration = len(chunk) / 32.0
            adjusted = TranscriptionResult(
                text=result.text,
                is_final=True,
                confidence=result.confidence,
                start_time_ms=int(cumulative_time_ms),
                end_time_ms=int(cumulative_time_ms + chunk_duration),
                language=result.language,
            )
            results.append(adjusted)
            cumulative_time_ms += chunk_duration - overlap_ms
        
        return results
    
    def warmup(self) -> None:
        """Warm up the model with a small inference."""
        import numpy as np
        
        logger.info("Warming up Whisper model...")
        
        # Create 1 second of silence
        dummy_audio = np.zeros(16000, dtype=np.float32)
        _ = self._model.transcribe(dummy_audio, language=self._language, verbose=False)
        
        logger.info("Whisper warmup complete")
