"""
CrisisTriage AI - Prosody Feature Extraction Service

Extracts acoustic/prosodic features from audio that correlate with
emotional state and distress levels.

Architecture:
    - Protocol defines the interface for prosody extraction
    - DummyProsodyExtractor: Placeholder for development/testing
    - LibrosaProsodyExtractor: Production implementation (TODO)

Feature Categories:
    1. Pitch (F0) features: Mean, std, range, contour shape
    2. Energy features: Loudness, dynamics, variation
    3. Temporal features: Speech rate, pause patterns
    4. Voice quality: Jitter, shimmer (perturbation measures)

Research Background:
    - Higher pitch variability often correlates with emotional arousal
    - Longer/more frequent pauses may indicate cognitive load or distress
    - Jitter/shimmer increase under stress (voice instability)
    - Speech rate changes can indicate anxiety (faster) or depression (slower)
    
References:
    - Cummins et al. (2015) "A review of depression and suicide risk assessment using speech analysis"
    - Scherer (2003) "Vocal communication of emotion: A review of research paradigms"
    
Privacy Considerations:
    - Prosody features are DERIVED data, less sensitive than raw audio
    - However, voice biometrics could potentially be reconstructed from detailed prosody
    - For maximum privacy, use aggregate statistics only
"""

from __future__ import annotations

import hashlib
import logging
import struct
from abc import abstractmethod
from typing import Protocol, Optional, runtime_checkable

from app.core.types import ProsodyFeatures, AudioChunk

logger = logging.getLogger(__name__)


# =============================================================================
# Protocol (Interface)
# =============================================================================

@runtime_checkable
class ProsodyExtractor(Protocol):
    """
    Protocol for prosody/acoustic feature extraction.
    
    Implementations should extract paralinguistic features that capture
    how something is said, independent of what is said.
    """

    @abstractmethod
    def extract(self, audio: AudioChunk) -> ProsodyFeatures:
        """
        Extract prosody features from audio.
        
        Args:
            audio: Raw audio bytes (PCM 16-bit, 16kHz, mono)
            
        Returns:
            ProsodyFeatures dataclass with extracted features
            
        Note:
            Returns features with None values for any that could not
            be extracted (e.g., pure silence).
        """
        ...

    @abstractmethod
    async def extract_async(self, audio: AudioChunk) -> ProsodyFeatures:
        """Async version for non-blocking extraction."""
        ...

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """Return list of feature names this extractor produces."""
        ...

    @property
    @abstractmethod
    def extractor_id(self) -> str:
        """Return identifier for logging/tracking."""
        ...


# =============================================================================
# Exceptions
# =============================================================================

class ProsodyExtractionError(Exception):
    """Raised when prosody extraction fails."""
    pass


# =============================================================================
# Dummy Implementation (Development/Testing)
# =============================================================================

class DummyProsodyExtractor:
    """
    Placeholder prosody extractor for development and testing.
    
    Generates deterministic dummy features based on audio characteristics.
    The features are realistic in range but not actually extracted from audio.
    
    WARNING: This is NOT suitable for production use.
    """

    FEATURE_NAMES = [
        "speech_rate",
        "pitch_mean",
        "pitch_std",
        "pitch_range",
        "energy_mean",
        "energy_std",
        "pause_ratio",
        "pause_count",
        "jitter",
        "shimmer",
        "voice_quality_score",
        "duration_seconds",
    ]

    def __init__(self, simulated_latency_ms: float = 20.0):
        """
        Initialize dummy prosody extractor.
        
        Args:
            simulated_latency_ms: Artificial delay to simulate processing time
        """
        self._simulated_latency_ms = simulated_latency_ms
        self._call_count = 0

    @property
    def feature_names(self) -> list[str]:
        return self.FEATURE_NAMES

    @property
    def extractor_id(self) -> str:
        return "dummy-prosody-v0.0.1"

    def extract(self, audio: AudioChunk) -> ProsodyFeatures:
        """
        Generate dummy prosody features based on audio hash.
        
        Features are deterministic for a given audio input to enable
        reproducible testing.
        """
        self._call_count += 1
        
        # Use audio hash for deterministic pseudo-random values
        audio_hash = hashlib.md5(audio).hexdigest()
        
        def hash_to_float(offset: int, min_val: float, max_val: float) -> float:
            """Extract a float from hash at given offset."""
            hex_slice = audio_hash[offset:offset + 4]
            normalized = int(hex_slice, 16) / 0xFFFF
            return min_val + normalized * (max_val - min_val)
        
        # Calculate duration from audio length
        duration_seconds = len(audio) / 32000.0  # 16kHz, 16-bit = 32000 bytes/sec
        
        # Generate realistic feature values
        features = ProsodyFeatures(
            speech_rate=hash_to_float(0, 2.0, 7.0),       # syllables/sec
            pitch_mean=hash_to_float(2, 100.0, 200.0),    # Hz (adult range)
            pitch_std=hash_to_float(4, 10.0, 50.0),       # Hz
            pitch_range=hash_to_float(6, 30.0, 150.0),    # Hz
            energy_mean=hash_to_float(8, 0.1, 0.8),       # normalized
            energy_std=hash_to_float(10, 0.05, 0.3),      # normalized
            pause_ratio=hash_to_float(12, 0.1, 0.4),      # fraction
            pause_count=int(hash_to_float(14, 1, 10)),    # count
            jitter=hash_to_float(16, 0.005, 0.03),        # typical range
            shimmer=hash_to_float(18, 0.02, 0.1),         # typical range
            voice_quality_score=hash_to_float(20, 0.5, 1.0),
            duration_seconds=duration_seconds,
        )
        
        logger.debug(
            "DummyProsody: processed %d bytes (%.2fs), call #%d",
            len(audio),
            duration_seconds,
            self._call_count,
        )
        
        return features

    async def extract_async(self, audio: AudioChunk) -> ProsodyFeatures:
        """Async wrapper around sync extraction."""
        import asyncio
        
        # Simulate processing latency
        await asyncio.sleep(self._simulated_latency_ms / 1000.0)
        
        return self.extract(audio)

    def _compute_basic_energy(self, audio: AudioChunk) -> Optional[float]:
        """
        Compute actual RMS energy from audio bytes.
        
        This is a simple real computation included as an example of
        what real extraction would do. Even the dummy extractor can
        compute this to provide some signal.
        """
        if len(audio) < 2:
            return None
            
        try:
            # Unpack 16-bit samples
            n_samples = len(audio) // 2
            samples = struct.unpack(f"<{n_samples}h", audio[:n_samples * 2])
            
            # Compute RMS
            sum_squares = sum(s * s for s in samples)
            rms = (sum_squares / n_samples) ** 0.5
            
            # Normalize to 0-1 range (max int16 is 32767)
            return rms / 32767.0
            
        except Exception as e:
            logger.warning("Failed to compute basic energy: %s", e)
            return None


# =============================================================================
# Librosa-based Prosody Extractor
# =============================================================================

class LibrosaProsodyExtractor:
    """
    Production prosody extractor using librosa.
    
    Extracts acoustic features that correlate with emotional state:
    - Pitch (F0): mean, std, range
    - Energy: RMS mean and std
    - Temporal: speech rate estimate, pause ratio
    - Voice quality: jitter, shimmer approximations
    
    All processing is done LOCALLY - no audio is sent to external APIs.
    
    IMPORTANT SAFETY NOTICE:
        This service is for RESEARCH AND SIMULATION ONLY.
        Not a medical device. Not suitable for real crisis intervention.
    
    Privacy Notes:
        - Audio is processed in-memory only
        - Features are aggregate statistics, not raw audio
        - Feature vectors are not stored unless explicitly configured
    
    Audio Format:
        Expects PCM 16-bit, 16kHz, mono audio bytes.
    """
    
    FEATURE_NAMES = [
        "speech_rate",
        "pitch_mean",
        "pitch_std",
        "pitch_range",
        "energy_mean",
        "energy_std",
        "pause_ratio",
        "pause_count",
        "jitter",
        "shimmer",
        "voice_quality_score",
        "duration_seconds",
    ]
    
    SAMPLE_RATE = 16000  # Expected sample rate
    
    def __init__(
        self,
        store_prosody_features: bool = False,
        anonymize_logs: bool = True,
    ):
        """
        Initialize librosa-based prosody extractor.
        
        Args:
            store_prosody_features: Whether features may be logged/stored
            anonymize_logs: Whether to minimize identifying info in logs
        """
        self._store_features = store_prosody_features
        self._anonymize_logs = anonymize_logs
        self._call_count = 0
        
        # Lazy load librosa
        self._librosa = None
        self._np = None
        self._load_dependencies()
    
    def _load_dependencies(self) -> None:
        """Load librosa and numpy."""
        try:
            import librosa
            import numpy as np
            
            self._librosa = librosa
            self._np = np
            
            logger.info("LibrosaProsodyExtractor initialized")
            
        except ImportError as e:
            raise ImportError(
                "LibrosaProsodyExtractor requires 'librosa' and 'numpy'. "
                "Install with: pip install librosa numpy"
            ) from e
    
    @property
    def feature_names(self) -> list[str]:
        return self.FEATURE_NAMES
    
    @property
    def extractor_id(self) -> str:
        return "librosa-prosody-v1.0"
    
    def extract(self, audio: AudioChunk) -> ProsodyFeatures:
        """
        Extract prosody features from audio.
        
        Args:
            audio: Raw audio bytes (PCM 16-bit, 16kHz, mono)
            
        Returns:
            ProsodyFeatures with extracted acoustic features
        """
        import time
        
        self._call_count += 1
        start_time = time.time()
        
        # Calculate duration from byte length
        duration_seconds = len(audio) / (self.SAMPLE_RATE * 2)  # 2 bytes per sample
        
        # Validate audio length
        if len(audio) < 1600:  # Less than 0.05 seconds
            logger.warning(
                "Audio too short for prosody extraction: %d bytes (%.3fs)",
                len(audio), duration_seconds
            )
            return self._default_features(duration_seconds)
        
        try:
            # Convert bytes to numpy array
            audio_np = self._np.frombuffer(audio, dtype=self._np.int16).astype(self._np.float32) / 32768.0
            
            # Extract features
            pitch_features = self._extract_pitch(audio_np)
            energy_features = self._extract_energy(audio_np)
            temporal_features = self._extract_temporal(audio_np, pitch_features)
            quality_features = self._extract_voice_quality(audio_np, pitch_features)
            
            processing_time = (time.time() - start_time) * 1000
            
            features = ProsodyFeatures(
                speech_rate=temporal_features.get("speech_rate"),
                pitch_mean=pitch_features.get("pitch_mean"),
                pitch_std=pitch_features.get("pitch_std"),
                pitch_range=pitch_features.get("pitch_range"),
                energy_mean=energy_features.get("energy_mean"),
                energy_std=energy_features.get("energy_std"),
                pause_ratio=temporal_features.get("pause_ratio"),
                pause_count=temporal_features.get("pause_count"),
                jitter=quality_features.get("jitter"),
                shimmer=quality_features.get("shimmer"),
                voice_quality_score=quality_features.get("voice_quality_score"),
                duration_seconds=duration_seconds,
            )
            
            logger.debug(
                "LibrosaProsody: %.2fs audio, pitch=%.0fHz, energy=%.3f (%.0fms)",
                duration_seconds,
                features.pitch_mean or 0,
                features.energy_mean or 0,
                processing_time
            )
            
            return features
            
        except Exception as e:
            logger.error("Prosody extraction failed: %s", str(e))
            return self._default_features(duration_seconds)
    
    def _extract_pitch(self, audio: "np.ndarray") -> dict:
        """
        Extract pitch (F0) features using librosa pyin.
        
        Returns:
            Dict with pitch_mean, pitch_std, pitch_range
        """
        try:
            # Use pyin for robust pitch tracking
            f0, voiced_flag, voiced_probs = self._librosa.pyin(
                audio,
                fmin=self._librosa.note_to_hz('C2'),   # ~65 Hz
                fmax=self._librosa.note_to_hz('C7'),   # ~2093 Hz
                sr=self.SAMPLE_RATE,
                frame_length=2048,
                hop_length=512,
            )
            
            # Filter to voiced frames only
            f0_voiced = f0[~self._np.isnan(f0)]
            
            if len(f0_voiced) < 3:
                return {"pitch_mean": None, "pitch_std": None, "pitch_range": None}
            
            return {
                "pitch_mean": float(self._np.mean(f0_voiced)),
                "pitch_std": float(self._np.std(f0_voiced)),
                "pitch_range": float(self._np.max(f0_voiced) - self._np.min(f0_voiced)),
            }
            
        except Exception as e:
            logger.debug("Pitch extraction failed: %s", str(e))
            return {"pitch_mean": None, "pitch_std": None, "pitch_range": None}
    
    def _extract_energy(self, audio: "np.ndarray") -> dict:
        """
        Extract energy (RMS) features.
        
        Returns:
            Dict with energy_mean, energy_std
        """
        try:
            # Compute RMS energy per frame
            rms = self._librosa.feature.rms(
                y=audio,
                frame_length=2048,
                hop_length=512,
            )[0]
            
            if len(rms) < 1:
                return {"energy_mean": None, "energy_std": None}
            
            return {
                "energy_mean": float(self._np.mean(rms)),
                "energy_std": float(self._np.std(rms)),
            }
            
        except Exception as e:
            logger.debug("Energy extraction failed: %s", str(e))
            return {"energy_mean": None, "energy_std": None}
    
    def _extract_temporal(self, audio: "np.ndarray", pitch_features: dict) -> dict:
        """
        Extract temporal features (speech rate, pauses).
        
        Returns:
            Dict with speech_rate, pause_ratio, pause_count
        """
        try:
            # Get RMS energy for pause detection
            rms = self._librosa.feature.rms(
                y=audio,
                frame_length=2048,
                hop_length=512,
            )[0]
            
            if len(rms) < 1:
                return {"speech_rate": None, "pause_ratio": None, "pause_count": 0}
            
            # Calculate frame duration
            hop_duration = 512 / self.SAMPLE_RATE
            total_duration = len(audio) / self.SAMPLE_RATE
            
            # Detect pauses: frames with very low energy
            energy_threshold = self._np.max(rms) * 0.1  # 10% of max
            is_pause = rms < energy_threshold
            
            pause_ratio = float(self._np.mean(is_pause))
            
            # Count pause segments (transitions from speech to pause)
            pause_starts = self._np.diff(is_pause.astype(int))
            pause_count = int(self._np.sum(pause_starts == 1))
            
            # Estimate speech rate: voiced frames / speaking time
            # This is a rough approximation
            speaking_duration = total_duration * (1 - pause_ratio)
            if speaking_duration > 0.1:
                # Rough estimate: ~4-5 syllables per second is normal
                # We use energy peaks as a proxy for syllables
                peaks = self._librosa.util.peak_pick(
                    rms,
                    pre_max=3, post_max=3,
                    pre_avg=3, post_avg=3,
                    delta=0.01, wait=5
                )
                syllable_estimate = len(peaks)
                speech_rate = float(syllable_estimate / speaking_duration)
                # Clamp to reasonable range
                speech_rate = max(1.0, min(10.0, speech_rate))
            else:
                speech_rate = None
            
            return {
                "speech_rate": speech_rate,
                "pause_ratio": pause_ratio,
                "pause_count": pause_count,
            }
            
        except Exception as e:
            logger.debug("Temporal extraction failed: %s", str(e))
            return {"speech_rate": None, "pause_ratio": None, "pause_count": 0}
    
    def _extract_voice_quality(self, audio: "np.ndarray", pitch_features: dict) -> dict:
        """
        Extract voice quality features (jitter, shimmer approximations).
        
        Note: True jitter/shimmer require period-by-period analysis.
        This provides rough approximations using frame-level variation.
        
        Returns:
            Dict with jitter, shimmer, voice_quality_score
        """
        try:
            # Jitter: relative variation in pitch period
            # We approximate using pitch std / mean
            pitch_mean = pitch_features.get("pitch_mean")
            pitch_std = pitch_features.get("pitch_std")
            
            if pitch_mean and pitch_std and pitch_mean > 0:
                # Coefficient of variation as jitter proxy
                jitter = float(pitch_std / pitch_mean)
                # Clamp to typical range
                jitter = max(0.001, min(0.1, jitter))
            else:
                jitter = None
            
            # Shimmer: relative variation in amplitude
            # We use RMS energy variation
            rms = self._librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
            
            if len(rms) > 1 and self._np.mean(rms) > 0:
                shimmer = float(self._np.std(rms) / self._np.mean(rms))
                shimmer = max(0.01, min(0.3, shimmer))
            else:
                shimmer = None
            
            # Voice quality score: inverse of jitter + shimmer (higher = better quality)
            if jitter is not None and shimmer is not None:
                # Normalize and invert
                quality_score = 1.0 - (jitter + shimmer) / 0.4
                quality_score = max(0.0, min(1.0, quality_score))
            else:
                quality_score = None
            
            return {
                "jitter": jitter,
                "shimmer": shimmer,
                "voice_quality_score": quality_score,
            }
            
        except Exception as e:
            logger.debug("Voice quality extraction failed: %s", str(e))
            return {"jitter": None, "shimmer": None, "voice_quality_score": None}
    
    def _default_features(self, duration: float) -> ProsodyFeatures:
        """Return default features when extraction fails."""
        return ProsodyFeatures(
            speech_rate=None,
            pitch_mean=None,
            pitch_std=None,
            pitch_range=None,
            energy_mean=None,
            energy_std=None,
            pause_ratio=None,
            pause_count=0,
            jitter=None,
            shimmer=None,
            voice_quality_score=None,
            duration_seconds=duration,
        )
    
    async def extract_async(self, audio: AudioChunk) -> ProsodyFeatures:
        """Async wrapper around sync extraction."""
        import asyncio
        
        # Run in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.extract, audio)
