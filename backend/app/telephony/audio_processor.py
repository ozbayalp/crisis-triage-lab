"""
CrisisTriage AI - Telephony Audio Processor

Handles audio format conversion for telephony streams.
Supports μ-law to PCM conversion and resampling.
"""

from __future__ import annotations

import logging
import struct
from typing import Optional

logger = logging.getLogger(__name__)


# μ-law decoding table
MULAW_DECODE_TABLE = []
for i in range(256):
    byte = ~i
    sign = byte & 0x80
    exponent = (byte >> 4) & 0x07
    mantissa = byte & 0x0F
    
    sample = (mantissa << 3) + 0x84
    sample <<= exponent
    sample -= 0x84
    
    if sign:
        sample = -sample
    
    MULAW_DECODE_TABLE.append(sample)


class TelephonyAudioProcessor:
    """
    Processes audio frames from telephony providers.
    
    Handles:
    - μ-law to PCM conversion (Twilio sends μ-law at 8kHz)
    - Base64 decoding
    - Resampling to target sample rate
    - Buffering for chunk processing
    
    Usage:
        processor = TelephonyAudioProcessor(
            target_sample_rate=16000,
            chunk_threshold_bytes=32000,
        )
        
        processor.configure_source(sample_rate=8000, encoding="audio/x-mulaw")
        
        for frame in audio_frames:
            pcm_chunk = processor.process_frame(frame)
            if pcm_chunk:
                await pipeline.process_audio_chunk(session_id, pcm_chunk)
        
        # Flush remaining audio
        final_chunk = processor.flush()
    """
    
    def __init__(
        self,
        target_sample_rate: int = 16000,
        chunk_threshold_bytes: int = 32000,
        max_buffer_bytes: int = 320000,
    ):
        """
        Initialize audio processor.
        
        Args:
            target_sample_rate: Output sample rate (Hz)
            chunk_threshold_bytes: Buffer size before emitting chunk
            max_buffer_bytes: Maximum buffer size (backpressure)
        """
        self._target_sample_rate = target_sample_rate
        self._chunk_threshold = chunk_threshold_bytes
        self._max_buffer = max_buffer_bytes
        self._buffer = bytearray()
        
        # Source format (configured by provider)
        self._source_sample_rate: int = 8000
        self._source_channels: int = 1
        self._source_encoding: str = "audio/x-mulaw"
        
        # Stats
        self._frames_processed: int = 0
        self._bytes_dropped: int = 0
    
    def configure_source(
        self,
        sample_rate: int = 8000,
        channels: int = 1,
        encoding: str = "audio/x-mulaw",
    ) -> None:
        """
        Configure source audio format.
        
        Args:
            sample_rate: Source sample rate (Hz)
            channels: Number of channels (usually 1)
            encoding: Audio encoding format
        """
        self._source_sample_rate = sample_rate
        self._source_channels = channels
        self._source_encoding = encoding
        
        logger.info(
            "Audio source configured: %dHz, %d ch, %s",
            sample_rate, channels, encoding
        )
    
    def process_frame(
        self,
        frame: bytes,
        encoding: Optional[str] = None,
    ) -> Optional[bytes]:
        """
        Process an incoming audio frame.
        
        Args:
            frame: Raw audio bytes
            encoding: Override encoding (or use configured)
        
        Returns:
            PCM bytes if buffer threshold reached, else None
        """
        if not frame:
            return None
        
        encoding = encoding or self._source_encoding
        
        try:
            # Decode based on encoding
            if encoding in ("mulaw", "audio/x-mulaw", "PCMU"):
                pcm = self._decode_mulaw(frame)
            elif encoding in ("pcm16", "audio/l16", "L16"):
                pcm = frame
            else:
                logger.warning("Unknown audio encoding: %s", encoding)
                return None
            
            # Resample if needed
            if self._source_sample_rate != self._target_sample_rate:
                pcm = self._resample(
                    pcm,
                    self._source_sample_rate,
                    self._target_sample_rate,
                )
            
            # Add to buffer
            self._buffer.extend(pcm)
            self._frames_processed += 1
            
            # Apply backpressure if buffer too large
            if len(self._buffer) > self._max_buffer:
                overflow = len(self._buffer) - self._max_buffer
                self._buffer = self._buffer[overflow:]
                self._bytes_dropped += overflow
                logger.warning(
                    "Audio buffer overflow, dropped %d bytes (total: %d)",
                    overflow, self._bytes_dropped
                )
            
            # Return chunk if threshold reached
            if len(self._buffer) >= self._chunk_threshold:
                chunk = bytes(self._buffer[:self._chunk_threshold])
                self._buffer = self._buffer[self._chunk_threshold:]
                return chunk
            
            return None
            
        except Exception as e:
            logger.error("Audio frame processing error: %s", str(e))
            return None
    
    def flush(self) -> Optional[bytes]:
        """
        Flush remaining buffer.
        
        Returns:
            Remaining audio bytes, or None if empty
        """
        if len(self._buffer) > 0:
            chunk = bytes(self._buffer)
            self._buffer.clear()
            return chunk
        return None
    
    def reset(self) -> None:
        """Reset processor state."""
        self._buffer.clear()
        self._frames_processed = 0
        self._bytes_dropped = 0
    
    @property
    def stats(self) -> dict:
        """Get processing statistics."""
        return {
            "frames_processed": self._frames_processed,
            "bytes_in_buffer": len(self._buffer),
            "bytes_dropped": self._bytes_dropped,
        }
    
    def _decode_mulaw(self, mulaw_bytes: bytes) -> bytes:
        """
        Decode μ-law to 16-bit PCM.
        
        μ-law is a companding algorithm used in telephony.
        8-bit μ-law samples are converted to 16-bit linear PCM.
        """
        pcm_samples = [MULAW_DECODE_TABLE[b] for b in mulaw_bytes]
        return struct.pack(f"<{len(pcm_samples)}h", *pcm_samples)
    
    def _resample(
        self,
        pcm: bytes,
        source_rate: int,
        target_rate: int,
    ) -> bytes:
        """
        Simple linear resampling.
        
        For production, consider using scipy or librosa for better quality.
        """
        if source_rate == target_rate:
            return pcm
        
        # Unpack samples
        num_samples = len(pcm) // 2
        samples = struct.unpack(f"<{num_samples}h", pcm)
        
        # Calculate resampling ratio
        ratio = target_rate / source_rate
        new_length = int(len(samples) * ratio)
        
        # Linear interpolation
        resampled = []
        for i in range(new_length):
            src_idx = i / ratio
            idx_floor = int(src_idx)
            idx_ceil = min(idx_floor + 1, len(samples) - 1)
            frac = src_idx - idx_floor
            
            sample = int(samples[idx_floor] * (1 - frac) + samples[idx_ceil] * frac)
            # Clamp to 16-bit range
            sample = max(-32768, min(32767, sample))
            resampled.append(sample)
        
        return struct.pack(f"<{len(resampled)}h", *resampled)


def decode_base64_audio(payload: str) -> bytes:
    """
    Decode base64-encoded audio.
    
    Used by Twilio media streams which send audio as base64 JSON.
    """
    import base64
    return base64.b64decode(payload)
