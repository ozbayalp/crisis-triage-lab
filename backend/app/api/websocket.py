"""
CrisisTriage AI - WebSocket Handlers

Real-time bidirectional communication for:
- Receiving audio chunks from client
- Receiving text messages from client
- Streaming triage updates to client
- Streaming transcript updates to client

Architecture:
    All audio/text processing flows through the TriagePipeline, ensuring:
    - Consistent triage logic across REST and WebSocket
    - Privacy policy enforcement
    - Centralized logging and metrics

Protocol:
    Client → Server:
        - Binary frames: Raw audio chunks (PCM 16-bit, 16kHz, mono)
        - Text frames: JSON messages
            {"type": "text", "data": {"text": "message content"}}
            {"type": "control", "action": "pause" | "resume" | "end"}
    
    Server → Client:
        - Text frames: JSON messages
            {"type": "connected", "session_id": "...", "message": "..."}
            {"type": "transcript", "data": {...}}
            {"type": "triage", "data": {...}}
            {"type": "alert", "level": "...", "message": "..."}
            {"type": "status", "message": "..."}
            {"type": "error", "message": "..."}
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Optional
import json
import logging
import time

from app.core.pipeline import TriagePipeline
from app.core.types import TriageResult, RiskLevel
from app.config import Settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])

# Active connections registry
# In production, this would be managed by Redis for multi-worker support
active_connections: Dict[str, WebSocket] = {}

# Session state (in production, use Redis)
session_states: Dict[str, dict] = {}

# Audio buffers per session (for chunking before processing)
audio_buffers: Dict[str, bytearray] = {}

# Track silence duration per session (for end-of-speech detection)
silence_counters: Dict[str, int] = {}


# =============================================================================
# WebSocket Endpoint
# =============================================================================

@router.websocket("/ws/session/{session_id}")
async def session_stream(websocket: WebSocket, session_id: str):
    """
    Main WebSocket endpoint for a triage session.
    
    Handles both audio streaming and text messages, routing all
    processing through the central TriagePipeline.
    """
    # Get pipeline from app state
    pipeline: TriagePipeline = websocket.app.state.pipeline
    settings: Settings = websocket.app.state.settings
    
    await websocket.accept()
    active_connections[session_id] = websocket
    
    # Initialize session state
    session_states[session_id] = {
        "connected_at": time.time(),
        "message_count": 0,
        "audio_bytes_received": 0,
        "audio_chunks_processed": 0,
        "paused": False,
    }
    
    # Initialize audio buffer for this session
    audio_buffers[session_id] = bytearray()
    
    logger.info("WebSocket connected: session=%s", session_id[:8] + "...")
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "message": "Session started. Send audio chunks or text messages to begin triage.",
            "protocol_version": "1.0",
        })
        
        # Main message loop
        while True:
            message = await websocket.receive()
            
            # Check if session is paused
            if session_states.get(session_id, {}).get("paused", False):
                if "text" in message:
                    # Still allow control messages when paused
                    try:
                        parsed = json.loads(message["text"])
                        if parsed.get("type") == "control":
                            await handle_control_message(
                                session_id, parsed, websocket, pipeline, settings
                            )
                            continue
                    except json.JSONDecodeError:
                        pass
                # Skip processing while paused
                continue
            
            if "bytes" in message:
                # Binary frame: audio chunk
                audio_chunk = message["bytes"]
                session_states[session_id]["audio_bytes_received"] += len(audio_chunk)
                session_states[session_id]["message_count"] += 1
                
                await handle_audio_chunk(
                    session_id, audio_chunk, websocket, pipeline, settings
                )
                
            elif "text" in message:
                # Text frame: JSON message
                try:
                    parsed = json.loads(message["text"])
                    msg_type = parsed.get("type", "unknown")
                    
                    if msg_type == "text":
                        # Text message for triage
                        text_data = parsed.get("data", {})
                        text_content = text_data.get("text", "")
                        session_states[session_id]["message_count"] += 1
                        
                        await handle_text_message(
                            session_id, text_content, websocket, pipeline, settings
                        )
                        
                    elif msg_type == "control":
                        # Control message
                        await handle_control_message(
                            session_id, parsed, websocket, pipeline, settings
                        )
                        
                    else:
                        logger.warning(
                            "Unknown message type: %s (session=%s)",
                            msg_type, session_id[:8]
                        )
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Unknown message type: {msg_type}",
                        })
                        
                except json.JSONDecodeError as e:
                    logger.warning(
                        "Invalid JSON in WebSocket message (session=%s): %s",
                        session_id[:8], str(e)
                    )
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON format",
                    })
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: session=%s", session_id[:8] + "...")
    except Exception as e:
        logger.error(
            "WebSocket error (session=%s): %s",
            session_id[:8], str(e),
            exc_info=True
        )
    finally:
        # Clean up
        await cleanup_session(session_id, settings)
        active_connections.pop(session_id, None)
        session_states.pop(session_id, None)
        audio_buffers.pop(session_id, None)
        silence_counters.pop(session_id, None)


# =============================================================================
# Message Handlers
# =============================================================================

def detect_silence(audio_bytes: bytes, threshold: float = 0.02) -> bool:
    """
    Detect if the audio chunk is mostly silence.
    
    Args:
        audio_bytes: PCM 16-bit audio bytes
        threshold: RMS threshold below which is considered silence (0-1 scale)
    
    Returns:
        True if the audio is mostly silence
    """
    import struct
    
    if len(audio_bytes) < 2:
        return True
    
    # Convert bytes to 16-bit samples
    num_samples = len(audio_bytes) // 2
    samples = struct.unpack(f'<{num_samples}h', audio_bytes[:num_samples * 2])
    
    # Calculate RMS
    sum_squares = sum(s * s for s in samples)
    rms = (sum_squares / num_samples) ** 0.5
    
    # Normalize to 0-1 scale (max 16-bit value is 32767)
    normalized_rms = rms / 32767.0
    
    return normalized_rms < threshold


async def handle_audio_chunk(
    session_id: str,
    audio_chunk: bytes,
    websocket: WebSocket,
    pipeline: TriagePipeline,
    settings: Settings,
):
    """
    Process an incoming audio chunk through the triage pipeline.
    
    Audio is buffered until EITHER:
    1. We have enough audio (~5 seconds minimum)
    2. User stops speaking (silence detected after speech)
    
    This ensures we get meaningful paragraphs for triage, not single words.
    
    Audio Format Expected:
        PCM 16-bit, 16kHz, mono (32000 bytes per second)
    """
    # Get buffer for this session
    buffer = audio_buffers.get(session_id)
    if buffer is None:
        buffer = bytearray()
        audio_buffers[session_id] = buffer
    
    # Append new audio to buffer
    buffer.extend(audio_chunk)
    
    # --- Smart buffering with silence detection ---
    # Minimum: 5 seconds of audio (160000 bytes at 16kHz 16-bit mono)
    # Maximum: 15 seconds of audio (480000 bytes)
    min_chunk_size = getattr(settings, 'audio_chunk_min_bytes', 160000)  # ~5 seconds
    max_buffer_size = getattr(settings, 'audio_buffer_max_bytes', 480000)  # ~15 seconds
    
    # Silence detection parameters
    silence_threshold = 0.015  # RMS threshold for silence
    silence_chunks_required = 3  # Number of consecutive silent chunks to trigger processing
    
    # Check for silence in the latest chunk
    is_silent = detect_silence(audio_chunk, silence_threshold)
    
    # Track consecutive silence
    if session_id not in silence_counters:
        silence_counters[session_id] = 0
    
    if is_silent:
        silence_counters[session_id] += 1
    else:
        silence_counters[session_id] = 0
    
    # Prevent buffer from growing too large - force process if at max
    if len(buffer) >= max_buffer_size:
        logger.info(
            "Audio buffer at max size (%d bytes), processing (session=%s)",
            len(buffer), session_id[:8]
        )
        # Process the buffer (will happen below)
    
    # Determine if we should process now
    should_process = False
    reason = ""
    
    # Case 1: Buffer at max size - must process
    if len(buffer) >= max_buffer_size:
        should_process = True
        reason = "max_buffer"
    
    # Case 2: Have minimum audio AND detected end of speech (silence)
    elif len(buffer) >= min_chunk_size and silence_counters[session_id] >= silence_chunks_required:
        should_process = True
        reason = "silence_detected"
        logger.debug(
            "End of speech detected (session=%s, buffer=%d bytes)",
            session_id[:8], len(buffer)
        )
    
    # Case 3: Not enough audio yet - keep buffering
    if not should_process:
        if len(buffer) % 50000 == 0:  # Log every ~1.5 seconds
            logger.debug(
                "Buffering audio: %d/%d bytes, silence_count=%d (session=%s)",
                len(buffer), min_chunk_size, silence_counters[session_id], session_id[:8]
            )
        return
    
    # Reset silence counter
    silence_counters[session_id] = 0
    
    # Extract the chunk to process
    chunk_to_process = bytes(buffer)
    
    # Clear the buffer completely for fresh start
    buffer.clear()
    
    try:
        # Update stats
        state = session_states.get(session_id, {})
        state["audio_chunks_processed"] = state.get("audio_chunks_processed", 0) + 1
        
        logger.debug(
            "Processing audio chunk: %d bytes (session=%s, chunk #%d)",
            len(chunk_to_process),
            session_id[:8],
            state.get("audio_chunks_processed", 0)
        )
        
        # Process through pipeline
        result = await pipeline.process_audio_chunk(
            session_id=session_id,
            audio_bytes=chunk_to_process,
            source="websocket",
        )
        
        # Send transcript update (if we have transcription)
        if result.transcript_segment:
            await websocket.send_json({
                "type": "transcript",
                "data": {
                    "text": result.transcript_segment,
                    "is_final": True,
                    "timestamp_ms": result.timestamp_ms,
                },
            })
        
        # Send triage update
        await send_triage_update(websocket, result)
        
        # Check for high-risk alerts
        await check_and_send_alerts(websocket, result, settings)
        
    except Exception as e:
        logger.error(
            "Audio processing error (session=%s): %s",
            session_id[:8], str(e),
            exc_info=True
        )
        await websocket.send_json({
            "type": "error",
            "message": "Audio processing failed",
        })


async def handle_text_message(
    session_id: str,
    text: str,
    websocket: WebSocket,
    pipeline: TriagePipeline,
    settings: Settings,
):
    """
    Process a text message through the triage pipeline.
    
    Used for text chat modality or testing without audio.
    """
    if not text or not text.strip():
        await websocket.send_json({
            "type": "error",
            "message": "Empty text message",
        })
        return
    
    try:
        # Process through pipeline
        result = await pipeline.process_text_message(
            session_id=session_id,
            message=text,
            source="websocket",
        )
        
        # Send triage update
        await send_triage_update(websocket, result)
        
        # Check for high-risk alerts
        await check_and_send_alerts(websocket, result, settings)
        
    except Exception as e:
        logger.error(
            "Text processing error (session=%s): %s",
            session_id[:8], str(e),
            exc_info=True
        )
        await websocket.send_json({
            "type": "error",
            "message": "Text processing failed",
        })


async def handle_control_message(
    session_id: str,
    control: dict,
    websocket: WebSocket,
    pipeline: TriagePipeline,
    settings: Settings,
):
    """Handle control messages from client."""
    action = control.get("action")
    
    if action == "pause":
        session_states[session_id]["paused"] = True
        logger.info("Session paused: %s", session_id[:8])
        await websocket.send_json({
            "type": "status",
            "message": "Session paused",
        })
        
    elif action == "resume":
        session_states[session_id]["paused"] = False
        logger.info("Session resumed: %s", session_id[:8])
        await websocket.send_json({
            "type": "status",
            "message": "Session resumed",
        })
        
    elif action == "end":
        logger.info("Session end requested: %s", session_id[:8])
        await websocket.send_json({
            "type": "status",
            "message": "Session ending",
        })
        await cleanup_session(session_id, settings)
        await websocket.close()
        
    else:
        logger.warning("Unknown control action: %s (session=%s)", action, session_id[:8])
        await websocket.send_json({
            "type": "error",
            "message": f"Unknown control action: {action}",
        })


# =============================================================================
# Response Helpers
# =============================================================================

async def send_triage_update(websocket: WebSocket, result: TriageResult):
    """Send a triage update message to the client."""
    await websocket.send_json({
        "type": "triage",
        "data": {
            "emotional_state": result.emotional_state.value,
            "risk_level": result.risk_level.value,
            "urgency_score": result.urgency_score,
            "recommended_action": result.recommended_action.value,
            "confidence": result.confidence,
            "explanation": result.explanation.to_dict(),
            "timestamp_ms": result.timestamp_ms,
            "processing_time_ms": result.processing_time_ms,
        },
    })


async def check_and_send_alerts(
    websocket: WebSocket,
    result: TriageResult,
    settings: Settings,
):
    """
    Check if result warrants an alert and send if needed.
    
    Alerts are sent for:
    - HIGH risk: warning alert
    - IMMINENT risk: critical alert
    """
    if result.risk_level == RiskLevel.IMMINENT:
        await websocket.send_json({
            "type": "alert",
            "level": "critical",
            "message": "IMMEDIATE ATTENTION REQUIRED: Imminent risk detected",
            "timestamp_ms": result.timestamp_ms,
        })
        logger.warning(
            "CRITICAL ALERT sent: session=%s, urgency=%d",
            result.session_id[:8], result.urgency_score
        )
        
    elif result.risk_level == RiskLevel.HIGH:
        await websocket.send_json({
            "type": "alert",
            "level": "high",
            "message": "High risk detected - consider escalation",
            "timestamp_ms": result.timestamp_ms,
        })
        logger.warning(
            "HIGH RISK ALERT sent: session=%s, urgency=%d",
            result.session_id[:8], result.urgency_score
        )


# =============================================================================
# Cleanup
# =============================================================================

async def cleanup_session(session_id: str, settings: Settings):
    """
    Clean up session resources.
    
    Applies privacy policies based on settings:
    - Audio buffers: Always deleted (ephemeral)
    - Transcripts: Deleted unless persist_transcripts=True
    - Features: Deleted unless persist_features=True
    """
    logger.info("Cleaning up session: %s", session_id[:8] + "...")
    
    state = session_states.get(session_id, {})
    duration = time.time() - state.get("connected_at", time.time())
    
    logger.info(
        "Session stats: duration=%.1fs, messages=%d, audio_bytes=%d",
        duration,
        state.get("message_count", 0),
        state.get("audio_bytes_received", 0),
    )
    
    # TODO: Implement actual cleanup in Redis/storage
    # Privacy-aware cleanup:
    # 
    # if not settings.store_audio:
    #     await audio_buffer_store.delete(session_id)
    #     logger.debug("Deleted audio buffers for session %s", session_id[:8])
    # 
    # if not settings.store_raw_transcripts:
    #     await transcript_store.delete(session_id)
    #     logger.debug("Deleted transcripts for session %s", session_id[:8])
    # 
    # if not settings.store_prosody_features:
    #     await feature_store.delete(session_id)
    #     logger.debug("Deleted prosody features for session %s", session_id[:8])


# =============================================================================
# Utility Functions
# =============================================================================

def get_active_session_count() -> int:
    """Get count of active WebSocket sessions."""
    return len(active_connections)


async def broadcast_to_session(session_id: str, message: dict) -> bool:
    """
    Send a message to a specific session if connected.
    
    Useful for server-initiated messages (e.g., admin alerts).
    Returns True if sent, False if session not found.
    """
    websocket = active_connections.get(session_id)
    if websocket:
        try:
            await websocket.send_json(message)
            return True
        except Exception as e:
            logger.warning("Failed to send to session %s: %s", session_id[:8], e)
    return False
