"""
CrisisTriage AI - Telephony WebSocket Handler

Handles real-time audio streaming from telephony providers.
Processes audio through the triage pipeline and returns results.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from app.config import Settings
from app.core.logging import call_id_var, session_id_var
from app.core.types import InputModality
from .audio_processor import TelephonyAudioProcessor
from .models import CallStatus
from .session_store import CallSessionStore

logger = logging.getLogger(__name__)


async def telephony_websocket_handler(
    websocket: WebSocket,
    call_id: str,
    pipeline,  # TriagePipeline
    call_store: CallSessionStore,
    settings: Settings,
):
    """
    WebSocket handler for telephony media streams.
    
    Protocol:
    1. Provider connects with call_id in path
    2. Provider sends JSON "start" message with stream metadata
    3. Provider sends binary audio frames or JSON with base64 audio
    4. Provider sends JSON "stop" message when call ends
    5. Server processes audio through triage pipeline
    6. Server may send JSON events back (optional)
    
    Error Handling:
    - Malformed frames are logged and skipped
    - Connection errors trigger session cleanup
    - Backpressure drops old audio if overwhelmed
    
    Privacy:
    - Audio is never persisted
    - call_id is masked in logs
    - Phone numbers are never logged
    """
    # Check telephony enabled
    if not settings.enable_telephony_integration:
        await websocket.close(code=4503, reason="Telephony disabled")
        return
    
    # Get call session
    session = await call_store.get_session(call_id)
    if not session:
        logger.warning("Call not found for WebSocket: ***%s", call_id[-4:])
        await websocket.close(code=4404, reason="Call not found")
        return
    
    # Set context for logging
    call_id_var.set(call_id)
    session_id_var.set(session.session_id)
    
    # Accept connection
    await websocket.accept()
    
    logger.info(
        "Telephony WebSocket connected: call=***%s, session=%s",
        call_id[-4:],
        session.session_id[:8],
    )
    
    # Initialize audio processor
    audio_processor = TelephonyAudioProcessor(
        target_sample_rate=settings.telephony_audio_sample_rate,
        chunk_threshold_bytes=settings.audio_chunk_min_bytes,
    )
    
    # Update call status
    await call_store.update_status(call_id, CallStatus.IN_PROGRESS)
    
    # Track stream state
    stream_started = False
    
    try:
        while True:
            try:
                # Receive with timeout for heartbeat
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                if websocket.client_state == WebSocketState.CONNECTED:
                    try:
                        await websocket.send_json({"event": "ping"})
                    except Exception:
                        break
                continue
            
            if message["type"] == "websocket.disconnect":
                break
            
            # Handle different message types
            if "bytes" in message:
                # Binary audio frame
                await _process_audio_frame(
                    frame=message["bytes"],
                    audio_processor=audio_processor,
                    pipeline=pipeline,
                    session=session,
                    call_store=call_store,
                    websocket=websocket,
                )
            
            elif "text" in message:
                # JSON control message
                try:
                    data = json.loads(message["text"])
                    stream_started = await _handle_control_message(
                        data=data,
                        audio_processor=audio_processor,
                        pipeline=pipeline,
                        session=session,
                        call_store=call_store,
                        websocket=websocket,
                        stream_started=stream_started,
                    )
                except json.JSONDecodeError:
                    logger.warning(
                        "Invalid JSON in telephony WS: %s",
                        message["text"][:50] if len(message["text"]) > 50 else message["text"],
                    )
    
    except WebSocketDisconnect:
        logger.info(
            "Telephony WebSocket disconnected: call=***%s",
            call_id[-4:],
        )
    
    except Exception as e:
        logger.error(
            "Telephony WebSocket error: %s",
            str(e),
            exc_info=True,
        )
    
    finally:
        # Cleanup
        await _cleanup_call_stream(
            call_id=call_id,
            audio_processor=audio_processor,
            pipeline=pipeline,
            session=session,
            call_store=call_store,
        )


async def _process_audio_frame(
    frame: bytes,
    audio_processor: TelephonyAudioProcessor,
    pipeline,
    session,
    call_store: CallSessionStore,
    websocket: WebSocket,
) -> None:
    """Process a binary audio frame."""
    # Record bytes received
    await call_store.record_audio_bytes(session.call_id, len(frame))
    
    # Process through audio processor
    pcm_chunk = audio_processor.process_frame(frame)
    
    if pcm_chunk:
        await _send_to_pipeline(
            pcm_chunk=pcm_chunk,
            pipeline=pipeline,
            session=session,
            call_store=call_store,
            websocket=websocket,
        )


async def _send_to_pipeline(
    pcm_chunk: bytes,
    pipeline,
    session,
    call_store: CallSessionStore,
    websocket: WebSocket,
) -> None:
    """Send audio chunk to pipeline and handle result."""
    try:
        result = await pipeline.process_audio_chunk(
            session_id=session.session_id,
            audio_bytes=pcm_chunk,
            modality=InputModality.PHONE_CALL,
        )
        
        if result:
            # Record triage event
            await call_store.record_triage_event(
                session.call_id,
                result.risk_level.value,
            )
            
            # Send result to provider (optional)
            try:
                await websocket.send_json({
                    "event": "triage",
                    "session_id": session.session_id[:8],
                    "risk_level": result.risk_level.value,
                    "urgency_score": result.urgency_score,
                    "emotional_state": result.emotional_state.value if result.emotional_state else None,
                    "transcript_preview": (
                        result.transcript[:100] + "..."
                        if result.transcript and len(result.transcript) > 100
                        else result.transcript
                    ),
                })
            except Exception as e:
                logger.debug("Could not send triage result to WS: %s", str(e))
    
    except Exception as e:
        logger.error("Pipeline error during phone call: %s", str(e))


async def _handle_control_message(
    data: dict,
    audio_processor: TelephonyAudioProcessor,
    pipeline,
    session,
    call_store: CallSessionStore,
    websocket: WebSocket,
    stream_started: bool,
) -> bool:
    """
    Handle JSON control messages.
    
    Returns updated stream_started state.
    """
    event = data.get("event")
    
    if event == "connected":
        # Initial connection message (Twilio)
        logger.info("Telephony stream connected")
        return stream_started
    
    elif event == "start":
        # Stream starting - configure audio processor
        stream_sid = data.get("streamSid", "")
        
        # Twilio sends media format info in start message
        start_data = data.get("start", {})
        if start_data:
            audio_processor.configure_source(
                sample_rate=int(start_data.get("mediaFormat", {}).get("sampleRate", 8000)),
                channels=int(start_data.get("mediaFormat", {}).get("channels", 1)),
                encoding=start_data.get("mediaFormat", {}).get("encoding", "audio/x-mulaw"),
            )
        
        logger.info(
            "Telephony stream started: stream=%s",
            stream_sid[-8:] if stream_sid else "unknown",
        )
        return True
    
    elif event == "media":
        # Twilio sends audio as base64 in JSON
        media_data = data.get("media", {})
        payload = media_data.get("payload", "")
        
        if payload:
            try:
                audio_bytes = base64.b64decode(payload)
                
                # Record bytes
                await call_store.record_audio_bytes(session.call_id, len(audio_bytes))
                
                # Process
                pcm_chunk = audio_processor.process_frame(audio_bytes)
                
                if pcm_chunk:
                    await _send_to_pipeline(
                        pcm_chunk=pcm_chunk,
                        pipeline=pipeline,
                        session=session,
                        call_store=call_store,
                        websocket=websocket,
                    )
            except Exception as e:
                logger.warning("Error processing media event: %s", str(e))
        
        return stream_started
    
    elif event == "stop":
        logger.info("Telephony stream stop received")
        return stream_started
    
    elif event == "pong":
        # Response to our ping
        pass
    
    else:
        logger.debug("Unknown telephony event: %s", event)
    
    return stream_started


async def _cleanup_call_stream(
    call_id: str,
    audio_processor: TelephonyAudioProcessor,
    pipeline,
    session,
    call_store: CallSessionStore,
) -> None:
    """Cleanup when call stream ends."""
    # Flush remaining audio
    remaining = audio_processor.flush()
    if remaining and len(remaining) > 1000:  # Only process if meaningful
        try:
            await pipeline.process_audio_chunk(
                session_id=session.session_id,
                audio_bytes=remaining,
                modality=InputModality.PHONE_CALL,
            )
        except Exception as e:
            logger.warning("Error flushing final audio: %s", str(e))
    
    # Update call status
    await call_store.update_status(call_id, CallStatus.COMPLETED)
    
    # Log summary
    logger.info(
        "Call stream ended: call=***%s, events=%d, highest_risk=%s, audio_stats=%s",
        call_id[-4:],
        session.triage_events_count,
        session.highest_risk_level or "none",
        audio_processor.stats,
    )
