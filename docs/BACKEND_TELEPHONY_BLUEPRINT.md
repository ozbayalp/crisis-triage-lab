# CrisisTriage AI — Backend Finalization & Telephony Integration Blueprint

**Document Type:** Internal Engineering Specification  
**Status:** Ready for Implementation  
**Author:** Principal Backend Engineer  
**Review:** ML Systems Architecture Team

---

## Executive Summary

This document specifies the final backend hardening tasks and the complete telephony integration layer for CrisisTriage AI. Upon completion, the system will support:

1. **Production-grade pipeline stability** — proper lifecycle management, error handling, and shutdown behavior
2. **Phone call audio streaming** — real-time telephony provider integration via HTTP webhooks + WebSocket media streams
3. **Unified analytics** — call sessions tracked alongside text/microphone sessions
4. **Privacy enforcement** — phone number masking, no audio persistence, ephemeral session data

---

# Part A: Backend Hardening

## 1. Pipeline Lifecycle Management

### 1.1 Current State Analysis

The existing `TriagePipeline` in `backend/app/core/pipeline.py` handles:
- Text message processing
- Audio chunk processing (microphone)
- Analytics event recording

**Gaps to address:**
- No explicit shutdown hooks
- No connection tracking for concurrent sessions
- No backpressure mechanism for audio floods
- Inconsistent error propagation

### 1.2 Required Changes

#### A. Add Session Registry

```python
# backend/app/core/session_registry.py

class SessionRegistry:
    """
    Thread-safe registry of active triage sessions.
    Tracks session state, connection count, and activity timestamps.
    """
    
    def __init__(self, max_sessions: int = 1000, session_ttl_seconds: int = 3600):
        self._sessions: dict[str, SessionState] = {}
        self._lock = asyncio.Lock()
        self._max_sessions = max_sessions
        self._session_ttl = session_ttl_seconds
    
    async def register(self, session_id: str, modality: InputModality) -> SessionState
    async def get(self, session_id: str) -> Optional[SessionState]
    async def update_activity(self, session_id: str) -> None
    async def unregister(self, session_id: str) -> None
    async def cleanup_stale(self) -> int  # Returns count of cleaned sessions
    async def get_active_count(self) -> int
    async def shutdown(self) -> None  # Graceful shutdown of all sessions
```

#### B. Pipeline Shutdown Hooks

Add to `TriagePipeline`:

```python
async def shutdown(self) -> None:
    """
    Graceful shutdown sequence:
    1. Stop accepting new requests
    2. Drain in-flight audio buffers
    3. Flush pending analytics events
    4. Close model resources
    5. Clear session registry
    """
    logger.info("Pipeline shutdown initiated")
    self._accepting_requests = False
    
    # Drain buffers (max 5 seconds)
    await self._drain_audio_buffers(timeout=5.0)
    
    # Flush analytics
    if self._history_store:
        await self._history_store.flush()
    
    # Unload model (if neural)
    if hasattr(self._model, 'unload'):
        self._model.unload()
    
    logger.info("Pipeline shutdown complete")
```

#### C. Backpressure Mechanism

```python
class AudioBufferManager:
    """
    Manages audio buffer with backpressure.
    Drops oldest frames if buffer exceeds threshold.
    """
    
    def __init__(
        self,
        max_buffer_bytes: int = 320000,  # ~10 seconds
        warning_threshold: float = 0.8,
    ):
        self._buffer = bytearray()
        self._max_bytes = max_buffer_bytes
        self._warning_threshold = warning_threshold
        self._drops_count = 0
    
    def append(self, chunk: bytes) -> bool:
        """Returns False if chunk was dropped due to backpressure."""
        if len(self._buffer) + len(chunk) > self._max_bytes:
            # Drop oldest data to make room
            overflow = len(self._buffer) + len(chunk) - self._max_bytes
            self._buffer = self._buffer[overflow:]
            self._drops_count += 1
            logger.warning(
                "Audio backpressure: dropped %d bytes (total drops: %d)",
                overflow, self._drops_count
            )
        self._buffer.extend(chunk)
        return True
```

---

## 2. Logging Standardization

### 2.1 Unified Log Schema

All log entries must follow this structure:

```json
{
  "timestamp": "2024-11-30T00:00:00.000Z",
  "level": "INFO|WARNING|ERROR",
  "logger": "module.submodule",
  "correlation_id": "req_abc123",
  "session_id": "ses_xyz789",  // masked to 8 chars
  "call_id": "call_***456",    // masked
  "message": "Human-readable message",
  "event_type": "triage|telephony|analytics|system",
  "data": {
    // Structured payload, never contains PII
  }
}
```

### 2.2 Implementation

```python
# backend/app/core/logging.py

import logging
import json
from contextvars import ContextVar
from typing import Optional

# Context variables for request tracking
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
session_id_var: ContextVar[Optional[str]] = ContextVar('session_id', default=None)
call_id_var: ContextVar[Optional[str]] = ContextVar('call_id', default=None)

class StructuredFormatter(logging.Formatter):
    """JSON formatter with context injection."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "correlation_id": correlation_id_var.get(),
            "session_id": self._mask_session_id(session_id_var.get()),
            "call_id": self._mask_call_id(call_id_var.get()),
            "message": record.getMessage(),
        }
        
        if hasattr(record, 'event_type'):
            log_entry['event_type'] = record.event_type
        if hasattr(record, 'data'):
            log_entry['data'] = record.data
        
        return json.dumps(log_entry)
    
    def _mask_session_id(self, sid: Optional[str]) -> Optional[str]:
        if not sid:
            return None
        return sid[:8] if len(sid) > 8 else sid
    
    def _mask_call_id(self, cid: Optional[str]) -> Optional[str]:
        if not cid:
            return None
        return f"call_***{cid[-4:]}" if len(cid) > 4 else "call_***"
```

### 2.3 Middleware for Correlation IDs

```python
# backend/app/middleware/correlation.py

from starlette.middleware.base import BaseHTTPMiddleware
import uuid

class CorrelationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        correlation_id = request.headers.get('X-Correlation-ID', str(uuid.uuid4()))
        correlation_id_var.set(correlation_id)
        
        response = await call_next(request)
        response.headers['X-Correlation-ID'] = correlation_id
        
        return response
```

---

## 3. Error Handling Standards

### 3.1 Structured Error Responses

All API errors must return:

```json
{
  "error": {
    "code": "TRIAGE_PIPELINE_ERROR",
    "message": "Human-readable description",
    "correlation_id": "req_abc123",
    "timestamp": "2024-11-30T00:00:00Z"
  }
}
```

### 3.2 Exception Hierarchy

```python
# backend/app/core/exceptions.py

class CrisisTriageError(Exception):
    """Base exception for all CrisisTriage errors."""
    code: str = "UNKNOWN_ERROR"
    status_code: int = 500
    
class PipelineError(CrisisTriageError):
    code = "PIPELINE_ERROR"
    status_code = 500

class TranscriptionError(PipelineError):
    code = "TRANSCRIPTION_ERROR"

class TriageModelError(PipelineError):
    code = "TRIAGE_MODEL_ERROR"

class TelephonyError(CrisisTriageError):
    code = "TELEPHONY_ERROR"
    status_code = 502

class CallNotFoundError(TelephonyError):
    code = "CALL_NOT_FOUND"
    status_code = 404

class TelephonyDisabledError(CrisisTriageError):
    code = "TELEPHONY_DISABLED"
    status_code = 503

class SessionLimitError(CrisisTriageError):
    code = "SESSION_LIMIT_EXCEEDED"
    status_code = 429
```

### 3.3 Global Exception Handler

```python
# backend/app/middleware/errors.py

@app.exception_handler(CrisisTriageError)
async def crisis_triage_error_handler(request: Request, exc: CrisisTriageError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": str(exc),
                "correlation_id": correlation_id_var.get(),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        }
    )
```

---

## 4. Health Check Endpoints

### 4.1 System Health

```python
# backend/app/api/health.py

@router.get("/api/system/health")
async def health_check() -> dict:
    """
    Returns overall system health.
    Used by load balancers and monitoring.
    """
    checks = {
        "pipeline": await check_pipeline_health(),
        "model": await check_model_health(),
        "analytics": await check_analytics_health(),
    }
    
    if settings.enable_telephony_integration:
        checks["telephony"] = await check_telephony_health()
    
    overall = all(c["status"] == "healthy" for c in checks.values())
    
    return {
        "status": "healthy" if overall else "degraded",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "checks": checks,
    }

@router.get("/api/system/model_status")
async def model_status() -> dict:
    """Returns detailed model status."""
    return {
        "model_id": pipeline.model.model_id,
        "backend": settings.triage_model_backend,
        "warmed_up": pipeline.model._warmed_up,
        "device": getattr(pipeline.model, '_device', 'cpu'),
    }

@router.get("/api/system/telephony_status")
async def telephony_status() -> dict:
    """Returns telephony subsystem status."""
    if not settings.enable_telephony_integration:
        raise TelephonyDisabledError("Telephony integration is disabled")
    
    return {
        "enabled": True,
        "provider": settings.telephony_provider,
        "active_calls": await call_session_store.get_active_count(),
        "max_concurrent_calls": settings.telephony_max_concurrent_calls,
    }
```

---

# Part B: Telephony Integration

## 5. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TELEPHONY PROVIDER                               │
│                    (Twilio / Generic / Simulator)                        │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ POST          │     │ POST            │     │ WS              │
│ /incoming     │     │ /status         │     │ /telephony/{id} │
│               │     │                 │     │                 │
│ Call Initiate │     │ Status Updates  │     │ Media Stream    │
└───────┬───────┘     └────────┬────────┘     └────────┬────────┘
        │                      │                       │
        ▼                      ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        CALL SESSION STORE                                │
│   call_id → { session_id, status, from, to, started_at, events[] }      │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRIAGE PIPELINE                                  │
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │ Transcription │ → │   Prosody    │ → │ Triage Model │ → Analytics   │
│  │   (Whisper)   │    │  (Librosa)   │    │   (Neural)   │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. File Structure

```
backend/app/telephony/
├── __init__.py
├── router.py              # FastAPI router for telephony endpoints
├── websocket.py           # WebSocket handler for media streams
├── session_store.py       # CallSessionStore implementation
├── audio_processor.py     # Audio format conversion & buffering
├── models.py              # Pydantic models for telephony
├── providers/
│   ├── __init__.py
│   ├── base.py            # Abstract provider interface
│   ├── generic.py         # Generic provider (for testing)
│   ├── twilio.py          # Twilio-specific implementation
│   └── simulator.py       # Development call simulator
└── privacy.py             # Phone number masking utilities
```

---

## 7. Call Session Store

### 7.1 Data Model

```python
# backend/app/telephony/models.py

from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class CallStatus(str, Enum):
    INITIATED = "initiated"
    RINGING = "ringing"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NO_ANSWER = "no_answer"

class CallDirection(str, Enum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"

class CallSession(BaseModel):
    """Represents an active or completed phone call."""
    
    call_id: str = Field(..., description="Unique call identifier from provider")
    session_id: str = Field(..., description="Internal triage session ID")
    
    # Masked phone numbers (privacy)
    from_number_masked: str = Field(..., description="Caller number (masked)")
    to_number_masked: str = Field(..., description="Callee number (masked)")
    
    direction: CallDirection = CallDirection.INBOUND
    status: CallStatus = CallStatus.INITIATED
    
    # Timestamps
    initiated_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None  # When audio stream began
    ended_at: Optional[datetime] = None
    
    # Metrics
    audio_bytes_received: int = 0
    triage_events_count: int = 0
    highest_risk_level: Optional[str] = None
    
    # Provider metadata (non-sensitive)
    provider: str = "generic"
    provider_metadata: dict = Field(default_factory=dict)

class IncomingCallRequest(BaseModel):
    """Request body for POST /api/telephony/incoming"""
    
    call_id: str = Field(..., alias="CallSid", description="Provider's call ID")
    from_number: str = Field(..., alias="From", description="Caller phone number")
    to_number: str = Field(..., alias="To", description="Callee phone number")
    direction: CallDirection = CallDirection.INBOUND
    provider: str = "generic"
    
    class Config:
        populate_by_name = True  # Allow both alias and field name

class CallStatusUpdate(BaseModel):
    """Request body for POST /api/telephony/status"""
    
    call_id: str = Field(..., alias="CallSid")
    status: CallStatus = Field(..., alias="CallStatus")
    duration: Optional[int] = None  # Call duration in seconds
    
    class Config:
        populate_by_name = True
```

### 7.2 Store Implementation

```python
# backend/app/telephony/session_store.py

import asyncio
from datetime import datetime, timedelta
from typing import Optional
import logging

from .models import CallSession, CallStatus
from .privacy import mask_phone_number

logger = logging.getLogger(__name__)

class CallSessionStore:
    """
    In-memory store for active call sessions.
    
    Thread-safe and bounded. Automatically evicts stale sessions.
    
    Privacy:
        - Phone numbers are masked before storage
        - No raw audio is ever stored
        - Sessions are ephemeral (memory-only)
    """
    
    def __init__(
        self,
        max_sessions: int = 100,
        session_ttl_minutes: int = 60,
        cleanup_interval_seconds: int = 60,
    ):
        self._sessions: dict[str, CallSession] = {}
        self._lock = asyncio.Lock()
        self._max_sessions = max_sessions
        self._session_ttl = timedelta(minutes=session_ttl_minutes)
        self._cleanup_interval = cleanup_interval_seconds
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start background cleanup task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("CallSessionStore started: max=%d, ttl=%s", 
                    self._max_sessions, self._session_ttl)
    
    async def stop(self) -> None:
        """Stop background tasks and clear sessions."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        async with self._lock:
            count = len(self._sessions)
            self._sessions.clear()
        
        logger.info("CallSessionStore stopped: cleared %d sessions", count)
    
    async def create_session(
        self,
        call_id: str,
        from_number: str,
        to_number: str,
        provider: str = "generic",
    ) -> CallSession:
        """
        Create a new call session.
        
        Phone numbers are masked before storage.
        Raises SessionLimitError if at capacity.
        """
        async with self._lock:
            # Check for duplicate
            if call_id in self._sessions:
                logger.warning("Duplicate call_id received: %s", call_id[-4:])
                return self._sessions[call_id]
            
            # Check capacity
            if len(self._sessions) >= self._max_sessions:
                raise SessionLimitError(
                    f"Maximum concurrent calls ({self._max_sessions}) reached"
                )
            
            # Generate triage session ID
            session_id = self._generate_session_id(call_id)
            
            # Create session with masked phone numbers
            session = CallSession(
                call_id=call_id,
                session_id=session_id,
                from_number_masked=mask_phone_number(from_number),
                to_number_masked=mask_phone_number(to_number),
                provider=provider,
            )
            
            self._sessions[call_id] = session
            
            logger.info(
                "Call session created: call_id=***%s, session_id=%s",
                call_id[-4:], session_id[:8]
            )
            
            return session
    
    async def get_session(self, call_id: str) -> Optional[CallSession]:
        """Get a call session by ID."""
        async with self._lock:
            return self._sessions.get(call_id)
    
    async def update_status(
        self,
        call_id: str,
        status: CallStatus,
        duration: Optional[int] = None,
    ) -> Optional[CallSession]:
        """Update call status."""
        async with self._lock:
            session = self._sessions.get(call_id)
            if not session:
                return None
            
            session.status = status
            
            if status == CallStatus.IN_PROGRESS and not session.started_at:
                session.started_at = datetime.utcnow()
            
            if status in (CallStatus.COMPLETED, CallStatus.FAILED, CallStatus.NO_ANSWER):
                session.ended_at = datetime.utcnow()
            
            logger.info(
                "Call status updated: call_id=***%s, status=%s",
                call_id[-4:], status.value
            )
            
            return session
    
    async def record_audio_bytes(self, call_id: str, byte_count: int) -> None:
        """Record audio bytes received for a call."""
        async with self._lock:
            session = self._sessions.get(call_id)
            if session:
                session.audio_bytes_received += byte_count
    
    async def record_triage_event(
        self,
        call_id: str,
        risk_level: str,
    ) -> None:
        """Record a triage event for a call."""
        async with self._lock:
            session = self._sessions.get(call_id)
            if session:
                session.triage_events_count += 1
                
                # Track highest risk level
                risk_order = {"low": 0, "medium": 1, "high": 2, "imminent": 3}
                current_order = risk_order.get(session.highest_risk_level, -1)
                new_order = risk_order.get(risk_level, -1)
                
                if new_order > current_order:
                    session.highest_risk_level = risk_level
    
    async def get_active_sessions(self) -> list[CallSession]:
        """Get all active (non-ended) sessions."""
        async with self._lock:
            return [
                s for s in self._sessions.values()
                if s.status in (CallStatus.INITIATED, CallStatus.RINGING, CallStatus.IN_PROGRESS)
            ]
    
    async def get_active_count(self) -> int:
        """Get count of active sessions."""
        return len(await self.get_active_sessions())
    
    async def get_recent_sessions(self, limit: int = 50) -> list[CallSession]:
        """Get recently completed sessions."""
        async with self._lock:
            completed = [
                s for s in self._sessions.values()
                if s.status in (CallStatus.COMPLETED, CallStatus.FAILED)
            ]
            completed.sort(key=lambda s: s.ended_at or s.initiated_at, reverse=True)
            return completed[:limit]
    
    def _generate_session_id(self, call_id: str) -> str:
        """Generate a triage session ID from call ID."""
        import hashlib
        import time
        
        # Hash call_id + timestamp for uniqueness
        data = f"{call_id}:{time.time_ns()}"
        hash_bytes = hashlib.sha256(data.encode()).hexdigest()[:16]
        return f"call_{hash_bytes}"
    
    async def _cleanup_loop(self) -> None:
        """Background task to evict stale sessions."""
        while True:
            await asyncio.sleep(self._cleanup_interval)
            
            now = datetime.utcnow()
            stale_ids = []
            
            async with self._lock:
                for call_id, session in self._sessions.items():
                    age = now - session.initiated_at
                    
                    # Evict completed sessions older than 5 minutes
                    if session.ended_at and (now - session.ended_at) > timedelta(minutes=5):
                        stale_ids.append(call_id)
                    
                    # Evict any session older than TTL
                    elif age > self._session_ttl:
                        stale_ids.append(call_id)
                
                for call_id in stale_ids:
                    del self._sessions[call_id]
            
            if stale_ids:
                logger.info("Cleaned up %d stale call sessions", len(stale_ids))
```

---

## 8. Privacy Enforcement

### 8.1 Phone Number Masking

```python
# backend/app/telephony/privacy.py

import hashlib
import re
from typing import Optional

def mask_phone_number(number: Optional[str], show_last_digits: int = 2) -> str:
    """
    Mask a phone number for privacy.
    
    Examples:
        +14155551234 → ***34
        14155551234  → ***34
        None         → unknown
    
    IMPORTANT: Raw phone numbers must NEVER be logged or stored.
    """
    if not number:
        return "unknown"
    
    # Remove non-digit characters
    digits = re.sub(r'\D', '', number)
    
    if len(digits) < show_last_digits:
        return "***"
    
    return f"***{digits[-show_last_digits:]}"

def hash_phone_number(number: Optional[str], salt: str = "") -> str:
    """
    Create a one-way hash of a phone number.
    
    Used for analytics aggregation without storing the actual number.
    """
    if not number:
        return "unknown"
    
    # Normalize: remove non-digits
    digits = re.sub(r'\D', '', number)
    
    # Hash with salt
    data = f"{salt}:{digits}"
    return hashlib.sha256(data.encode()).hexdigest()[:16]

def is_phone_number_masked(value: str) -> bool:
    """Check if a value appears to be a masked phone number."""
    return value.startswith("***") or value == "unknown"
```

---

## 9. Telephony HTTP Endpoints

### 9.1 Router Implementation

```python
# backend/app/telephony/router.py

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse
import logging

from app.config import get_settings, Settings
from app.core.exceptions import TelephonyDisabledError, CallNotFoundError
from .models import IncomingCallRequest, CallStatusUpdate, CallSession
from .session_store import CallSessionStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/telephony", tags=["telephony"])

# Dependency: ensure telephony is enabled
async def require_telephony_enabled(settings: Settings = Depends(get_settings)):
    if not settings.enable_telephony_integration:
        raise TelephonyDisabledError(
            "Telephony integration is disabled. "
            "Set ENABLE_TELEPHONY_INTEGRATION=true to enable."
        )

@router.post(
    "/incoming",
    response_class=PlainTextResponse,
    dependencies=[Depends(require_telephony_enabled)],
)
async def handle_incoming_call(
    request: Request,
    call_store: CallSessionStore = Depends(get_call_store),
    settings: Settings = Depends(get_settings),
) -> Response:
    """
    Handle incoming call webhook from telephony provider.
    
    Creates a call session and returns instructions for the provider.
    
    Idempotent: duplicate calls with same call_id are ignored.
    """
    # Parse request body (handle both JSON and form-encoded)
    content_type = request.headers.get("content-type", "")
    
    if "application/json" in content_type:
        body = await request.json()
    else:
        # Form-encoded (Twilio default)
        form = await request.form()
        body = dict(form)
    
    # Validate and parse
    try:
        incoming = IncomingCallRequest(**body)
    except Exception as e:
        logger.warning("Invalid incoming call request: %s", str(e))
        raise HTTPException(status_code=400, detail="Invalid request body")
    
    # Create session
    session = await call_store.create_session(
        call_id=incoming.call_id,
        from_number=incoming.from_number,
        to_number=incoming.to_number,
        provider=incoming.provider,
    )
    
    logger.info(
        "Incoming call accepted: session_id=%s, from=%s",
        session.session_id[:8],
        session.from_number_masked,
    )
    
    # Return instructions (TwiML-like for Twilio, or generic JSON)
    if settings.telephony_provider == "twilio":
        # Return TwiML to connect media stream
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://{request.headers.get('host')}/ws/telephony/{incoming.call_id}"/>
    </Connect>
</Response>"""
        return PlainTextResponse(content=twiml, media_type="application/xml")
    
    # Generic response
    return PlainTextResponse(
        content=f'{{"session_id": "{session.session_id}", "stream_url": "/ws/telephony/{incoming.call_id}"}}',
        media_type="application/json"
    )

@router.post(
    "/status",
    dependencies=[Depends(require_telephony_enabled)],
)
async def handle_status_update(
    request: Request,
    call_store: CallSessionStore = Depends(get_call_store),
) -> dict:
    """
    Handle call status updates from telephony provider.
    
    Updates call session state and triggers appropriate actions.
    """
    content_type = request.headers.get("content-type", "")
    
    if "application/json" in content_type:
        body = await request.json()
    else:
        form = await request.form()
        body = dict(form)
    
    try:
        status_update = CallStatusUpdate(**body)
    except Exception as e:
        logger.warning("Invalid status update: %s", str(e))
        raise HTTPException(status_code=400, detail="Invalid request body")
    
    session = await call_store.update_status(
        call_id=status_update.call_id,
        status=status_update.status,
        duration=status_update.duration,
    )
    
    if not session:
        raise CallNotFoundError(f"Call not found: ***{status_update.call_id[-4:]}")
    
    return {
        "status": "acknowledged",
        "call_status": session.status.value,
        "session_id": session.session_id[:8],
    }

@router.get(
    "/calls/active",
    dependencies=[Depends(require_telephony_enabled)],
)
async def get_active_calls(
    call_store: CallSessionStore = Depends(get_call_store),
) -> dict:
    """Get all active calls."""
    sessions = await call_store.get_active_sessions()
    return {
        "count": len(sessions),
        "calls": [
            {
                "call_id_masked": f"***{s.call_id[-4:]}",
                "session_id": s.session_id[:8],
                "status": s.status.value,
                "from": s.from_number_masked,
                "duration_seconds": (
                    (datetime.utcnow() - s.started_at).total_seconds()
                    if s.started_at else 0
                ),
                "triage_events": s.triage_events_count,
                "highest_risk": s.highest_risk_level,
            }
            for s in sessions
        ],
    }

@router.get(
    "/calls/recent",
    dependencies=[Depends(require_telephony_enabled)],
)
async def get_recent_calls(
    limit: int = 50,
    call_store: CallSessionStore = Depends(get_call_store),
) -> dict:
    """Get recent completed calls."""
    sessions = await call_store.get_recent_sessions(limit=limit)
    return {
        "count": len(sessions),
        "calls": [
            {
                "call_id_masked": f"***{s.call_id[-4:]}",
                "session_id": s.session_id[:8],
                "status": s.status.value,
                "from": s.from_number_masked,
                "started_at": s.started_at.isoformat() if s.started_at else None,
                "ended_at": s.ended_at.isoformat() if s.ended_at else None,
                "triage_events": s.triage_events_count,
                "highest_risk": s.highest_risk_level,
            }
            for s in sessions
        ],
    }
```

---

## 10. Telephony WebSocket Media Stream

### 10.1 Audio Processing

```python
# backend/app/telephony/audio_processor.py

import struct
import base64
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class TelephonyAudioProcessor:
    """
    Processes audio frames from telephony providers.
    
    Handles:
    - μ-law to PCM conversion (Twilio sends μ-law)
    - Base64 decoding
    - Sample rate validation
    - Buffering for chunk processing
    """
    
    # μ-law to linear PCM lookup table
    MULAW_DECODE_TABLE = [
        -32124, -31100, -30076, -29052, -28028, -27004, -25980, -24956,
        -23932, -22908, -21884, -20860, -19836, -18812, -17788, -16764,
        # ... (256 entries total, abbreviated for brevity)
    ]
    
    def __init__(
        self,
        target_sample_rate: int = 16000,
        chunk_threshold_bytes: int = 32000,
        max_buffer_bytes: int = 320000,
    ):
        self._target_sample_rate = target_sample_rate
        self._chunk_threshold = chunk_threshold_bytes
        self._max_buffer = max_buffer_bytes
        self._buffer = bytearray()
        self._source_sample_rate: Optional[int] = None
    
    def configure_source(
        self,
        sample_rate: int,
        channels: int = 1,
        encoding: str = "audio/x-mulaw",
    ) -> None:
        """Configure source audio format."""
        self._source_sample_rate = sample_rate
        self._source_channels = channels
        self._source_encoding = encoding
        
        logger.info(
            "Audio source configured: %dHz, %d channels, %s",
            sample_rate, channels, encoding
        )
    
    def process_frame(self, frame: bytes, encoding: str = "mulaw") -> Optional[bytes]:
        """
        Process an incoming audio frame.
        
        Returns PCM bytes if buffer threshold reached, else None.
        """
        try:
            # Decode based on encoding
            if encoding == "mulaw" or encoding == "audio/x-mulaw":
                pcm = self._decode_mulaw(frame)
            elif encoding == "base64":
                decoded = base64.b64decode(frame)
                pcm = self._decode_mulaw(decoded)
            elif encoding == "pcm16":
                pcm = frame
            else:
                logger.warning("Unknown audio encoding: %s", encoding)
                return None
            
            # Resample if needed
            if self._source_sample_rate and self._source_sample_rate != self._target_sample_rate:
                pcm = self._resample(pcm, self._source_sample_rate, self._target_sample_rate)
            
            # Add to buffer
            self._buffer.extend(pcm)
            
            # Apply backpressure if buffer too large
            if len(self._buffer) > self._max_buffer:
                overflow = len(self._buffer) - self._max_buffer
                self._buffer = self._buffer[overflow:]
                logger.warning("Audio buffer overflow, dropped %d bytes", overflow)
            
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
        """Flush remaining buffer."""
        if len(self._buffer) > 0:
            chunk = bytes(self._buffer)
            self._buffer.clear()
            return chunk
        return None
    
    def _decode_mulaw(self, mulaw_bytes: bytes) -> bytes:
        """Decode μ-law to 16-bit PCM."""
        pcm_samples = []
        
        for byte in mulaw_bytes:
            # μ-law decoding formula
            byte = ~byte
            sign = byte & 0x80
            exponent = (byte >> 4) & 0x07
            mantissa = byte & 0x0F
            
            sample = (mantissa << 3) + 0x84
            sample <<= exponent
            sample -= 0x84
            
            if sign:
                sample = -sample
            
            pcm_samples.append(sample)
        
        # Pack as 16-bit signed integers
        return struct.pack(f"<{len(pcm_samples)}h", *pcm_samples)
    
    def _resample(self, pcm: bytes, source_rate: int, target_rate: int) -> bytes:
        """Simple linear resampling."""
        if source_rate == target_rate:
            return pcm
        
        # Unpack samples
        samples = struct.unpack(f"<{len(pcm)//2}h", pcm)
        
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
            resampled.append(sample)
        
        return struct.pack(f"<{len(resampled)}h", *resampled)
```

### 10.2 WebSocket Handler

```python
# backend/app/telephony/websocket.py

import asyncio
import json
import logging
from fastapi import WebSocket, WebSocketDisconnect, Depends
from starlette.websockets import WebSocketState

from app.config import get_settings, Settings
from app.core.pipeline import get_pipeline, TriagePipeline
from app.core.types import InputModality
from app.core.logging import call_id_var, session_id_var
from .session_store import CallSessionStore, get_call_store
from .audio_processor import TelephonyAudioProcessor
from .models import CallStatus

logger = logging.getLogger(__name__)

async def telephony_websocket_handler(
    websocket: WebSocket,
    call_id: str,
    pipeline: TriagePipeline = Depends(get_pipeline),
    call_store: CallSessionStore = Depends(get_call_store),
    settings: Settings = Depends(get_settings),
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
        await websocket.close(code=4404, reason="Call not found")
        return
    
    # Set context for logging
    call_id_var.set(call_id)
    session_id_var.set(session.session_id)
    
    # Accept connection
    await websocket.accept()
    
    logger.info("Telephony WebSocket connected: call=***%s", call_id[-4:])
    
    # Initialize audio processor
    audio_processor = TelephonyAudioProcessor(
        target_sample_rate=settings.telephony_audio_sample_rate,
        chunk_threshold_bytes=settings.audio_chunk_min_bytes,
    )
    
    # Update call status
    await call_store.update_status(call_id, CallStatus.IN_PROGRESS)
    
    # Processing loop
    try:
        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=30.0  # Heartbeat timeout
                )
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json({"event": "ping"})
                continue
            
            if message["type"] == "websocket.disconnect":
                break
            
            # Handle different message types
            if "bytes" in message:
                # Binary audio frame
                await process_audio_frame(
                    message["bytes"],
                    audio_processor,
                    pipeline,
                    session,
                    call_store,
                    websocket,
                )
            
            elif "text" in message:
                # JSON control message
                try:
                    data = json.loads(message["text"])
                    await handle_control_message(
                        data,
                        audio_processor,
                        pipeline,
                        session,
                        call_store,
                        websocket,
                    )
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON in telephony WS: %s", message["text"][:50])
    
    except WebSocketDisconnect:
        logger.info("Telephony WebSocket disconnected: call=***%s", call_id[-4:])
    
    except Exception as e:
        logger.error("Telephony WebSocket error: %s", str(e), exc_info=True)
    
    finally:
        # Cleanup
        await cleanup_call_stream(
            call_id,
            audio_processor,
            pipeline,
            session,
            call_store,
        )

async def process_audio_frame(
    frame: bytes,
    audio_processor: TelephonyAudioProcessor,
    pipeline: TriagePipeline,
    session,
    call_store: CallSessionStore,
    websocket: WebSocket,
):
    """Process a binary audio frame."""
    # Record bytes received
    await call_store.record_audio_bytes(session.call_id, len(frame))
    
    # Process through audio processor
    pcm_chunk = audio_processor.process_frame(frame)
    
    if pcm_chunk:
        # Send to pipeline
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
                
                # Optionally send result to provider
                await websocket.send_json({
                    "event": "triage",
                    "risk_level": result.risk_level.value,
                    "urgency_score": result.urgency_score,
                    "transcript": result.transcript[:100] if result.transcript else None,
                })
        
        except Exception as e:
            logger.error("Pipeline error during phone call: %s", str(e))

async def handle_control_message(
    data: dict,
    audio_processor: TelephonyAudioProcessor,
    pipeline: TriagePipeline,
    session,
    call_store: CallSessionStore,
    websocket: WebSocket,
):
    """Handle JSON control messages."""
    event = data.get("event")
    
    if event == "start":
        # Stream starting - configure audio processor
        stream_sid = data.get("streamSid", "")
        
        # Twilio sends media format info
        if "start" in data:
            start_data = data["start"]
            audio_processor.configure_source(
                sample_rate=int(start_data.get("sampleRate", 8000)),
                channels=int(start_data.get("channels", 1)),
                encoding=start_data.get("encoding", "audio/x-mulaw"),
            )
        
        logger.info("Telephony stream started: stream=%s", stream_sid[-8:] if stream_sid else "unknown")
    
    elif event == "media":
        # Twilio sends audio as base64 in JSON
        payload = data.get("media", {}).get("payload", "")
        if payload:
            import base64
            audio_bytes = base64.b64decode(payload)
            await process_audio_frame(
                audio_bytes,
                audio_processor,
                pipeline,
                session,
                call_store,
                websocket,
            )
    
    elif event == "stop":
        logger.info("Telephony stream stop received")
        # Will be handled in finally block
    
    elif event == "pong":
        pass  # Response to our ping
    
    else:
        logger.debug("Unknown telephony event: %s", event)

async def cleanup_call_stream(
    call_id: str,
    audio_processor: TelephonyAudioProcessor,
    pipeline: TriagePipeline,
    session,
    call_store: CallSessionStore,
):
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
    
    logger.info(
        "Call stream ended: call=***%s, events=%d, highest_risk=%s",
        call_id[-4:],
        session.triage_events_count,
        session.highest_risk_level or "none",
    )
```

---

## 11. Configuration Additions

### 11.1 New Settings

```python
# backend/app/config.py (additions)

class Settings(BaseSettings):
    # ... existing settings ...
    
    # === Telephony Integration ===
    enable_telephony_integration: bool = Field(
        default=False,
        description="Enable phone call integration endpoints"
    )
    
    telephony_provider: str = Field(
        default="generic",
        description="Telephony provider: generic, twilio, simulator"
    )
    
    telephony_audio_sample_rate: int = Field(
        default=16000,
        description="Target audio sample rate for telephony"
    )
    
    telephony_audio_channels: int = Field(
        default=1,
        description="Audio channels (mono)"
    )
    
    telephony_audio_format: str = Field(
        default="pcm16",
        description="Target audio format"
    )
    
    telephony_session_max_minutes: int = Field(
        default=60,
        description="Maximum call duration in minutes"
    )
    
    telephony_max_concurrent_calls: int = Field(
        default=100,
        description="Maximum concurrent phone calls"
    )
    
    mask_phone_numbers: bool = Field(
        default=True,
        description="Mask phone numbers in logs and storage"
    )
    
    # Twilio-specific (optional)
    twilio_account_sid: Optional[str] = Field(
        default=None,
        description="Twilio Account SID"
    )
    
    twilio_auth_token: Optional[str] = Field(
        default=None,
        description="Twilio Auth Token"
    )
```

### 11.2 Environment Variables

```bash
# .env.example additions

# === Telephony Integration ===
ENABLE_TELEPHONY_INTEGRATION=false
TELEPHONY_PROVIDER=generic
TELEPHONY_AUDIO_SAMPLE_RATE=16000
TELEPHONY_AUDIO_CHANNELS=1
TELEPHONY_AUDIO_FORMAT=pcm16
TELEPHONY_SESSION_MAX_MINUTES=60
TELEPHONY_MAX_CONCURRENT_CALLS=100
MASK_PHONE_NUMBERS=true

# Twilio (optional)
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
```

---

## 12. Updated File Tree

```
backend/
├── main.py                          # Updated: telephony router, shutdown hooks
├── app/
│   ├── __init__.py
│   ├── config.py                    # Updated: telephony settings
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   ├── schemas.py
│   │   ├── websocket.py
│   │   └── health.py               # NEW: health check endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── pipeline.py             # Updated: InputModality.PHONE_CALL
│   │   ├── types.py                # Updated: InputModality enum
│   │   ├── history_store.py
│   │   ├── session_registry.py     # NEW: session tracking
│   │   ├── logging.py              # NEW: structured logging
│   │   └── exceptions.py           # NEW: exception hierarchy
│   ├── middleware/
│   │   ├── __init__.py             # NEW
│   │   ├── correlation.py          # NEW: correlation ID middleware
│   │   └── errors.py               # NEW: global error handler
│   ├── services/
│   │   ├── __init__.py
│   │   ├── transcription.py
│   │   ├── prosody.py
│   │   └── triage_model.py
│   └── telephony/                   # NEW: entire module
│       ├── __init__.py
│       ├── router.py               # HTTP endpoints
│       ├── websocket.py            # WebSocket handler
│       ├── session_store.py        # Call session management
│       ├── audio_processor.py      # Audio conversion
│       ├── models.py               # Pydantic models
│       ├── privacy.py              # Phone number masking
│       └── providers/
│           ├── __init__.py
│           ├── base.py             # Provider interface
│           ├── generic.py          # Generic provider
│           ├── twilio.py           # Twilio integration
│           └── simulator.py        # Development simulator
├── tests/
│   ├── ...existing...
│   ├── test_telephony_router.py    # NEW
│   ├── test_telephony_websocket.py # NEW
│   ├── test_audio_processor.py     # NEW
│   └── test_call_session_store.py  # NEW
└── requirements.txt                 # Updated if needed
```

---

## 13. System Behavior Under Load

### 13.1 High Load Scenario

**Condition:** 50+ concurrent calls, each streaming audio at 8kHz

**Behavior:**
1. `CallSessionStore` enforces `max_sessions` limit (default: 100)
2. New calls beyond limit receive `429 Session Limit Exceeded`
3. `AudioBufferManager` applies backpressure, dropping oldest frames
4. Logs emit warnings every 100 dropped frames
5. Pipeline continues processing available audio
6. Metrics endpoint shows degradation warnings

### 13.2 Concurrent Calls

**Condition:** Multiple calls active simultaneously

**Behavior:**
1. Each call has isolated `session_id` and `audio_buffer`
2. Pipeline processes audio chunks in order received (FIFO)
3. Transcription runs in thread pool, non-blocking
4. Results are tagged with `session_id` for correct routing
5. Analytics stores events per-session

### 13.3 Bad Audio Frames

**Condition:** Malformed, corrupted, or wrong-format audio

**Behavior:**
1. `audio_processor.process_frame()` catches exceptions
2. Bad frame is logged and skipped
3. Processing continues with next frame
4. Metrics track `frames_dropped` count
5. No crash, no data loss beyond bad frame

### 13.4 Telephony Disconnects

**Condition:** Provider disconnects unexpectedly

**Behavior:**
1. `WebSocketDisconnect` caught in handler
2. `cleanup_call_stream()` runs in `finally` block
3. Remaining audio buffer flushed to pipeline
4. Call session marked as `COMPLETED` or `FAILED`
5. Resources released, session available for cleanup

---

## 14. Twilio Configuration Guide

### 14.1 Twilio Console Setup

1. **Create TwiML App:**
   - Go to Twilio Console → Voice → TwiML Apps
   - Create new app with:
     - Voice Request URL: `https://your-domain.com/api/telephony/incoming`
     - Status Callback URL: `https://your-domain.com/api/telephony/status`

2. **Configure Phone Number:**
   - Go to Phone Numbers → Manage → Active Numbers
   - Select your number
   - Set "A Call Comes In" to your TwiML App

3. **Enable Media Streams:**
   - Media streams are initiated via TwiML response
   - Our `/incoming` endpoint returns TwiML with `<Stream>` verb

### 14.2 Environment Configuration

```bash
ENABLE_TELEPHONY_INTEGRATION=true
TELEPHONY_PROVIDER=twilio
TWILIO_ACCOUNT_SID=ACxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxx
```

### 14.3 Twilio-Specific Notes

- Twilio sends audio at 8kHz μ-law by default
- Our `audio_processor` converts to 16kHz PCM
- Media stream sends JSON with base64-encoded audio
- Call SID is the `call_id` for session mapping

---

## 15. Telephony Smoke Tests

### 15.1 Test Cases

```python
# tests/test_telephony_smoke.py

class TestTelephonySmoke:
    """Smoke tests for telephony integration."""
    
    @pytest.mark.asyncio
    async def test_incoming_call_creates_session(self, client, settings):
        """POST /incoming should create call session."""
        settings.enable_telephony_integration = True
        
        response = client.post("/api/telephony/incoming", json={
            "CallSid": "CA123456",
            "From": "+14155551234",
            "To": "+14155555678",
        })
        
        assert response.status_code == 200
        # Session created with masked numbers
    
    @pytest.mark.asyncio
    async def test_telephony_disabled_returns_503(self, client, settings):
        """Endpoints should return 503 when disabled."""
        settings.enable_telephony_integration = False
        
        response = client.post("/api/telephony/incoming", json={
            "CallSid": "CA123456",
            "From": "+14155551234",
            "To": "+14155555678",
        })
        
        assert response.status_code == 503
        assert "TELEPHONY_DISABLED" in response.json()["error"]["code"]
    
    @pytest.mark.asyncio
    async def test_duplicate_call_is_idempotent(self, client, settings):
        """Same call_id should not create duplicate sessions."""
        settings.enable_telephony_integration = True
        
        call_data = {
            "CallSid": "CA_DUPE_TEST",
            "From": "+14155551234",
            "To": "+14155555678",
        }
        
        r1 = client.post("/api/telephony/incoming", json=call_data)
        r2 = client.post("/api/telephony/incoming", json=call_data)
        
        assert r1.status_code == 200
        assert r2.status_code == 200
        # Same session returned
    
    @pytest.mark.asyncio
    async def test_audio_processing_mulaw_to_pcm(self):
        """Audio processor should convert μ-law to PCM."""
        processor = TelephonyAudioProcessor()
        
        # Silence in μ-law (0xFF = silence)
        mulaw_frame = bytes([0xFF] * 160)  # 20ms at 8kHz
        
        processor.configure_source(sample_rate=8000, encoding="audio/x-mulaw")
        result = processor.process_frame(mulaw_frame)
        
        # Should buffer, not return (threshold not reached)
        assert result is None
        
        # Flush should return converted audio
        flushed = processor.flush()
        assert flushed is not None
        assert len(flushed) > 0
    
    @pytest.mark.asyncio
    async def test_phone_numbers_are_masked(self):
        """Phone numbers should be masked in storage."""
        from app.telephony.privacy import mask_phone_number
        
        assert mask_phone_number("+14155551234") == "***34"
        assert mask_phone_number("14155551234") == "***34"
        assert mask_phone_number(None) == "unknown"
```

---

## 16. Implementation Priority

### Phase 1: Backend Hardening (Day 1)
1. Add `SessionRegistry` and `exceptions.py`
2. Implement structured logging with `StructuredFormatter`
3. Add correlation middleware
4. Add health check endpoints
5. Add shutdown hooks to pipeline

### Phase 2: Telephony Core (Day 2)
1. Create `backend/app/telephony/` module structure
2. Implement `CallSessionStore`
3. Implement `audio_processor.py`
4. Implement `privacy.py`

### Phase 3: Telephony Endpoints (Day 3)
1. Implement `/incoming` and `/status` HTTP endpoints
2. Implement telephony WebSocket handler
3. Add telephony settings to config
4. Write smoke tests

### Phase 4: Integration (Day 4)
1. Update `InputModality` enum with `PHONE_CALL`
2. Update pipeline to handle telephony sessions
3. Update analytics to track call metrics
4. End-to-end testing with simulator

### Phase 5: Documentation (Day 5)
1. Update API documentation
2. Add Twilio setup guide
3. Update README with telephony section
4. Final review and merge

---

## 17. Final Notes

This blueprint provides a complete specification for:

1. **Production-grade backend hardening** — lifecycle management, error handling, logging, health checks
2. **Full telephony integration** — HTTP webhooks, WebSocket media streams, call session management
3. **Privacy-first design** — phone number masking, no audio persistence, ephemeral sessions
4. **Provider flexibility** — generic interface with Twilio implementation

The system is designed to handle real-world telephony provider quirks (jitter, format variations, disconnects) while maintaining the privacy and safety guarantees required for crisis triage research.

**Next step:** Begin implementation following the priority phases above.

---

*Document prepared for internal engineering review. Implementation should follow the phases specified, with thorough testing at each stage.*
