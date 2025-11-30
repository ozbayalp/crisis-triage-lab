"""
CrisisTriage AI - Telephony HTTP Endpoints

HTTP webhooks for telephony provider integration.
Handles incoming calls and status updates.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import PlainTextResponse, JSONResponse

from app.config import Settings, get_settings
from app.core.exceptions import TelephonyDisabledError, CallNotFoundError
from .models import (
    IncomingCallRequest,
    CallStatusUpdate,
    CallSessionResponse,
)
from .session_store import CallSessionStore, get_call_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/telephony", tags=["telephony"])


# =============================================================================
# Dependencies
# =============================================================================

async def require_telephony_enabled(
    settings: Settings = Depends(get_settings),
) -> Settings:
    """Dependency that ensures telephony is enabled."""
    if not settings.enable_telephony_integration:
        raise TelephonyDisabledError(
            "Telephony integration is disabled. "
            "Set ENABLE_TELEPHONY_INTEGRATION=true to enable."
        )
    return settings


# =============================================================================
# Webhook Endpoints
# =============================================================================

@router.post(
    "/incoming",
    response_class=PlainTextResponse,
    summary="Handle incoming call",
    description="Webhook endpoint for telephony provider to notify of incoming calls.",
)
async def handle_incoming_call(
    request: Request,
    settings: Settings = Depends(require_telephony_enabled),
    call_store: CallSessionStore = Depends(get_call_store),
) -> PlainTextResponse:
    """
    Handle incoming call webhook from telephony provider.
    
    Creates a call session and returns instructions for the provider.
    Idempotent: duplicate calls with same call_id are ignored.
    
    Supports:
    - JSON body
    - Form-encoded body (Twilio default)
    """
    # Parse request body
    content_type = request.headers.get("content-type", "")
    
    try:
        if "application/json" in content_type:
            body = await request.json()
        else:
            # Form-encoded (Twilio default)
            form = await request.form()
            body = dict(form)
        
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
        "Incoming call accepted: session=%s, from=%s",
        session.session_id[:8],
        session.from_number_masked,
    )
    
    # Return provider-specific instructions
    if settings.telephony_provider == "twilio":
        # Return TwiML to connect media stream
        host = request.headers.get("host", "localhost:8000")
        scheme = "wss" if request.url.scheme == "https" else "ws"
        
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{scheme}://{host}/ws/telephony/{incoming.call_id}"/>
    </Connect>
</Response>"""
        return PlainTextResponse(content=twiml, media_type="application/xml")
    
    # Generic response
    return PlainTextResponse(
        content=f'{{"session_id": "{session.session_id}", "status": "accepted"}}',
        media_type="application/json",
    )


@router.post(
    "/status",
    summary="Handle call status update",
    description="Webhook endpoint for call status updates from telephony provider.",
)
async def handle_status_update(
    request: Request,
    settings: Settings = Depends(require_telephony_enabled),
    call_store: CallSessionStore = Depends(get_call_store),
) -> dict:
    """
    Handle call status updates from telephony provider.
    
    Updates call session state and triggers appropriate actions.
    """
    content_type = request.headers.get("content-type", "")
    
    try:
        if "application/json" in content_type:
            body = await request.json()
        else:
            form = await request.form()
            body = dict(form)
        
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
        raise CallNotFoundError(
            f"Call not found: ***{status_update.call_id[-4:]}"
            if len(status_update.call_id) > 4
            else "Call not found"
        )
    
    return {
        "status": "acknowledged",
        "call_status": session.status.value,
        "session_id": session.session_id[:8],
    }


# =============================================================================
# Call Management Endpoints
# =============================================================================

@router.get(
    "/calls/active",
    summary="Get active calls",
    description="Get all currently active phone calls.",
)
async def get_active_calls(
    settings: Settings = Depends(require_telephony_enabled),
    call_store: CallSessionStore = Depends(get_call_store),
) -> dict:
    """Get all active calls."""
    sessions = await call_store.get_active_sessions()
    
    return {
        "count": len(sessions),
        "calls": [
            CallSessionResponse.from_session(s).model_dump()
            for s in sessions
        ],
    }


@router.get(
    "/calls/recent",
    summary="Get recent calls",
    description="Get recently completed phone calls.",
)
async def get_recent_calls(
    limit: int = 50,
    settings: Settings = Depends(require_telephony_enabled),
    call_store: CallSessionStore = Depends(get_call_store),
) -> dict:
    """Get recent completed calls."""
    sessions = await call_store.get_recent_sessions(limit=limit)
    
    return {
        "count": len(sessions),
        "calls": [
            CallSessionResponse.from_session(s).model_dump()
            for s in sessions
        ],
    }


@router.get(
    "/calls/{call_id}",
    summary="Get call details",
    description="Get details for a specific call.",
)
async def get_call_details(
    call_id: str,
    settings: Settings = Depends(require_telephony_enabled),
    call_store: CallSessionStore = Depends(get_call_store),
) -> dict:
    """Get details for a specific call."""
    session = await call_store.get_session_or_raise(call_id)
    
    return {
        "call": CallSessionResponse.from_session(session).model_dump(),
    }


# =============================================================================
# Development Endpoints
# =============================================================================

@router.post(
    "/simulate/incoming",
    summary="Simulate incoming call (dev only)",
    description="Simulate an incoming call for development/testing.",
)
async def simulate_incoming_call(
    from_number: str = "+15551234567",
    to_number: str = "+15559876543",
    settings: Settings = Depends(require_telephony_enabled),
    call_store: CallSessionStore = Depends(get_call_store),
) -> dict:
    """
    Simulate an incoming call for development/testing.
    
    Creates a call session that can be connected via WebSocket.
    """
    if settings.app_env not in ("development", "test"):
        raise HTTPException(
            status_code=403,
            detail="Simulation endpoints only available in development",
        )
    
    import uuid
    call_id = f"SIM_{uuid.uuid4().hex[:12]}"
    
    session = await call_store.create_session(
        call_id=call_id,
        from_number=from_number,
        to_number=to_number,
        provider="simulator",
    )
    
    return {
        "call_id": call_id,
        "session_id": session.session_id,
        "websocket_url": f"/ws/telephony/{call_id}",
        "status": "simulated",
    }
