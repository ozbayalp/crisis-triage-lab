"""
CrisisTriage AI - Telephony Data Models

Pydantic models for telephony requests and call session state.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class CallStatus(str, Enum):
    """Call lifecycle status."""
    INITIATED = "initiated"
    RINGING = "ringing"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NO_ANSWER = "no_answer"
    BUSY = "busy"
    CANCELED = "canceled"


class CallDirection(str, Enum):
    """Call direction."""
    INBOUND = "inbound"
    OUTBOUND = "outbound"


class CallSession(BaseModel):
    """
    Represents an active or completed phone call.
    
    Privacy:
        - Phone numbers are stored in masked form only
        - No raw audio is ever stored
        - Session data is ephemeral (memory-only)
    """
    
    # Identifiers
    call_id: str = Field(..., description="Unique call identifier from provider")
    session_id: str = Field(..., description="Internal triage session ID")
    
    # Masked phone numbers (privacy)
    from_number_masked: str = Field(..., description="Caller number (masked)")
    to_number_masked: str = Field(..., description="Callee number (masked)")
    
    # Call metadata
    direction: CallDirection = CallDirection.INBOUND
    status: CallStatus = CallStatus.INITIATED
    provider: str = "generic"
    
    # Timestamps
    initiated_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None  # When audio stream began
    ended_at: Optional[datetime] = None
    
    # Metrics
    audio_bytes_received: int = 0
    triage_events_count: int = 0
    highest_risk_level: Optional[str] = None
    
    # Provider-specific metadata (non-sensitive only)
    provider_metadata: dict = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() + "Z" if v else None
        }


class IncomingCallRequest(BaseModel):
    """
    Request body for POST /api/telephony/incoming
    
    Supports both Twilio-style (CamelCase with aliases) and 
    generic (snake_case) field names.
    """
    
    call_id: str = Field(..., alias="CallSid", description="Provider's call ID")
    from_number: str = Field(..., alias="From", description="Caller phone number")
    to_number: str = Field(..., alias="To", description="Callee phone number")
    direction: CallDirection = CallDirection.INBOUND
    provider: str = "generic"
    
    class Config:
        populate_by_name = True  # Allow both alias and field name


class CallStatusUpdate(BaseModel):
    """
    Request body for POST /api/telephony/status
    
    Handles call status updates from telephony providers.
    """
    
    call_id: str = Field(..., alias="CallSid", description="Provider's call ID")
    status: CallStatus = Field(..., alias="CallStatus", description="New call status")
    duration: Optional[int] = Field(None, alias="CallDuration", description="Duration in seconds")
    
    class Config:
        populate_by_name = True


class CallSessionResponse(BaseModel):
    """API response for call session data."""
    
    call_id_masked: str = Field(..., description="Masked call ID")
    session_id: str = Field(..., description="Truncated session ID")
    status: CallStatus
    from_number: str = Field(..., description="Masked caller number")
    to_number: str = Field(..., description="Masked callee number")
    direction: CallDirection
    initiated_at: datetime
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    triage_events: int = 0
    highest_risk: Optional[str] = None
    
    @classmethod
    def from_session(cls, session: CallSession) -> "CallSessionResponse":
        """Create response from CallSession."""
        duration = None
        if session.started_at and session.ended_at:
            duration = int((session.ended_at - session.started_at).total_seconds())
        elif session.started_at:
            duration = int((datetime.utcnow() - session.started_at).total_seconds())
        
        return cls(
            call_id_masked=f"***{session.call_id[-4:]}" if len(session.call_id) > 4 else "***",
            session_id=session.session_id[:8],
            status=session.status,
            from_number=session.from_number_masked,
            to_number=session.to_number_masked,
            direction=session.direction,
            initiated_at=session.initiated_at,
            started_at=session.started_at,
            ended_at=session.ended_at,
            duration_seconds=duration,
            triage_events=session.triage_events_count,
            highest_risk=session.highest_risk_level,
        )
