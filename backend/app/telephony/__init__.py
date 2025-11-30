"""
CrisisTriage AI - Telephony Integration Module

Provides phone call integration for real-time crisis triage.

Components:
- router: HTTP endpoints for call webhooks
- websocket: Media stream handler
- session_store: Call session management
- audio_processor: Audio format conversion
- privacy: Phone number masking

SAFETY NOTICE:
    This module is for RESEARCH AND SIMULATION ONLY.
    Phone numbers are masked and never stored in cleartext.
    Audio is processed in memory and never persisted.
"""

from .models import CallSession, CallStatus, CallDirection
from .privacy import mask_phone_number, hash_phone_number
from .session_store import CallSessionStore

__all__ = [
    "CallSession",
    "CallStatus", 
    "CallDirection",
    "CallSessionStore",
    "mask_phone_number",
    "hash_phone_number",
]
