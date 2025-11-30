"""
CrisisTriage AI - Call Session Store

In-memory store for active call sessions with automatic cleanup.
Thread-safe and bounded to prevent memory exhaustion.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from datetime import datetime, timedelta
from typing import Optional

from app.core.exceptions import CallNotFoundError, CallLimitError
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
    
    Usage:
        store = CallSessionStore(max_sessions=100)
        await store.start()
        
        session = await store.create_session(
            call_id="CA123",
            from_number="+14155551234",
            to_number="+14155555678",
        )
        
        await store.stop()
    """
    
    def __init__(
        self,
        max_sessions: int = 100,
        session_ttl_minutes: int = 60,
        cleanup_interval_seconds: int = 60,
    ):
        """
        Initialize the call session store.
        
        Args:
            max_sessions: Maximum concurrent call sessions
            session_ttl_minutes: Session time-to-live in minutes
            cleanup_interval_seconds: Background cleanup interval
        """
        self._sessions: dict[str, CallSession] = {}
        self._lock = asyncio.Lock()
        self._max_sessions = max_sessions
        self._session_ttl = timedelta(minutes=session_ttl_minutes)
        self._cleanup_interval = cleanup_interval_seconds
        self._cleanup_task: Optional[asyncio.Task] = None
        self._started = False
    
    async def start(self) -> None:
        """Start background cleanup task."""
        if self._started:
            return
        
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._started = True
        
        logger.info(
            "CallSessionStore started: max=%d, ttl=%s, cleanup_interval=%ds",
            self._max_sessions,
            self._session_ttl,
            self._cleanup_interval,
        )
    
    async def stop(self) -> None:
        """Stop background tasks and clear sessions."""
        if not self._started:
            return
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        async with self._lock:
            count = len(self._sessions)
            self._sessions.clear()
        
        self._started = False
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
        
        Args:
            call_id: Unique call identifier from provider
            from_number: Caller phone number (will be masked)
            to_number: Callee phone number (will be masked)
            provider: Telephony provider name
        
        Returns:
            Created CallSession
        
        Raises:
            CallLimitError: If at capacity
        """
        async with self._lock:
            # Check for duplicate (idempotency)
            if call_id in self._sessions:
                logger.warning(
                    "Duplicate call_id received: ***%s",
                    call_id[-4:] if len(call_id) > 4 else "***"
                )
                return self._sessions[call_id]
            
            # Check capacity
            active_count = sum(
                1 for s in self._sessions.values()
                if s.status in (CallStatus.INITIATED, CallStatus.RINGING, CallStatus.IN_PROGRESS)
            )
            
            if active_count >= self._max_sessions:
                raise CallLimitError(
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
                "Call session created: call=***%s, session=%s, from=%s",
                call_id[-4:] if len(call_id) > 4 else "***",
                session_id[:8],
                session.from_number_masked,
            )
            
            return session
    
    async def get_session(self, call_id: str) -> Optional[CallSession]:
        """Get a call session by ID."""
        async with self._lock:
            return self._sessions.get(call_id)
    
    async def get_session_or_raise(self, call_id: str) -> CallSession:
        """Get a call session by ID or raise if not found."""
        session = await self.get_session(call_id)
        if not session:
            raise CallNotFoundError(
                f"Call not found: ***{call_id[-4:]}" if len(call_id) > 4 else "Call not found"
            )
        return session
    
    async def update_status(
        self,
        call_id: str,
        status: CallStatus,
        duration: Optional[int] = None,
    ) -> Optional[CallSession]:
        """
        Update call status.
        
        Args:
            call_id: Call identifier
            status: New status
            duration: Call duration in seconds (optional)
        
        Returns:
            Updated CallSession or None if not found
        """
        async with self._lock:
            session = self._sessions.get(call_id)
            if not session:
                return None
            
            session.status = status
            
            if status == CallStatus.IN_PROGRESS and not session.started_at:
                session.started_at = datetime.utcnow()
            
            if status in (
                CallStatus.COMPLETED,
                CallStatus.FAILED,
                CallStatus.NO_ANSWER,
                CallStatus.BUSY,
                CallStatus.CANCELED,
            ):
                session.ended_at = datetime.utcnow()
            
            logger.info(
                "Call status updated: call=***%s, status=%s",
                call_id[-4:] if len(call_id) > 4 else "***",
                status.value,
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
        """
        Record a triage event for a call.
        
        Tracks event count and highest risk level seen.
        """
        async with self._lock:
            session = self._sessions.get(call_id)
            if session:
                session.triage_events_count += 1
                
                # Track highest risk level
                risk_order = {"low": 0, "medium": 1, "high": 2, "imminent": 3}
                current_order = risk_order.get(session.highest_risk_level, -1)
                new_order = risk_order.get(risk_level.lower(), -1)
                
                if new_order > current_order:
                    session.highest_risk_level = risk_level.lower()
    
    async def get_active_sessions(self) -> list[CallSession]:
        """Get all active (non-ended) sessions."""
        async with self._lock:
            return [
                s for s in self._sessions.values()
                if s.status in (
                    CallStatus.INITIATED,
                    CallStatus.RINGING,
                    CallStatus.IN_PROGRESS,
                )
            ]
    
    async def get_active_count(self) -> int:
        """Get count of active sessions."""
        sessions = await self.get_active_sessions()
        return len(sessions)
    
    async def get_recent_sessions(
        self,
        limit: int = 50,
        include_active: bool = False,
    ) -> list[CallSession]:
        """
        Get recently completed sessions.
        
        Args:
            limit: Maximum number of sessions to return
            include_active: Include active sessions in results
        """
        async with self._lock:
            if include_active:
                sessions = list(self._sessions.values())
            else:
                sessions = [
                    s for s in self._sessions.values()
                    if s.status in (
                        CallStatus.COMPLETED,
                        CallStatus.FAILED,
                        CallStatus.NO_ANSWER,
                        CallStatus.BUSY,
                        CallStatus.CANCELED,
                    )
                ]
            
            # Sort by end time or initiated time
            sessions.sort(
                key=lambda s: s.ended_at or s.initiated_at,
                reverse=True,
            )
            
            return sessions[:limit]
    
    def _generate_session_id(self, call_id: str) -> str:
        """Generate a triage session ID from call ID."""
        # Hash call_id + timestamp for uniqueness
        data = f"{call_id}:{time.time_ns()}"
        hash_bytes = hashlib.sha256(data.encode()).hexdigest()[:16]
        return f"call_{hash_bytes}"
    
    async def _cleanup_loop(self) -> None:
        """Background task to evict stale sessions."""
        while True:
            try:
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
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in cleanup loop: %s", str(e))


# Global instance (initialized in main.py)
_call_session_store: Optional[CallSessionStore] = None


def get_call_store() -> CallSessionStore:
    """Get the global call session store."""
    global _call_session_store
    if _call_session_store is None:
        _call_session_store = CallSessionStore()
    return _call_session_store


async def init_call_store(
    max_sessions: int = 100,
    session_ttl_minutes: int = 60,
) -> CallSessionStore:
    """Initialize and start the global call session store."""
    global _call_session_store
    _call_session_store = CallSessionStore(
        max_sessions=max_sessions,
        session_ttl_minutes=session_ttl_minutes,
    )
    await _call_session_store.start()
    return _call_session_store
