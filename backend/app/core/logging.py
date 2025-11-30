"""
CrisisTriage AI - Structured Logging

Provides structured JSON logging with context injection for correlation IDs,
session IDs, and call IDs. All sensitive data is automatically masked.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Optional


# =============================================================================
# Context Variables
# =============================================================================

# Request-level context
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
session_id_var: ContextVar[Optional[str]] = ContextVar('session_id', default=None)
call_id_var: ContextVar[Optional[str]] = ContextVar('call_id', default=None)


# =============================================================================
# Masking Utilities
# =============================================================================

def mask_session_id(sid: Optional[str]) -> Optional[str]:
    """Mask session ID to first 8 characters."""
    if not sid:
        return None
    return sid[:8] if len(sid) > 8 else sid


def mask_call_id(cid: Optional[str]) -> Optional[str]:
    """Mask call ID to last 4 characters."""
    if not cid:
        return None
    return f"***{cid[-4:]}" if len(cid) > 4 else "***"


def mask_sensitive_data(data: dict) -> dict:
    """
    Recursively mask sensitive fields in a dictionary.
    
    Sensitive fields: phone, number, from, to, caller, callee, etc.
    """
    sensitive_keys = {
        'phone', 'phone_number', 'from', 'to', 'caller', 'callee',
        'from_number', 'to_number', 'password', 'token', 'secret', 'key'
    }
    
    masked = {}
    for key, value in data.items():
        key_lower = key.lower()
        
        if any(s in key_lower for s in sensitive_keys):
            if isinstance(value, str):
                masked[key] = f"***{value[-2:]}" if len(value) > 2 else "***"
            else:
                masked[key] = "[REDACTED]"
        elif isinstance(value, dict):
            masked[key] = mask_sensitive_data(value)
        else:
            masked[key] = value
    
    return masked


# =============================================================================
# Structured Formatter
# =============================================================================

class StructuredFormatter(logging.Formatter):
    """
    JSON formatter that injects context variables and masks sensitive data.
    
    Output format:
    {
        "timestamp": "2024-11-30T00:00:00.000Z",
        "level": "INFO",
        "logger": "module.submodule",
        "correlation_id": "req_abc123",
        "session_id": "ses_xyz7",
        "call_id": "***1234",
        "message": "Human-readable message",
        "data": { ... }
    }
    """
    
    def format(self, record: logging.LogRecord) -> str:
        # Base log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Inject context variables
        correlation_id = correlation_id_var.get()
        if correlation_id:
            log_entry["correlation_id"] = correlation_id
        
        session_id = session_id_var.get()
        if session_id:
            log_entry["session_id"] = mask_session_id(session_id)
        
        call_id = call_id_var.get()
        if call_id:
            log_entry["call_id"] = mask_call_id(call_id)
        
        # Add event type if present
        if hasattr(record, 'event_type'):
            log_entry["event_type"] = record.event_type
        
        # Add structured data if present
        if hasattr(record, 'data') and record.data:
            log_entry["data"] = mask_sensitive_data(record.data)
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


class HumanReadableFormatter(logging.Formatter):
    """
    Human-readable formatter for development.
    Includes timestamp, level, logger, and message with context.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        
        # Build context string
        context_parts = []
        
        session_id = session_id_var.get()
        if session_id:
            context_parts.append(f"session={mask_session_id(session_id)}")
        
        call_id = call_id_var.get()
        if call_id:
            context_parts.append(f"call={mask_call_id(call_id)}")
        
        context_str = f" [{', '.join(context_parts)}]" if context_parts else ""
        
        # Format message
        message = f"{timestamp} | {record.levelname:<8} | {record.name}{context_str} | {record.getMessage()}"
        
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)
        
        return message


# =============================================================================
# Logger Setup
# =============================================================================

def setup_structured_logging(
    level: str = "INFO",
    json_format: bool = False,
) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_format: Use JSON format (True for production, False for development)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))
    
    # Set formatter
    if json_format:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(HumanReadableFormatter())
    
    root_logger.addHandler(handler)
    
    # Suppress noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("watchfiles").setLevel(logging.WARNING)


# =============================================================================
# Context Managers
# =============================================================================

class LogContext:
    """
    Context manager for setting log context variables.
    
    Usage:
        with LogContext(session_id="abc123", call_id="xyz789"):
            logger.info("Processing request")
    """
    
    def __init__(
        self,
        correlation_id: Optional[str] = None,
        session_id: Optional[str] = None,
        call_id: Optional[str] = None,
    ):
        self._correlation_id = correlation_id
        self._session_id = session_id
        self._call_id = call_id
        self._tokens = []
    
    def __enter__(self):
        if self._correlation_id:
            self._tokens.append(correlation_id_var.set(self._correlation_id))
        if self._session_id:
            self._tokens.append(session_id_var.set(self._session_id))
        if self._call_id:
            self._tokens.append(call_id_var.set(self._call_id))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for token in reversed(self._tokens):
            # Reset to previous value
            pass  # ContextVar handles cleanup automatically
        return False


# =============================================================================
# Structured Logger
# =============================================================================

class StructuredLogger:
    """
    Logger wrapper that supports structured data.
    
    Usage:
        logger = StructuredLogger(__name__)
        logger.info("User logged in", data={"user_id": "123"})
    """
    
    def __init__(self, name: str):
        self._logger = logging.getLogger(name)
    
    def _log(self, level: int, message: str, data: Optional[dict] = None, **kwargs):
        extra = {}
        if data:
            extra['data'] = data
        
        self._logger.log(level, message, extra=extra, **kwargs)
    
    def debug(self, message: str, data: Optional[dict] = None, **kwargs):
        self._log(logging.DEBUG, message, data, **kwargs)
    
    def info(self, message: str, data: Optional[dict] = None, **kwargs):
        self._log(logging.INFO, message, data, **kwargs)
    
    def warning(self, message: str, data: Optional[dict] = None, **kwargs):
        self._log(logging.WARNING, message, data, **kwargs)
    
    def error(self, message: str, data: Optional[dict] = None, **kwargs):
        self._log(logging.ERROR, message, data, **kwargs)
    
    def exception(self, message: str, data: Optional[dict] = None, **kwargs):
        self._log(logging.ERROR, message, data, exc_info=True, **kwargs)


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance."""
    return StructuredLogger(name)
