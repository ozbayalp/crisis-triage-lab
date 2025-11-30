"""
CrisisTriage AI - Exception Hierarchy

Structured exceptions for consistent error handling across the system.
All exceptions include error codes for API responses.
"""

from typing import Optional


class CrisisTriageError(Exception):
    """Base exception for all CrisisTriage errors."""
    
    code: str = "UNKNOWN_ERROR"
    status_code: int = 500
    
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


# =============================================================================
# Pipeline Errors
# =============================================================================

class PipelineError(CrisisTriageError):
    """Error during triage pipeline processing."""
    code = "PIPELINE_ERROR"
    status_code = 500


class TranscriptionError(PipelineError):
    """Error during audio transcription."""
    code = "TRANSCRIPTION_ERROR"


class ProsodyExtractionError(PipelineError):
    """Error during prosody feature extraction."""
    code = "PROSODY_EXTRACTION_ERROR"


class TriageModelError(PipelineError):
    """Error during triage model inference."""
    code = "TRIAGE_MODEL_ERROR"


class ModelNotLoadedError(TriageModelError):
    """Model not loaded or unavailable."""
    code = "MODEL_NOT_LOADED"
    status_code = 503


# =============================================================================
# Session Errors
# =============================================================================

class SessionError(CrisisTriageError):
    """Error related to session management."""
    code = "SESSION_ERROR"
    status_code = 400


class SessionNotFoundError(SessionError):
    """Session not found."""
    code = "SESSION_NOT_FOUND"
    status_code = 404


class SessionLimitError(SessionError):
    """Maximum concurrent sessions exceeded."""
    code = "SESSION_LIMIT_EXCEEDED"
    status_code = 429


class SessionExpiredError(SessionError):
    """Session has expired."""
    code = "SESSION_EXPIRED"
    status_code = 410


# =============================================================================
# Telephony Errors
# =============================================================================

class TelephonyError(CrisisTriageError):
    """Error in telephony subsystem."""
    code = "TELEPHONY_ERROR"
    status_code = 502


class TelephonyDisabledError(TelephonyError):
    """Telephony integration is disabled."""
    code = "TELEPHONY_DISABLED"
    status_code = 503


class CallNotFoundError(TelephonyError):
    """Call session not found."""
    code = "CALL_NOT_FOUND"
    status_code = 404


class CallLimitError(TelephonyError):
    """Maximum concurrent calls exceeded."""
    code = "CALL_LIMIT_EXCEEDED"
    status_code = 429


class AudioProcessingError(TelephonyError):
    """Error processing audio stream."""
    code = "AUDIO_PROCESSING_ERROR"
    status_code = 500


# =============================================================================
# Validation Errors
# =============================================================================

class ValidationError(CrisisTriageError):
    """Input validation error."""
    code = "VALIDATION_ERROR"
    status_code = 400


class InvalidAudioFormatError(ValidationError):
    """Invalid audio format."""
    code = "INVALID_AUDIO_FORMAT"


class InvalidMessageError(ValidationError):
    """Invalid message format."""
    code = "INVALID_MESSAGE"


# =============================================================================
# Configuration Errors
# =============================================================================

class ConfigurationError(CrisisTriageError):
    """Configuration error."""
    code = "CONFIGURATION_ERROR"
    status_code = 500
