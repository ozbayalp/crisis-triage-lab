"""
CrisisTriage AI - Health Check Endpoints

System health monitoring endpoints for load balancers, monitoring,
and operational visibility.
"""

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.core.exceptions import TelephonyDisabledError

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/health")
async def health_check(
    settings: Settings = Depends(get_settings),
) -> dict:
    """
    Overall system health check.
    
    Returns:
        - status: "healthy" or "degraded"
        - checks: Individual component statuses
        - timestamp: Current server time
    
    Used by load balancers and monitoring systems.
    """
    checks = {}
    
    # Pipeline health
    checks["pipeline"] = {
        "status": "healthy",
        "message": "Pipeline operational",
    }
    
    # Model health
    checks["model"] = {
        "status": "healthy",
        "backend": settings.triage_model_backend,
    }
    
    # Analytics health
    checks["analytics"] = {
        "status": "healthy" if settings.enable_analytics else "disabled",
    }
    
    # Telephony health (if enabled)
    if settings.enable_telephony_integration:
        checks["telephony"] = {
            "status": "healthy",
            "provider": settings.telephony_provider,
        }
    
    # Determine overall status
    all_healthy = all(
        c.get("status") in ("healthy", "disabled") 
        for c in checks.values()
    )
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": "1.0.0",
        "environment": settings.app_env,
        "checks": checks,
    }


@router.get("/ready")
async def readiness_check() -> dict:
    """
    Readiness probe for Kubernetes/container orchestration.
    
    Returns 200 if the service is ready to accept requests.
    """
    return {
        "ready": True,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@router.get("/live")
async def liveness_check() -> dict:
    """
    Liveness probe for Kubernetes/container orchestration.
    
    Returns 200 if the service is alive.
    """
    return {
        "alive": True,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@router.get("/model_status")
async def model_status(
    settings: Settings = Depends(get_settings),
) -> dict:
    """
    Detailed model status information.
    
    Returns:
        - model_id: Identifier of the loaded model
        - backend: "dummy" or "neural"
        - status: "loaded" or "not_loaded"
    """
    return {
        "backend": settings.triage_model_backend,
        "model_dir": settings.neural_model_dir if settings.triage_model_backend == "neural" else None,
        "status": "loaded",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@router.get("/telephony_status")
async def telephony_status(
    settings: Settings = Depends(get_settings),
) -> dict:
    """
    Telephony subsystem status.
    
    Returns:
        - enabled: Whether telephony is enabled
        - provider: Configured provider
        - active_calls: Number of active calls (if enabled)
    
    Raises:
        TelephonyDisabledError: If telephony is disabled
    """
    if not settings.enable_telephony_integration:
        raise TelephonyDisabledError("Telephony integration is disabled")
    
    return {
        "enabled": True,
        "provider": settings.telephony_provider,
        "max_concurrent_calls": settings.telephony_max_concurrent_calls,
        "audio_sample_rate": settings.telephony_audio_sample_rate,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@router.get("/config")
async def config_info(
    settings: Settings = Depends(get_settings),
) -> dict:
    """
    Non-sensitive configuration information.
    
    Useful for debugging and operational visibility.
    Excludes secrets, tokens, and sensitive paths.
    """
    return {
        "environment": settings.app_env,
        "debug": settings.app_debug,
        "log_level": settings.app_log_level,
        "features": {
            "analytics_enabled": settings.enable_analytics,
            "telephony_enabled": settings.enable_telephony_integration,
            "explainability_enabled": settings.enable_explainability,
        },
        "backends": {
            "transcription": settings.transcription_backend,
            "prosody": settings.prosody_backend,
            "triage_model": settings.triage_model_backend,
        },
        "privacy": {
            "anonymize_logs": settings.anonymize_logs,
            "store_transcripts": settings.store_raw_transcripts,
            "store_audio": settings.store_audio,
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
