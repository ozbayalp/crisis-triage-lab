"""
CrisisTriage AI - Backend Entrypoint

FastAPI application factory and server configuration.
Run with: uvicorn backend.main:app --reload
"""

from contextlib import asynccontextmanager
import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings, get_settings
from app.api import routes, websocket
from app.core.pipeline import TriagePipeline, create_pipeline

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.app_log_level.upper()),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Startup:
        - Create and initialize the triage pipeline
        - Warm up models for low-latency first inference
        - Validate configuration
    
    Shutdown:
        - Gracefully shutdown pipeline
        - Flush any pending metrics
    """
    # === Startup ===
    logger.info("🚀 CrisisTriage AI starting in %s mode", settings.app_env)
    
    # Create the triage pipeline with configured services
    pipeline = create_pipeline(settings)
    
    # Store pipeline in app state for dependency injection
    app.state.pipeline = pipeline
    app.state.settings = settings
    
    # Warm up the pipeline (loads models, runs dummy inference)
    await pipeline.startup()
    
    logger.info("✅ Pipeline initialized and ready")
    logger.info(
        "   Privacy: anonymize_logs=%s, store_transcripts=%s, store_audio=%s",
        settings.anonymize_logs,
        settings.store_raw_transcripts,
        settings.store_audio,
    )
    logger.info(
        "   Analytics: enabled=%s, store_snippets=%s, max_events=%d",
        settings.enable_analytics,
        settings.store_analytics_text_snippets,
        settings.analytics_max_events,
    )
    
    yield
    
    # === Shutdown ===
    logger.info("👋 CrisisTriage AI shutting down")
    await pipeline.shutdown()
    logger.info("✅ Shutdown complete")


def create_app() -> FastAPI:
    """Application factory."""
    
    app = FastAPI(
        title="CrisisTriage AI",
        description="Real-time triage API for mental health hotline conversations",
        version="0.1.0",
        docs_url="/docs" if settings.app_debug else None,
        redoc_url="/redoc" if settings.app_debug else None,
        lifespan=lifespan,
    )
    
    # --- CORS ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # --- Routes ---
    app.include_router(routes.router, prefix="/api")
    app.include_router(websocket.router)
    
    return app


# Create app instance
app = create_app()


# --- Health check at root ---
@app.get("/")
async def root():
    """Root health check."""
    return {
        "service": "CrisisTriage AI",
        "status": "operational",
        "version": "0.1.0",
    }
