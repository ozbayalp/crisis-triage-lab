"""
CrisisTriage AI - Configuration Management

Centralized configuration using Pydantic Settings.
All secrets and environment-specific values are loaded from environment variables.
"""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Hierarchy (highest to lowest priority):
    1. Environment variables
    2. .env file
    3. Default values
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # --- Application ---
    app_env: str = "development"
    app_debug: bool = True
    app_log_level: str = "INFO"
    
    # --- Server ---
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    backend_workers: int = 1
    
    # --- Model Paths ---
    whisper_model_size: str = "base"
    whisper_model_path: str = "./models/whisper"
    triage_model_path: str = "./models/triage/latest.pt"
    
    # --- Transcription Service ---
    # "dummy" = placeholder responses (default, no ML dependencies)
    # "whisper" = OpenAI Whisper local inference (requires openai-whisper)
    transcription_backend: str = "dummy"
    whisper_model_name: str = "base"  # tiny, base, small, medium, large
    whisper_language: str = "en"  # Force English for consistency
    
    # --- Prosody Extraction ---
    # "dummy" = placeholder features (default, no ML dependencies)
    # "librosa" = Real extraction using librosa (requires librosa, numpy)
    prosody_backend: str = "dummy"
    
    # --- Triage Model Selection ---
    # "dummy" = keyword-based heuristic (default, no ML dependencies)
    # "neural" = trained neural classifier (requires transformers, torch)
    triage_model_backend: str = "dummy"
    
    # Path to neural model artifact (only used when triage_model_backend="neural")
    # Should contain: model weights, tokenizer, artifact.json
    neural_model_dir: str = "./ml/outputs/baseline/best_model"
    
    # --- Audio Processing ---
    # Minimum audio chunk size in bytes before processing (1 second at 16kHz 16-bit mono = 32000)
    audio_chunk_min_bytes: int = 32000
    # Maximum audio buffer size in bytes (prevent memory issues)
    audio_buffer_max_bytes: int = 320000  # ~10 seconds
    
    # --- Privacy & Data Retention ---
    persist_raw_audio: bool = False
    persist_transcripts: bool = False
    persist_features: bool = False
    session_ttl_seconds: int = 0  # 0 = delete immediately after session
    
    # --- Privacy Controls (Pipeline) ---
    # These control what data is stored/logged during processing
    store_raw_transcripts: bool = False  # If True, transcripts may be stored (respects persist_transcripts)
    store_audio: bool = False            # If True, audio may be stored (respects persist_raw_audio)
    store_prosody_features: bool = False # If True, prosody features may be stored
    anonymize_logs: bool = True          # If True, logs contain minimal identifying info
    
    # --- Explainability ---
    enable_explainability: bool = True   # Include feature attribution in results
    log_explanations: bool = False       # Log detailed explanations (may contain sensitive info)
    
    # --- Analytics & History ---
    enable_analytics: bool = True                    # Enable analytics API and history recording
    store_analytics_text_snippets: bool = False      # Store truncated text in analytics (privacy-sensitive)
    analytics_max_events: int = 10000                # Max events to keep in memory (bounded buffer)
    analytics_example_snippets_per_risk: int = 3     # Max example snippets per risk level
    
    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"
    
    # --- Security ---
    secret_key: str = "CHANGE-ME-IN-PRODUCTION"
    allowed_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    
    @property
    def allowed_origins_list(self) -> List[str]:
        """Parse comma-separated origins into list."""
        return [origin.strip() for origin in self.allowed_origins.split(",")]
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.app_env.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings instance.
    
    Use dependency injection in routes:
        settings: Settings = Depends(get_settings)
    """
    return Settings()


# Convenience export
settings = get_settings()
