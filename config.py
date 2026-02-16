"""Centralised application configuration using Pydantic BaseSettings.

All environment variables are declared once here and validated at startup.
A module-level ``settings`` singleton is exported for use across the codebase.
"""

from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings – loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Database ──────────────────────────────────────────────────────────
    DATABASE_URL: str

    # ── JWT / Auth ────────────────────────────────────────────────────────
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_DAYS: int = 7

    @field_validator("JWT_SECRET_KEY")
    @classmethod
    def _jwt_secret_min_length(cls, v: str) -> str:
        if len(v) < 32:
            raise ValueError("JWT_SECRET_KEY must be at least 32 characters long")
        return v

    # ── Embedding service ─────────────────────────────────────────────────
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_NORMALIZE: bool = True
    EMBEDDING_CACHE_SIZE: int = 1000

    # ── OpenAI (optional) ─────────────────────────────────────────────────
    OPENAI_EMBED_MODEL: Optional[str] = None

    # ── Frontend / CORS ───────────────────────────────────────────────────
    FRONTEND_ORIGIN: str = "http://localhost:5173"


settings = Settings()
