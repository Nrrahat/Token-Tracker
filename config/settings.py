"""Environment-backed application settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from config.constants import DEFAULT_DATABASE_URL, DEFAULT_LOG_LEVEL


class Settings(BaseSettings):
    """Runtime configuration; secrets are intentionally not persisted here."""

    model_config = SettingsConfigDict(
        env_prefix="AOP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    database_url: str = DEFAULT_DATABASE_URL
    log_level: str = DEFAULT_LOG_LEVEL


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()