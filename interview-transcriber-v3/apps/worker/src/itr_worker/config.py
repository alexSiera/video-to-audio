"""Worker-specific settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkerSettings(BaseSettings):
    """Settings for the worker service. Reads from env / .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "postgresql+asyncpg://itr:itr@localhost:5432/interview_transcriber"
    log_level: str = "INFO"


@lru_cache
def get_worker_settings() -> WorkerSettings:
    return WorkerSettings()
