"""Application settings loaded from environment variables."""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration. Reads from env / .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- Application ---
    app_env: str = Field(default="dev", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    web_origin: str = Field(default="http://localhost:3000", alias="WEB_ORIGIN")
    debug: bool = False

    # --- Database ---
    database_url: str = Field(
        default="postgresql+asyncpg://itr:itr@localhost:5432/interview_transcriber",
        alias="DATABASE_URL",
    )

    # --- Redis ---
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # --- Object Storage ---
    s3_endpoint: str = Field(default="http://localhost:9000", alias="S3_ENDPOINT")
    s3_region: str = Field(default="us-east-1", alias="S3_REGION")
    s3_access_key_id: str = Field(default="minioadmin", alias="S3_ACCESS_KEY_ID")
    s3_secret_access_key: str = Field(default="minioadmin", alias="S3_SECRET_ACCESS_KEY")
    s3_bucket_media: str = Field(default="media", alias="S3_BUCKET_MEDIA")
    s3_bucket_artifacts: str = Field(default="artifacts", alias="S3_BUCKET_ARTIFACTS")
    s3_bucket_exports: str = Field(default="exports", alias="S3_BUCKET_EXPORTS")
    s3_use_path_style: bool = Field(default=True, alias="S3_USE_PATH_STYLE")
    s3_presigned_ttl_seconds: int = Field(default=3600, alias="S3_PRESIGNED_TTL_SECONDS")

    # --- JWT ---
    jwt_secret: str = Field(default="change-me", alias="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_access_ttl_seconds: int = Field(default=900, alias="JWT_ACCESS_TTL_SECONDS")
    jwt_refresh_ttl_seconds: int = Field(default=2592000, alias="JWT_REFRESH_TTL_SECONDS")

    # --- Rate limits ---
    rate_limit_login_per_min: int = Field(default=5, alias="RATE_LIMIT_LOGIN_PER_MIN")
    rate_limit_register_per_hour: int = Field(default=3, alias="RATE_LIMIT_REGISTER_PER_HOUR")
    rate_limit_upload_per_min: int = Field(default=10, alias="RATE_LIMIT_UPLOAD_PER_MIN")
    rate_limit_default_per_min: int = Field(default=100, alias="RATE_LIMIT_DEFAULT_PER_MIN")

    # --- Upload ---
    max_upload_bytes: int = Field(default=2_147_483_648, alias="MAX_UPLOAD_BYTES")  # 2 GiB
    max_audio_duration_hours: int = Field(default=10, alias="MAX_AUDIO_DURATION_HOURS")
    upload_part_bytes: int = Field(default=8_388_608, alias="UPLOAD_PART_BYTES")  # 8 MiB

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR"}
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got {v_upper}")
        return v_upper

    @property
    def is_dev(self) -> bool:
        return self.app_env == "dev"

    @property
    def is_prod(self) -> bool:
        return self.app_env == "prod"


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance. Use this in FastAPI dependencies."""
    return Settings()
