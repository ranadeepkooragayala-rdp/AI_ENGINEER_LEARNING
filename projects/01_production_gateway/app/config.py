from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Enterprise AI Gateway"
    environment: str = "development"
    openai_api_key: str = "mock-openai-key"
    openai_timeout_seconds: float = 30.0
    openai_max_retries: int = 2

    # Cost calculation constants for gpt-4o-mini
    input_token_price: float = 0.15 / 1_000_000
    output_token_price: float = 0.60 / 1_000_000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()