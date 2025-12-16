"""Configuration module using pydantic-settings for Healthcare Multi-Agent Lab."""

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


def is_running_in_spcs() -> bool:
    """Detect if running inside Snowpark Container Services."""
    # SPCS containers have specific environment variables set by Snowflake
    return os.environ.get("SNOWFLAKE_HOST") is not None or os.path.exists(
        "/snowflake/session/token"
    )


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Snowflake connection - required for local, optional for SPCS
    snowflake_account: str | None = None
    snowflake_user: str | None = None
    snowflake_private_key_path: str | None = None
    snowflake_private_key_passphrase: str | None = None
    snowflake_database: str = "HEALTHCARE_DB"
    snowflake_schema: str = "PUBLIC"
    snowflake_warehouse: str = "PAYERS_CC_WH"
    snowflake_role: str = "ACCOUNTADMIN"

    # SPCS-specific settings (auto-detected)
    is_spcs: bool = False

    # Agent settings
    max_agent_steps: int = 3
    agent_timeout_seconds: float = 30.0

    # Cortex settings
    cortex_llm_model: str = "llama3.1-70b"
    cortex_search_limit: int = 5

    # LangChain tracing (optional)
    langchain_tracing_v2: bool = False
    langchain_api_key: str | None = None

    def model_post_init(self, _context: object) -> None:
        """Post-initialization to detect SPCS environment."""
        object.__setattr__(self, "is_spcs", is_running_in_spcs())


# Singleton instance
settings = Settings()
