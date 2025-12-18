"""Configuration module using pydantic-settings for Healthcare Multi-Agent Lab."""

import os
from typing import Self

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def is_running_in_spcs() -> bool:
    """Detect if running inside Snowpark Container Services.

    Returns:
        True if running in SPCS environment (OAuth token or SNOWFLAKE_HOST present).
    """
    return os.environ.get("SNOWFLAKE_HOST") is not None or os.path.exists(
        "/snowflake/session/token"
    )


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Validates that either SPCS credentials (auto-detected) or local credentials
    (account, user, private_key_path) are properly configured.
    """

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

    # CORS settings (configure for production)
    cors_origins: str = "*"

    @model_validator(mode="after")
    def validate_snowflake_config(self) -> Self:
        """Validate Snowflake credentials based on environment.

        Note: We detect SPCS here (not in model_post_init) because validators
        run BEFORE model_post_init in Pydantic v2.

        Raises:
            ValueError: If local mode and required credentials are missing.

        Returns:
            Validated Settings instance.
        """
        # Detect SPCS environment first (must happen in validator, not model_post_init)
        spcs_detected = is_running_in_spcs()
        object.__setattr__(self, "is_spcs", spcs_detected)

        # Only require local credentials when NOT in SPCS
        if not spcs_detected:
            missing: list[str] = []
            if not self.snowflake_account:
                missing.append("SNOWFLAKE_ACCOUNT")
            if not self.snowflake_user:
                missing.append("SNOWFLAKE_USER")
            if not self.snowflake_private_key_path:
                missing.append("SNOWFLAKE_PRIVATE_KEY_PATH")
            if missing:
                raise ValueError(f"Local mode requires environment variables: {', '.join(missing)}")
        return self


# Singleton instance
settings = Settings()
