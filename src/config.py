"""Configuration module using pydantic-settings for Healthcare Multi-Agent Lab."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Snowflake connection
    snowflake_account: str
    snowflake_user: str
    snowflake_private_key_path: str
    snowflake_private_key_passphrase: str
    snowflake_database: str = "HEALTHCARE_DB"
    snowflake_schema: str = "PUBLIC"
    snowflake_warehouse: str = "PAYERS_CC_WH"
    snowflake_role: str = "ACCOUNTADMIN"

    # Agent settings
    max_agent_steps: int = 3
    agent_timeout_seconds: float = 30.0

    # Cortex settings
    cortex_llm_model: str = "llama3.1-70b"
    cortex_search_limit: int = 5

    # LangChain tracing (optional)
    langchain_tracing_v2: bool = False
    langchain_api_key: str | None = None


# Singleton instance
settings = Settings()

