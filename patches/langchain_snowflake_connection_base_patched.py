"""
PATCHED: SnowflakeConnectionMixin._get_session() session validation bypass.

BUG DISCOVERED: Session validation causes "Password is empty" error
=======================================================================
When `SnowflakeConnectionMixin._get_session()` is called, it passes the session
to `SnowflakeSessionManager.get_or_create_session()` which runs `SELECT 1` to
validate the session. This can fail due to:

1. Async context issues (session created in different async context)
2. Warehouse not set on the session
3. Session timeout/expiration edge cases

When validation fails, the session manager tries to create a NEW session using
`password=self.password`, which is None when using key-pair authentication,
causing: "Error in create Snowflake session: 251006: Password is empty"

FIX: Trust the provided session directly without running validation queries.
Since the caller explicitly passed a session, we should use it as-is.

APPLIES TO:
    langchain_snowflake/_connection/base.py

To apply this patch:
    cp patches/langchain_snowflake_connection_base_patched.py \\
       original_langchain_snwoflake_repo/langchain-snowflake/libs/snowflake/langchain_snowflake/_connection/base.py
"""

import logging
from typing import Any

from pydantic import Field, SecretStr
from snowflake.snowpark import Session

from .._error_handling import SnowflakeErrorHandler
from .session_manager import SnowflakeSessionManager

logger = logging.getLogger(__name__)


class SnowflakeConnectionMixin:
    """Mixin providing common Snowflake connection functionality.

    This mixin provides:
    - Shared Pydantic field definitions for connection parameters
    - Common session management logic
    - Standardized connection configuration building

    Classes inheriting from this mixin get consistent connection handling
    across the langchain-snowflake package.
    """

    # Shared Pydantic field definitions
    session: Any = Field(default=None, exclude=True, description="Active Snowflake session")
    account: str | None = Field(default=None, description="Snowflake account identifier")
    user: str | None = Field(default=None, description="Snowflake username")
    password: SecretStr | None = Field(default=None, description="Snowflake password")
    token: str | None = Field(default=None, description="OAuth token for authentication")
    private_key_path: str | None = Field(default=None, description="Path to private key file")
    private_key_passphrase: str | None = Field(default=None, description="Private key passphrase")
    warehouse: str | None = Field(default=None, description="Snowflake warehouse")
    database: str | None = Field(default=None, description="Snowflake database")
    schema: str | None = Field(default=None, description="Snowflake schema")

    # Timeout configuration - follows Snowflake parameter hierarchy
    request_timeout: int = Field(default=30, description="HTTP request timeout in seconds (default: 30)")
    respect_session_timeout: bool = Field(
        default=True,
        description="Whether to respect session STATEMENT_TIMEOUT_IN_SECONDS parameter (default: True)",
    )

    # SSL configuration
    verify_ssl: bool = Field(default=True, description="Whether to verify SSL certificates for HTTPS requests")

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the mixin with session caching."""
        super().__init__(**kwargs)
        self._session = None

    def _get_session(self) -> Session:
        """Get or create Snowflake session using centralized session manager.

        This method eliminates 150+ lines of duplicated session creation logic
        across the tools.py file by using the shared SnowflakeSessionManager.

        Returns:
            Active Snowflake session

        Raises:
            ValueError: If session creation fails
        """
        # PATCH: Trust the provided session directly without validation
        # The session manager's test_session_connection can fail due to
        # async context issues or warehouse configuration, causing it to
        # try creating a new session with password auth (which we don't use).
        # Since we explicitly pass a session, trust it and use it directly.
        if self.session is not None:
            self._session = self.session
            return self.session

        # Fall back to session manager for cached or new sessions
        session = SnowflakeSessionManager.get_or_create_session(
            existing_session=None,  # Already handled above
            cached_session=self._session,
            account=self.account,
            user=self.user,
            password=self.password,
            token=getattr(self, "token", None),
            private_key_path=getattr(self, "private_key_path", None),
            private_key_passphrase=getattr(self, "private_key_passphrase", None),
            warehouse=self.warehouse,
            database=self.database,
            schema=self.schema,
        )

        # Cache the session for reuse
        self._session = session
        return session

    def _build_connection_config(self) -> dict:
        """Build connection configuration dictionary from instance attributes.

        Returns:
            Connection configuration dictionary
        """
        return SnowflakeSessionManager.build_connection_config(
            account=self.account,
            user=self.user,
            password=self.password,
            token=getattr(self, "token", None),
            private_key_path=getattr(self, "private_key_path", None),
            private_key_passphrase=getattr(self, "private_key_passphrase", None),
            warehouse=self.warehouse,
            database=self.database,
            schema=self.schema,
        )

    def _get_effective_timeout(self) -> int:
        """Get effective timeout respecting Snowflake parameter hierarchy.

        Returns:
            Effective timeout in seconds following Snowflake best practices:
            1. Session STATEMENT_TIMEOUT_IN_SECONDS (if enabled and > 0)
            2. Tool-specific request_timeout (fallback)
        """
        from .auth_utils import SnowflakeAuthUtils

        session = self._get_session()
        return SnowflakeAuthUtils.get_effective_timeout(
            session=session,
            request_timeout=self.request_timeout,
            respect_session_timeout=self.respect_session_timeout,
        )

    def _count_tokens(self, text: str, model: str = "llama3.1-70b") -> int:
        """Count tokens using official Snowflake CORTEX.COUNT_TOKENS function.

        Args:
            text: Text to count tokens for
            model: Model name to use for tokenization (affects token count)

        Returns:
            Number of tokens according to the specified model's tokenizer

        Note:
            Uses SNOWFLAKE.CORTEX.COUNT_TOKENS for accurate counting.
            Falls back to word-count estimation if Cortex function fails.
        """
        session = self._get_session()

        try:
            # Use official Snowflake tokenizer
            escaped_text = text.replace("'", "''")
            sql = f"""
            SELECT SNOWFLAKE.CORTEX.COUNT_TOKENS(
                '{model}',
                '{escaped_text}'
            ) as token_count
            """

            result = session.sql(sql).collect()

            if result and len(result) > 0:
                token_count = result[0]["TOKEN_COUNT"]
                return int(token_count) if token_count is not None else len(text.split())
            else:
                # Fallback to word count estimation
                return len(text.split())

        except Exception as e:
            # Fallback to word count estimation if Cortex function unavailable
            SnowflakeErrorHandler.log_debug(
                "token counting fallback",
                f"Could not use CORTEX.COUNT_TOKENS, falling back to word count: {e}",
                logger,
            )
            return len(text.split())


