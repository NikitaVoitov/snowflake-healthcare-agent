"""ChatSnowflake LLM service for healthcare agent - FULLY ASYNC.

Provides async factory functions to create chat model instances with appropriate
authentication for both SPCS (OAuth) and local (key pair) environments.

Both environments use langchain-snowflake ChatSnowflake, but with patched versions:
1. rest_client.py: Uses SNOWFLAKE_HOST env var (required for SPCS OAuth)
2. tools.py: Extracts text from content_list to avoid duplication (issue #35)

SPCS MODE: Session-only auth triggers correct 'Snowflake Token="{token}"' format.
LOCAL MODE: Key pair auth with private_key_path for REST API tool calling.

STREAMING:
- By default, streaming is disabled (disable_streaming=True) to avoid
  "No generations found in stream" errors when tools are bound.
- To enable streaming (after applying langchain-snowflake patches), set:
  ENABLE_LLM_STREAMING=true
- See patches/README.md for how to apply the streaming patches.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_snowflake import ChatSnowflake

from src.config import settings

# Check if streaming should be enabled (requires langchain-snowflake patches)
# Set ENABLE_LLM_STREAMING=true to enable token-level streaming
ENABLE_LLM_STREAMING = os.getenv("ENABLE_LLM_STREAMING", "false").lower() == "true"

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from snowflake.snowpark import Session

logger = logging.getLogger(__name__)

# SPCS OAuth token path (for validation)
SPCS_TOKEN_PATH = Path("/snowflake/session/token")

# Cache for Snowpark session (create once, reuse)
_cached_session: Session | None = None
_session_lock = asyncio.Lock()


def set_session(session: Session) -> None:
    """Set the cached session for ChatSnowflake to use.

    In SPCS mode, main.py creates the Snowpark session during startup
    and must call this function to make it available to ChatSnowflake.
    """
    global _cached_session
    _cached_session = session
    logger.info("Snowpark session cached for ChatSnowflake")


def get_session() -> Session | None:
    """Get the cached Snowpark session."""
    return _cached_session


async def get_chat_snowflake(
    tools: list[BaseTool] | None = None,
    model: str | None = None,
    temperature: float = 0,
) -> BaseChatModel:
    """Get ChatSnowflake instance with optional tool binding (ASYNC).

    Automatically detects environment and uses appropriate authentication:
    - SPCS: Session-only auth (triggers correct Snowflake Token format)
    - Local: Key pair auth with private_key_path for REST API

    Args:
        tools: Optional list of tools to bind to the model.
        model: Cortex model name (default: from settings.cortex_llm_model).
        temperature: Model temperature (default: 0 for deterministic output).

    Returns:
        ChatSnowflake instance, optionally with tools bound.

    Raises:
        ValueError: If required authentication parameters are missing.
    """
    model_name = model or settings.cortex_llm_model

    # Create model based on environment
    if settings.is_spcs:
        llm = await _create_spcs_chat_model(model_name, temperature)
    else:
        llm = await _create_local_chat_model(model_name, temperature)

    # Bind tools if provided
    if tools:
        llm = llm.bind_tools(tools)
        logger.debug("Bound %d tools to ChatSnowflake", len(tools))

    logger.info(
        "ChatSnowflake initialized: model=%s, tools=%d, spcs=%s",
        model_name,
        len(tools or []),
        settings.is_spcs,
    )
    return llm


def _get_session_info(session: Session) -> tuple[str | None, str | None]:
    """Extract account and user from an existing session.

    Args:
        session: Active Snowpark session

    Returns:
        Tuple of (account, user) extracted from session
    """
    try:
        account = session.get_current_account()
        user = session.get_current_user()

        # Remove quotes if present (Snowflake returns quoted identifiers)
        if account and account.startswith('"') and account.endswith('"'):
            account = account[1:-1]
        if user and user.startswith('"') and user.endswith('"'):
            user = user[1:-1]

        return account, user
    except Exception as e:
        logger.warning("Failed to extract account/user from session: %s", e)
        return None, None


async def _create_spcs_chat_model(model: str, temperature: float) -> ChatSnowflake:
    """Create ChatSnowflake for SPCS environment using session-only auth.

    IMPORTANT: Pass ONLY session, no explicit auth params.
    This triggers Method 3 in auth_utils.py which uses correct format:
    'Snowflake Token="{rest.token}"' (not Bearer)

    The patched rest_client.py ensures SNOWFLAKE_HOST is used for the URL.

    Args:
        model: Cortex model name.
        temperature: Model temperature.

    Returns:
        ChatSnowflake configured for SPCS.

    Raises:
        ValueError: If no session is available.
    """
    session = _cached_session
    if session is None:
        raise ValueError(
            "No Snowpark session available in SPCS mode. "
            "Ensure main.py calls llm_service.set_session() during startup."
        )

    account, user = _get_session_info(session)
    logger.info(
        "Creating ChatSnowflake for SPCS: account=%s, user=%s, model=%s",
        account,
        user,
        model,
    )

    # Session-only auth: DO NOT pass token, account, user, private_key_path
    # This triggers the correct 'Snowflake Token="{token}"' format in auth_utils.py
    return ChatSnowflake(
        session=session,
        model=model,
        temperature=temperature,
        # Disable streaming by default to avoid "No generations found in stream" error
        # Set ENABLE_LLM_STREAMING=true after applying langchain-snowflake patches
        disable_streaming=not ENABLE_LLM_STREAMING,
    )


def _load_private_key_sync(key_path: str, passphrase: str | None = None) -> bytes:
    """Load and return private key bytes for Snowpark Session (SYNC - for to_thread).

    Snowpark Session requires the private key as DER-encoded bytes,
    not a file path. This function loads and converts the key.

    Args:
        key_path: Path to the RSA private key file (.p8).
        passphrase: Optional passphrase for encrypted keys.

    Returns:
        DER-encoded private key bytes.

    Raises:
        FileNotFoundError: If key file doesn't exist.
        ValueError: If key cannot be loaded.
    """
    from cryptography.hazmat.primitives import serialization

    key_file = Path(key_path).expanduser()
    if not key_file.exists():
        raise FileNotFoundError(f"Private key not found: {key_file}")

    with open(key_file, "rb") as f:
        private_key = serialization.load_pem_private_key(
            f.read(),
            password=passphrase.encode() if passphrase else None,
        )

    return private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _create_snowpark_session_sync(private_key_bytes: bytes) -> Session:
    """Create Snowpark session (SYNC - for to_thread).

    This is the blocking operation that connects to Snowflake.
    """
    from snowflake.snowpark import Session

    session = Session.builder.configs(
        {
            "account": settings.snowflake_account,
            "user": settings.snowflake_user,
            "private_key": private_key_bytes,
            "warehouse": settings.snowflake_warehouse,
            "database": settings.snowflake_database,
            "role": settings.snowflake_role,
        }
    ).create()

    return session


async def _get_or_create_session() -> Session:
    """Get or create a cached Snowpark session (ASYNC with locking)."""
    global _cached_session

    if _cached_session is not None:
        return _cached_session

    async with _session_lock:
        # Double-check after acquiring lock
        if _cached_session is not None:
            return _cached_session

        logger.debug("Creating new Snowpark session...")

        # Load private key in thread pool
        private_key_bytes = await asyncio.to_thread(
            _load_private_key_sync,
            settings.snowflake_private_key_path,
            settings.snowflake_private_key_passphrase,
        )

        # Create session in thread pool (blocking network call)
        _cached_session = await asyncio.to_thread(
            _create_snowpark_session_sync,
            private_key_bytes,
        )

        logger.info("Snowpark session created and cached")
        return _cached_session


async def _create_local_chat_model(model: str, temperature: float) -> ChatSnowflake:
    """Create ChatSnowflake for local development with key pair auth (ASYNC).

    Uses key pair authentication for local development, which requires:
    - SNOWFLAKE_ACCOUNT
    - SNOWFLAKE_USER
    - SNOWFLAKE_PRIVATE_KEY_PATH
    - Optional: SNOWFLAKE_PRIVATE_KEY_PASSPHRASE

    For tool calling (REST API), we pass the private_key_path so ChatSnowflake
    can authenticate to the REST endpoint.

    Args:
        model: Cortex model name.
        temperature: Model temperature.

    Returns:
        ChatSnowflake configured for local development.

    Raises:
        ValueError: If required credentials are missing.
    """
    # Validate required credentials
    if not settings.snowflake_account:
        raise ValueError("SNOWFLAKE_ACCOUNT required for local development")
    if not settings.snowflake_user:
        raise ValueError("SNOWFLAKE_USER required for local development")
    if not settings.snowflake_private_key_path:
        raise ValueError("SNOWFLAKE_PRIVATE_KEY_PATH required for local development")

    logger.debug(
        "Creating ChatSnowflake for local: account=%s, user=%s",
        settings.snowflake_account,
        settings.snowflake_user,
    )

    # Get or create cached session (async)
    session = await _get_or_create_session()

    logger.debug("Snowpark session ready for ChatSnowflake")

    return ChatSnowflake(
        model=model,
        temperature=temperature,
        session=session,  # For SQL mode
        # For REST API mode (tool calling):
        account=settings.snowflake_account,
        user=settings.snowflake_user,
        private_key_path=settings.snowflake_private_key_path,
        private_key_passphrase=settings.snowflake_private_key_passphrase,
        # Database context
        database=settings.snowflake_database,
        warehouse=settings.snowflake_warehouse,
        # Disable streaming by default to avoid "No generations found in stream" error
        # Set ENABLE_LLM_STREAMING=true after applying langchain-snowflake patches
        disable_streaming=not ENABLE_LLM_STREAMING,
    )


async def get_chat_snowflake_with_structured_output(
    schema: type,
    model: str | None = None,
    temperature: float = 0,
) -> BaseChatModel:
    """Get ChatSnowflake with structured output enforced (ASYNC).

    Uses with_structured_output() to ensure responses conform to a Pydantic schema.

    Args:
        schema: Pydantic model class defining the output structure.
        model: Cortex model name (default: from settings).
        temperature: Model temperature (default: 0).

    Returns:
        ChatSnowflake with structured output configured.
    """
    llm = await get_chat_snowflake(model=model, temperature=temperature)
    return llm.with_structured_output(schema)
