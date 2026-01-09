"""Resilient Snowflake session management with automatic token refresh for SPCS.

This module provides:
1. ResilientSnowflakeSession: Wrapper with automatic reconnection on token expiration
2. Session factory functions using token_file_path for dynamic token refresh
3. Health check utilities for validating session connectivity

Key insight: SPCS automatically refreshes the token file at /snowflake/session/token
every few minutes. By using token_file_path instead of token, the connector reads
the fresh token on each connection, eliminating 390114 token expiration errors.

Reference: https://docs.snowflake.com/en/developer-guide/snowpark-container-services/additional-considerations-services-jobs
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import snowflake.connector
from snowflake.snowpark import Session

from src.config import is_running_in_spcs, settings

if TYPE_CHECKING:
    from snowflake.connector import SnowflakeConnection

logger = logging.getLogger(__name__)

# SPCS OAuth token path - refreshed automatically by SPCS every few minutes
SPCS_TOKEN_PATH = "/snowflake/session/token"

# Token expiration error codes
TOKEN_EXPIRED_ERRORS = {
    "390114",  # Authentication token has expired
    "390303",  # OAUTH_ACCESS_TOKEN_INVALID
    "08001",   # Connection error (often accompanies 390114)
}

T = TypeVar("T")


def _is_token_expired_error(error: Exception) -> bool:
    """Check if exception is a token expiration error.
    
    Args:
        error: Exception to check.
        
    Returns:
        True if this is a token expiration error that can be resolved by reconnecting.
    """
    error_str = str(error).lower()
    return any(code in error_str for code in TOKEN_EXPIRED_ERRORS) or \
           "token" in error_str and ("expired" in error_str or "invalid" in error_str)


# =============================================================================
# SPCS Session Factory Functions (using token_file_path)
# =============================================================================


def create_spcs_snowpark_session() -> Session:
    """Create Snowpark session for SPCS using token_file_path for dynamic refresh.
    
    Uses token_file_path instead of token so the connector reads fresh tokens
    automatically from the file that SPCS refreshes every few minutes.
    
    Returns:
        Active Snowpark Session.
        
    Raises:
        ValueError: If SPCS environment variables are not set.
    """
    host = os.getenv("SNOWFLAKE_HOST")
    if not host:
        raise ValueError("SNOWFLAKE_HOST not set in SPCS environment")
    
    token_path = Path(SPCS_TOKEN_PATH)
    if not token_path.exists():
        raise ValueError(f"SPCS token file not found: {SPCS_TOKEN_PATH}")
    
    logger.info("Creating Snowpark session with token_file_path (SPCS auto-refresh enabled)")
    
    return Session.builder.configs({
        "account": host.replace(".snowflakecomputing.com", ""),
        "host": host,
        "authenticator": "oauth",
        "token_file_path": SPCS_TOKEN_PATH,  # Dynamic refresh!
        "database": settings.snowflake_database,
        "warehouse": settings.snowflake_warehouse,
    }).create()


def create_spcs_connector_connection() -> SnowflakeConnection:
    """Create snowflake.connector connection for SPCS using token_file_path.
    
    Uses token_file_path instead of token so the connector reads fresh tokens
    automatically from the file that SPCS refreshes every few minutes.
    
    Returns:
        Active SnowflakeConnection.
        
    Raises:
        ValueError: If SPCS environment variables are not set.
    """
    host = os.getenv("SNOWFLAKE_HOST")
    if not host:
        raise ValueError("SNOWFLAKE_HOST not set in SPCS environment")
    
    token_path = Path(SPCS_TOKEN_PATH)
    if not token_path.exists():
        raise ValueError(f"SPCS token file not found: {SPCS_TOKEN_PATH}")
    
    logger.info("Creating Snowflake connector with token_file_path (SPCS auto-refresh enabled)")
    
    return snowflake.connector.connect(
        account=host.replace(".snowflakecomputing.com", ""),
        host=host,
        authenticator="oauth",
        token_file_path=SPCS_TOKEN_PATH,  # Dynamic refresh!
        database=settings.snowflake_database,
        warehouse=settings.snowflake_warehouse,
    )


def create_local_snowpark_session() -> Session:
    """Create Snowpark session for local development with key-pair auth.
    
    Returns:
        Active Snowpark Session.
        
    Raises:
        ValueError: If required credentials are missing.
    """
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization
    
    if not settings.snowflake_private_key_path:
        raise ValueError("SNOWFLAKE_PRIVATE_KEY_PATH required for local Snowpark")
    
    with open(settings.snowflake_private_key_path, "rb") as f:
        private_key = serialization.load_pem_private_key(
            f.read(),
            password=(
                settings.snowflake_private_key_passphrase.encode()
                if settings.snowflake_private_key_passphrase
                else None
            ),
            backend=default_backend(),
        )
    
    private_key_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    
    logger.info("Creating Snowpark session with key-pair auth (local)")
    
    return Session.builder.configs({
        "account": settings.snowflake_account,
        "user": settings.snowflake_user,
        "private_key": private_key_bytes,
        "database": settings.snowflake_database,
        "warehouse": settings.snowflake_warehouse,
    }).create()


def create_local_connector_connection() -> SnowflakeConnection:
    """Create snowflake.connector connection for local development with key-pair auth.
    
    Returns:
        Active SnowflakeConnection.
        
    Raises:
        ValueError: If required credentials are missing.
    """
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization
    
    if not settings.snowflake_private_key_path:
        raise ValueError("SNOWFLAKE_PRIVATE_KEY_PATH required for local")
    
    with open(settings.snowflake_private_key_path, "rb") as f:
        private_key = serialization.load_pem_private_key(
            f.read(),
            password=(
                settings.snowflake_private_key_passphrase.encode()
                if settings.snowflake_private_key_passphrase
                else None
            ),
            backend=default_backend(),
        )
    
    private_key_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    
    logger.info("Creating Snowflake connector with key-pair auth (local)")
    
    return snowflake.connector.connect(
        account=settings.snowflake_account,
        user=settings.snowflake_user,
        private_key=private_key_bytes,
        database=settings.snowflake_database,
        warehouse=settings.snowflake_warehouse,
    )


# =============================================================================
# Resilient Session Wrapper
# =============================================================================


class ResilientSnowflakeSession:
    """Snowpark session wrapper with automatic reconnection on token expiration.
    
    This class wraps a Snowpark Session and provides:
    1. Automatic reconnection when OAuth tokens expire (error 390114)
    2. Thread-safe session access
    3. Health check/validation methods
    
    Usage:
        resilient = ResilientSnowflakeSession()
        session = resilient.get_session()  # Auto-reconnects if needed
        
        # Or execute with automatic retry on token expiration:
        result = resilient.execute_with_retry(lambda s: s.sql("SELECT 1").collect())
    """
    
    def __init__(
        self,
        session_factory: Callable[[], Session] | None = None,
        max_retries: int = 2,
    ) -> None:
        """Initialize resilient session wrapper.
        
        Args:
            session_factory: Optional custom factory function for creating sessions.
                            If not provided, auto-detects SPCS vs local environment.
            max_retries: Maximum reconnection attempts on token expiration.
        """
        self._factory = session_factory or self._default_session_factory
        self._session: Session | None = None
        self._lock = threading.Lock()
        self._max_retries = max_retries
    
    @staticmethod
    def _default_session_factory() -> Session:
        """Create session based on environment detection."""
        if is_running_in_spcs():
            return create_spcs_snowpark_session()
        return create_local_snowpark_session()
    
    def get_session(self) -> Session:
        """Get the current session, creating one if needed.
        
        Thread-safe session access with lazy initialization.
        
        Returns:
            Active Snowpark Session.
        """
        with self._lock:
            if self._session is None:
                self._session = self._factory()
                logger.info("Created new Snowpark session")
            return self._session
    
    def reconnect(self) -> Session:
        """Force reconnection by closing current session and creating a new one.
        
        Thread-safe reconnection with proper cleanup.
        
        Returns:
            New Snowpark Session.
        """
        with self._lock:
            if self._session is not None:
                try:
                    self._session.close()
                    logger.info("Closed stale Snowpark session")
                except Exception as e:
                    logger.warning(f"Error closing session (ignored): {e}")
            
            self._session = self._factory()
            logger.info("Created new Snowpark session after reconnect")
            return self._session
    
    def execute_with_retry(
        self,
        operation: Callable[[Session], T],
        max_retries: int | None = None,
    ) -> T:
        """Execute operation with automatic retry on token expiration.
        
        If a token expiration error occurs, reconnects and retries the operation.
        
        Args:
            operation: Function that takes a Session and returns a result.
            max_retries: Override default max retries.
            
        Returns:
            Result of the operation.
            
        Raises:
            Exception: If operation fails after all retries.
        """
        retries = max_retries if max_retries is not None else self._max_retries
        last_error: Exception | None = None
        
        for attempt in range(retries + 1):
            try:
                return operation(self.get_session())
            except Exception as e:
                last_error = e
                if _is_token_expired_error(e) and attempt < retries:
                    logger.warning(
                        f"Token expired (attempt {attempt + 1}/{retries + 1}), reconnecting..."
                    )
                    self.reconnect()
                else:
                    raise
        
        # Should not reach here, but satisfy type checker
        raise last_error  # type: ignore[misc]
    
    async def aexecute_with_retry(
        self,
        operation: Callable[[Session], T],
        max_retries: int | None = None,
    ) -> T:
        """Async version of execute_with_retry.
        
        Runs the synchronous operation in a thread pool with retry logic.
        
        Args:
            operation: Sync function that takes a Session and returns a result.
            max_retries: Override default max retries.
            
        Returns:
            Result of the operation.
        """
        return await asyncio.to_thread(
            self.execute_with_retry, operation, max_retries
        )
    
    def validate_session(self) -> bool:
        """Validate the current session is still active.
        
        Runs a simple query to verify connectivity.
        
        Returns:
            True if session is valid, False otherwise.
        """
        try:
            session = self.get_session()
            session.sql("SELECT 1").collect()
            return True
        except Exception as e:
            logger.warning(f"Session validation failed: {e}")
            return False
    
    async def avalidate_session(self) -> bool:
        """Async version of validate_session."""
        return await asyncio.to_thread(self.validate_session)
    
    def close(self) -> None:
        """Close the session and release resources."""
        with self._lock:
            if self._session is not None:
                try:
                    self._session.close()
                    logger.info("Closed Snowpark session")
                except Exception as e:
                    logger.warning(f"Error closing session: {e}")
                finally:
                    self._session = None


class ResilientSnowflakeConnector:
    """Snowflake connector wrapper with automatic reconnection on token expiration.
    
    Similar to ResilientSnowflakeSession but for snowflake.connector connections.
    Used by the checkpointer and other components that need raw connector access.
    """
    
    def __init__(
        self,
        connection_factory: Callable[[], SnowflakeConnection] | None = None,
        max_retries: int = 2,
    ) -> None:
        """Initialize resilient connector wrapper.
        
        Args:
            connection_factory: Optional custom factory for creating connections.
            max_retries: Maximum reconnection attempts.
        """
        self._factory = connection_factory or self._default_connection_factory
        self._connection: SnowflakeConnection | None = None
        self._lock = threading.Lock()
        self._max_retries = max_retries
    
    @staticmethod
    def _default_connection_factory() -> SnowflakeConnection:
        """Create connection based on environment detection."""
        if is_running_in_spcs():
            return create_spcs_connector_connection()
        return create_local_connector_connection()
    
    def get_connection(self) -> SnowflakeConnection:
        """Get the current connection, creating one if needed."""
        with self._lock:
            if self._connection is None or self._connection.is_closed():
                self._connection = self._factory()
                logger.info("Created new Snowflake connection")
            return self._connection
    
    def reconnect(self) -> SnowflakeConnection:
        """Force reconnection."""
        with self._lock:
            if self._connection is not None:
                try:
                    self._connection.close()
                except Exception as e:
                    logger.warning(f"Error closing connection (ignored): {e}")
            
            self._connection = self._factory()
            logger.info("Created new Snowflake connection after reconnect")
            return self._connection
    
    def execute_with_retry(
        self,
        operation: Callable[[SnowflakeConnection], T],
        max_retries: int | None = None,
    ) -> T:
        """Execute operation with automatic retry on token expiration."""
        retries = max_retries if max_retries is not None else self._max_retries
        last_error: Exception | None = None
        
        for attempt in range(retries + 1):
            try:
                return operation(self.get_connection())
            except Exception as e:
                last_error = e
                if _is_token_expired_error(e) and attempt < retries:
                    logger.warning(
                        f"Token expired (attempt {attempt + 1}/{retries + 1}), reconnecting..."
                    )
                    self.reconnect()
                else:
                    raise
        
        raise last_error  # type: ignore[misc]
    
    def close(self) -> None:
        """Close the connection."""
        with self._lock:
            if self._connection is not None:
                try:
                    self._connection.close()
                except Exception:
                    pass
                finally:
                    self._connection = None


# =============================================================================
# Health Check Utilities
# =============================================================================


async def check_snowflake_health() -> dict[str, Any]:
    """Comprehensive Snowflake health check.
    
    Validates:
    1. Token file exists and is readable (SPCS only)
    2. Session can be created
    3. Simple query executes successfully
    
    Returns:
        Dictionary with health status details.
    """
    result: dict[str, Any] = {
        "status": "healthy",
        "is_spcs": is_running_in_spcs(),
        "checks": {},
    }
    
    # Check 1: Token file (SPCS only)
    if is_running_in_spcs():
        token_path = Path(SPCS_TOKEN_PATH)
        try:
            if token_path.exists():
                # Check file is readable and non-empty
                token_content = token_path.read_text().strip()
                result["checks"]["token_file"] = {
                    "status": "ok",
                    "exists": True,
                    "readable": True,
                    "has_content": len(token_content) > 0,
                }
            else:
                result["checks"]["token_file"] = {
                    "status": "error",
                    "exists": False,
                    "error": "Token file not found",
                }
                result["status"] = "unhealthy"
        except Exception as e:
            result["checks"]["token_file"] = {
                "status": "error",
                "error": str(e),
            }
            result["status"] = "unhealthy"
    
    # Check 2: Session creation and connectivity
    try:
        def _test_query() -> bool:
            if is_running_in_spcs():
                session = create_spcs_snowpark_session()
            else:
                session = create_local_snowpark_session()
            try:
                session.sql("SELECT 1 AS health_check").collect()
                return True
            finally:
                session.close()
        
        success = await asyncio.to_thread(_test_query)
        result["checks"]["session"] = {
            "status": "ok" if success else "error",
            "can_connect": success,
            "query_executed": success,
        }
        if not success:
            result["status"] = "unhealthy"
            
    except Exception as e:
        error_msg = str(e)
        is_token_error = _is_token_expired_error(e)
        result["checks"]["session"] = {
            "status": "error",
            "error": error_msg,
            "is_token_expiration": is_token_error,
        }
        result["status"] = "unhealthy"
        
        # Add specific guidance for token errors
        if is_token_error:
            result["checks"]["session"]["hint"] = (
                "Token expired. If using static token, migrate to token_file_path. "
                "SPCS automatically refreshes the token file."
            )
    
    return result


# =============================================================================
# Global Singleton Instance
# =============================================================================

# Global resilient session instance - lazily initialized
_global_resilient_session: ResilientSnowflakeSession | None = None
_global_lock = threading.Lock()


def get_resilient_session() -> ResilientSnowflakeSession:
    """Get or create the global resilient session instance.
    
    This provides a singleton pattern for the resilient session,
    ensuring session reuse across the application.
    
    Returns:
        Global ResilientSnowflakeSession instance.
    """
    global _global_resilient_session
    
    if _global_resilient_session is None:
        with _global_lock:
            if _global_resilient_session is None:
                _global_resilient_session = ResilientSnowflakeSession()
    
    return _global_resilient_session


def set_global_session(session: Session) -> None:
    """Set an externally-created session as the global session.
    
    This allows main.py to create a session during startup and share it
    with the resilient wrapper.
    
    Args:
        session: Pre-created Snowpark Session to use.
    """
    global _global_resilient_session
    
    with _global_lock:
        if _global_resilient_session is None:
            _global_resilient_session = ResilientSnowflakeSession()
        
        # Inject the session directly (bypassing factory)
        with _global_resilient_session._lock:
            _global_resilient_session._session = session
        
        logger.info("Global resilient session initialized with pre-created session")

