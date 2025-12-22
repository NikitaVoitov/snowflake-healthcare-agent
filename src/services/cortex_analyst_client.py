"""Async client for Snowflake Cortex Analyst REST API.

This module provides an async interface to the Cortex Analyst API for
converting natural language questions to SQL queries using semantic models.

The client supports:
- Single and multi-turn conversations with conversation history
- Streaming and non-streaming responses
- Semantic model files from Snowflake stages
- Feedback submission for monitoring
- Async/non-blocking operations using aiohttp
- SPCS OAuth token authentication
- Local JWT key-pair authentication

Usage:
    from src.services.cortex_analyst_client import AsyncCortexAnalystClient

    client = AsyncCortexAnalystClient(session, database="HEALTHCARE_DB", schema="STAGING")
    response = await client.execute_query("How many members do we have?")
    print(response["results"])
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
import jwt
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from snowflake.snowpark import Session

from src.config import settings

logger = logging.getLogger(__name__)

# SPCS token file path
SPCS_TOKEN_FILE = "/snowflake/session/token"


# =============================================================================
# Response Data Classes
# =============================================================================


@dataclass
class CortexAnalystResponse:
    """Structured response from Cortex Analyst API.

    Attributes:
        request_id: Unique request identifier for tracking/feedback.
        role: Response role (always 'analyst').
        text: Interpretation/explanation text from the model.
        sql: Generated SQL statement (if available).
        suggestions: Alternative questions if query was ambiguous.
        confidence: Confidence metadata including verified query info.
        warnings: List of warning messages.
        model_names: LLM models used for generation.
        raw_response: Full raw API response for debugging.
    """

    request_id: str | None = None
    role: str = "analyst"
    text: str | None = None
    sql: str | None = None
    suggestions: list[str] = field(default_factory=list)
    confidence: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    model_names: list[str] = field(default_factory=list)
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationMessage:
    """A message in the conversation history.

    Attributes:
        role: Either 'user' or 'analyst'.
        content: List of content blocks with type and text/sql.
    """

    role: str
    content: list[dict[str, Any]]


# =============================================================================
# Cortex Analyst Client
# =============================================================================


class AsyncCortexAnalystClient:
    """Async client for Snowflake Cortex Analyst REST API.

    Provides natural language to SQL conversion using semantic models.
    Maintains conversation history for multi-turn interactions.

    Attributes:
        session: Active Snowpark session for token retrieval.
        semantic_model_path: Full stage path to semantic model YAML.
        conversation_history: List of messages for multi-turn support.
    """

    # API endpoint path (appended to account URL)
    API_PATH = "/api/v2/cortex/analyst/message"
    FEEDBACK_PATH = "/api/v2/cortex/analyst/feedback"

    def __init__(
        self,
        session: Session,
        semantic_model_path: str | None = None,
        database: str | None = None,
        schema: str | None = None,
    ) -> None:
        """Initialize Cortex Analyst client.

        Args:
            session: Active Snowpark Session with OAuth token.
            semantic_model_path: Full path to semantic model YAML file.
                Example: "@HEALTHCARE_DB.STAGING.SEMANTIC_MODELS/healthcare_semantic_model.yaml"
            database: Database name (defaults to settings.snowflake_database).
            schema: Schema name (defaults to 'STAGING').
        """
        self.session = session
        self.database = database or settings.snowflake_database
        self.schema = schema or "STAGING"

        # Build semantic model path if not provided
        if semantic_model_path:
            self.semantic_model_path = semantic_model_path
        else:
            self.semantic_model_path = f"@{self.database}.{self.schema}.SEMANTIC_MODELS/healthcare_semantic_model.yaml"

        # Build API URL from session context
        self._api_url: str | None = None
        self._token: str | None = None
        self._token_type: str = "OAUTH"  # 'OAUTH' or 'KEYPAIR_JWT'

        # Conversation history for multi-turn support
        self.conversation_history: list[ConversationMessage] = []

    def _get_api_url(self) -> str:
        """Get or build the Cortex Analyst API URL.

        In SPCS, uses SNOWFLAKE_HOST environment variable.
        Locally, uses the account locator from settings.

        Returns:
            Full API endpoint URL.
        """
        if self._api_url:
            return self._api_url

        # SPCS: Use SNOWFLAKE_HOST environment variable (recommended)
        snowflake_host = os.environ.get("SNOWFLAKE_HOST")
        if snowflake_host:
            self._api_url = f"https://{snowflake_host}{self.API_PATH}"
            logger.info(f"Using SPCS Cortex Analyst API URL: {self._api_url}")
            return self._api_url

        # Local/non-SPCS: Use account locator from settings
        # The format is: https://<account_locator>.snowflakecomputing.com/api/v2/cortex/analyst/message
        account = settings.snowflake_account
        if not account:
            # Fallback: try to get from environment
            account = os.environ.get("SNOWFLAKE_ACCOUNT")

        if not account:
            raise ValueError("Cannot determine Snowflake account. Set SNOWFLAKE_ACCOUNT environment variable.")

        # Use account locator directly (e.g., "LFB71918")
        self._api_url = f"https://{account}.snowflakecomputing.com{self.API_PATH}"
        logger.info(f"Cortex Analyst API URL: {self._api_url}")

        return self._api_url

    def _get_token(self) -> tuple[str, str]:
        """Get authentication token for Cortex Analyst API.

        Authentication priority:
        1. SPCS: OAuth token from /snowflake/session/token file
        2. Local: JWT token from key-pair authentication

        Returns:
            Tuple of (token, token_type) where token_type is 'OAUTH' or 'KEYPAIR_JWT'.

        Raises:
            ValueError: If token cannot be retrieved.
        """
        if self._token:
            return self._token, self._token_type

        # SPCS: Read OAuth token from container file (recommended method)
        token_path = Path(SPCS_TOKEN_FILE)
        if token_path.exists():
            self._token = token_path.read_text().strip()
            self._token_type = "OAUTH"
            logger.debug("Retrieved OAuth token from SPCS container")
            return self._token, self._token_type

        # Local: Try to get OAuth token from Snowpark session internals
        try:
            if hasattr(self.session, "_conn") and hasattr(self.session._conn, "_rest"):
                rest = self.session._conn._rest
                if hasattr(rest, "_token") and rest._token:
                    self._token = rest._token
                    self._token_type = "OAUTH"
                    logger.debug("Retrieved OAuth token from Snowpark session")
                    return self._token, self._token_type
        except Exception as exc:
            logger.debug(f"Could not get OAuth token from session: {exc}")

        # Local fallback: Generate JWT from key-pair authentication
        jwt_token = self._generate_jwt_token()
        if jwt_token:
            self._token = jwt_token
            self._token_type = "KEYPAIR_JWT"
            logger.info("Generated JWT token for key-pair authentication")
            return self._token, self._token_type

        raise ValueError(
            "Cannot retrieve authentication token. In SPCS, ensure container has access to "
            f"{SPCS_TOKEN_FILE}. Locally, ensure SNOWFLAKE_PRIVATE_KEY_PATH is set."
        )

    def _generate_jwt_token(self) -> str | None:
        """Generate JWT token for key-pair authentication.

        Uses the private key configured in settings to generate a JWT
        that can authenticate with Snowflake REST APIs.

        Returns:
            JWT token string or None if key-pair auth not configured.
        """
        # Check if key-pair auth is configured
        key_path_str = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PATH") or getattr(settings, "snowflake_private_key_path", None)
        if not key_path_str:
            logger.debug("No private key path configured, JWT auth unavailable")
            return None

        try:
            key_path = Path(key_path_str).expanduser()
            if not key_path.exists():
                logger.warning(f"Private key file not found: {key_path}")
                return None

            # Load private key - check both env var and settings for passphrase
            passphrase = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE") or getattr(settings, "snowflake_private_key_passphrase", None)
            with open(key_path, "rb") as f:
                private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=passphrase.encode() if passphrase else None,
                    backend=default_backend(),
                )

            # Get account and user
            account = (settings.snowflake_account or "").upper()
            user = (settings.snowflake_user or "").upper()
            if not account or not user:
                logger.warning("SNOWFLAKE_ACCOUNT or SNOWFLAKE_USER not configured")
                return None

            qualified_username = f"{account}.{user}"

            # Compute public key fingerprint
            public_key = private_key.public_key()
            public_key_bytes = public_key.public_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            sha256_hash = hashlib.sha256(public_key_bytes).digest()
            public_key_fp = "SHA256:" + base64.b64encode(sha256_hash).decode()

            # Build JWT payload
            now = int(time.time())
            payload = {
                "iss": f"{qualified_username}.{public_key_fp}",
                "sub": qualified_username,
                "iat": now,
                "exp": now + 3600,  # 1 hour validity
            }

            # Generate JWT
            token = jwt.encode(payload, private_key, algorithm="RS256")
            logger.debug(f"Generated JWT token for {qualified_username}")
            return token

        except Exception as exc:
            logger.error(f"Failed to generate JWT token: {exc}", exc_info=True)
            return None

    def _build_request_body(
        self,
        question: str,
        include_history: bool = True,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Build the request body for Cortex Analyst API.

        Args:
            question: Natural language question from user.
            include_history: Whether to include conversation history.
            stream: Whether to enable streaming response.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        # Build messages array
        messages = []

        # Include conversation history if requested
        if include_history and self.conversation_history:
            for msg in self.conversation_history:
                messages.append(
                    {
                        "role": msg.role,
                        "content": msg.content,
                    }
                )

        # Add current user question
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": question}],
            }
        )

        return {
            "messages": messages,
            "semantic_model_file": self.semantic_model_path,
            "stream": stream,
        }

    def _parse_response(self, response_data: dict[str, Any], request_id: str | None) -> CortexAnalystResponse:
        """Parse API response into structured format.

        Args:
            response_data: Raw JSON response from API.
            request_id: Request ID from response headers.

        Returns:
            Structured CortexAnalystResponse object.
        """
        result = CortexAnalystResponse(
            request_id=request_id or response_data.get("request_id"),
            raw_response=response_data,
        )

        # Parse message content
        message = response_data.get("message", {})
        result.role = message.get("role", "analyst")

        content_blocks = message.get("content", [])
        for block in content_blocks:
            block_type = block.get("type")

            if block_type == "text":
                result.text = block.get("text")

            elif block_type == "sql":
                result.sql = block.get("statement")
                result.confidence = block.get("confidence", {})

            elif block_type == "suggestions":
                result.suggestions = block.get("suggestions", [])

        # Parse warnings
        warnings = response_data.get("warnings", [])
        result.warnings = [w.get("message", "") for w in warnings if w.get("message")]

        # Parse metadata
        metadata = response_data.get("response_metadata", {})
        result.model_names = metadata.get("model_names", [])

        return result

    async def query(
        self,
        question: str,
        clear_history: bool = False,
        timeout: float = 60.0,
    ) -> CortexAnalystResponse:
        """Send a natural language question to Cortex Analyst.

        This method:
        1. Builds the request with optional conversation history
        2. Calls the Cortex Analyst REST API asynchronously
        3. Parses the response and updates conversation history
        4. Returns structured response with SQL and metadata

        Args:
            question: Natural language question about data.
            clear_history: If True, starts a fresh conversation.
            timeout: Request timeout in seconds.

        Returns:
            CortexAnalystResponse with SQL, text, and metadata.

        Raises:
            aiohttp.ClientError: On network/HTTP errors.
            ValueError: On API authentication failures.
        """
        if clear_history:
            self.conversation_history = []

        api_url = self._get_api_url()
        token, token_type = self._get_token()

        # Use Bearer token format with appropriate token type
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Snowflake-Authorization-Token-Type": token_type,
        }

        request_body = self._build_request_body(question, include_history=True, stream=False)

        logger.info(f"Calling Cortex Analyst: {question[:100]}...")
        logger.debug(f"Request body: {json.dumps(request_body, default=str)}")

        try:
            async with (
                aiohttp.ClientSession() as http_session,
                http_session.post(
                    api_url,
                    json=request_body,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as response,
            ):
                request_id = response.headers.get("X-Snowflake-Request-Id")

                if response.status >= 400:
                    error_text = await response.text()
                    logger.error(f"Cortex Analyst API error {response.status}: {error_text}")
                    raise aiohttp.ClientResponseError(
                        response.request_info,
                        response.history,
                        status=response.status,
                        message=f"API error: {error_text}",
                    )

                response_data = await response.json()
                log_msg = json.dumps(response_data, default=str)[:500]
                logger.debug(f"Raw API response: {log_msg}")

        except TimeoutError as exc:
            logger.error(f"Cortex Analyst request timed out after {timeout}s")
            raise TimeoutError(f"Request timed out after {timeout} seconds") from exc

        # Parse response
        result = self._parse_response(response_data, request_id)

        # Update conversation history
        self._update_history(question, result)

        # Log result summary with SQL preview for debugging
        if result.sql:
            logger.info(f"Generated SQL ({len(result.sql)} chars): {result.sql[:200]}...")
        elif result.suggestions:
            logger.info(f"Received {len(result.suggestions)} suggestions (ambiguous query)")
        else:
            logger.warning("No SQL or suggestions in response")

        return result

    def _update_history(self, question: str, response: CortexAnalystResponse) -> None:
        """Update conversation history with the latest exchange.

        Args:
            question: User's original question.
            response: Parsed response from API.
        """
        # Add user message
        self.conversation_history.append(
            ConversationMessage(
                role="user",
                content=[{"type": "text", "text": question}],
            )
        )

        # Add analyst response
        content = []
        if response.text:
            content.append({"type": "text", "text": response.text})
        if response.sql:
            content.append({"type": "sql", "statement": response.sql})
        if response.suggestions:
            content.append({"type": "suggestions", "suggestions": response.suggestions})

        if content:
            self.conversation_history.append(ConversationMessage(role="analyst", content=content))

    async def execute_query(self, question: str, clear_history: bool = False) -> dict[str, Any]:
        """Query Cortex Analyst and execute the generated SQL.

        This is a convenience method that:
        1. Calls Cortex Analyst to get SQL
        2. Executes the SQL using Snowpark
        3. Returns both the SQL and results

        Args:
            question: Natural language question.
            clear_history: Whether to clear conversation history.

        Returns:
            Dictionary with:
                - sql: The generated SQL statement
                - results: List of result dictionaries
                - interpretation: Text explanation from model
                - suggestions: Alternative questions if ambiguous
                - request_id: For feedback submission
                - error: Error message if SQL execution failed
        """
        # Get SQL from Cortex Analyst
        analyst_response = await self.query(question, clear_history=clear_history)

        result: dict[str, Any] = {
            "sql": analyst_response.sql,
            "interpretation": analyst_response.text,
            "suggestions": analyst_response.suggestions,
            "request_id": analyst_response.request_id,
            "warnings": analyst_response.warnings,
            "results": [],
            "error": None,
        }

        # If we got suggestions instead of SQL, return them
        if analyst_response.suggestions and not analyst_response.sql:
            result["error"] = "Question was ambiguous. See suggestions for alternatives."
            return result

        # If no SQL was generated, return error
        if not analyst_response.sql:
            result["error"] = "Could not generate SQL for this question."
            return result

        # Execute the SQL
        try:

            def _execute_sql() -> list[dict]:
                """Execute SQL synchronously (for thread pool)."""
                rows = self.session.sql(analyst_response.sql).collect()
                # Convert rows to dictionaries
                results = []
                for row in rows:
                    if hasattr(row, "as_dict"):
                        results.append(row.as_dict())
                    else:
                        # Fallback for older Snowpark versions
                        results.append(dict(zip(row._fields, row, strict=True)))
                return results

            result["results"] = await asyncio.to_thread(_execute_sql)
            logger.info(f"Executed SQL successfully, returned {len(result['results'])} rows")

        except Exception as exc:
            logger.error(f"SQL execution failed: {exc}")
            result["error"] = f"SQL execution error: {exc!s}"

        return result

    async def send_feedback(
        self,
        request_id: str,
        is_positive: bool,
        feedback_message: str = "",
    ) -> bool:
        """Submit feedback for a Cortex Analyst response.

        Feedback is used by Snowflake to monitor and improve the service.
        Visible in Snowsight under AI & ML > Cortex Analyst > Monitoring.

        Args:
            request_id: The request_id from the original response.
            is_positive: True for thumbs up, False for thumbs down.
            feedback_message: Optional text feedback.

        Returns:
            True if feedback was submitted successfully.
        """
        api_url = self._get_api_url().replace(self.API_PATH, self.FEEDBACK_PATH)
        token, token_type = self._get_token()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Snowflake-Authorization-Token-Type": token_type,
        }

        payload = {
            "request_id": request_id,
            "positive": is_positive,
            "feedback_message": feedback_message,
        }

        try:
            async with (
                aiohttp.ClientSession() as http_session,
                http_session.post(
                    api_url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response,
            ):
                if response.status >= 400:
                    error_text = await response.text()
                    logger.warning(f"Failed to submit feedback: {error_text}")
                    return False

                logger.info(f"Feedback submitted for request {request_id}")
                return True

        except Exception as exc:
            logger.warning(f"Error submitting feedback: {exc}")
            return False

    def clear_history(self) -> None:
        """Clear conversation history for a fresh start."""
        self.conversation_history = []
        logger.debug("Conversation history cleared")

    @staticmethod
    def extract_sql(response: CortexAnalystResponse) -> str | None:
        """Extract SQL statement from response.

        Args:
            response: CortexAnalystResponse object.

        Returns:
            SQL statement string or None.
        """
        return response.sql

    @staticmethod
    def extract_interpretation(response: CortexAnalystResponse) -> str | None:
        """Extract interpretation text from response.

        Args:
            response: CortexAnalystResponse object.

        Returns:
            Interpretation text or None.
        """
        return response.text

    @staticmethod
    def get_verified_query_info(response: CortexAnalystResponse) -> dict[str, Any] | None:
        """Get verified query information if one was used.

        Args:
            response: CortexAnalystResponse object.

        Returns:
            Dictionary with verified query details or None.
        """
        return response.confidence.get("verified_query_used")


# =============================================================================
# Factory Function
# =============================================================================


def create_cortex_analyst_client(
    session: Session,
    database: str | None = None,
    schema: str | None = None,
    semantic_model_path: str | None = None,
) -> AsyncCortexAnalystClient:
    """Factory function to create a Cortex Analyst client.

    Args:
        session: Active Snowpark Session.
        database: Optional database name override.
        schema: Optional schema name override.
        semantic_model_path: Optional full path to semantic model.

    Returns:
        Configured AsyncCortexAnalystClient instance.
    """
    return AsyncCortexAnalystClient(
        session=session,
        database=database,
        schema=schema,
        semantic_model_path=semantic_model_path,
    )
