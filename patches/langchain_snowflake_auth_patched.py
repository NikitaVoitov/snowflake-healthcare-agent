"""
PATCHED: SnowflakeAuth._get_session() - OAuth token authentication fix.

BUG DISCOVERED: Missing 'authenticator=oauth' when using token authentication
=====================================================================================
When SnowflakeAuth._get_session() is called with a token parameter, it passes
the token to Session.builder.configs() but DOES NOT include 'authenticator=oauth'.

This causes the Snowflake connector to fail with error:
    "251005: User is empty, but it must be provided unless authenticator is 
    one of OAUTH_CLIENT_CREDENTIALS, PROGRAMMATIC_ACCESS_TOKEN, NO_AUTH, OAUTH, 
    OAUTH_AUTHORIZATION_CODE, PAT_WITH_EXTERNAL_SESSION, WORKLOAD_IDENTITY."

FIX: When 'token' is provided, automatically add 'authenticator=oauth' to the
connection parameters. Also add 'host' from SNOWFLAKE_HOST environment variable
for SPCS OAuth authentication.

APPLIES TO:
    langchain_snowflake/chat_models/auth.py

To apply this patch:
    cp patches/langchain_snowflake_auth_patched.py \
       langchain-snowflake/libs/snowflake/langchain_snowflake/chat_models/auth.py
"""

import logging
import os
from typing import Any, Optional

from snowflake.snowpark import Session

logger = logging.getLogger(__name__)


class SnowflakeAuth:
    """Mixin class for Snowflake authentication functionality."""

    session: Optional[Session]

    def _get_session(self) -> Session:
        """Get or create Snowflake session using LLM intelligence for credential prioritization.
        
        PATCHED: Adds 'authenticator=oauth' and 'host' when token is provided.
        This is required for SPCS OAuth authentication per Snowflake documentation:
        https://docs.snowflake.com/en/developer-guide/snowpark-container-services/additional-considerations-services-jobs
        """
        # Use existing session if available
        if self.session is not None:
            return self.session

        # Build connection parameters from provided credentials
        connection_params = {}

        if self.account:
            connection_params["account"] = self.account
        if self.user:
            connection_params["user"] = self.user
        if self.password:
            if hasattr(self.password, "get_secret_value"):
                # Pydantic SecretStr
                connection_params["password"] = self.password.get_secret_value()
            else:
                # Regular string (for tests and direct usage)
                connection_params["password"] = str(self.password)
        if self.token:
            connection_params["token"] = self.token
            # PATCH: When using token, we need to specify authenticator=oauth
            # This is required per Snowflake documentation for SPCS OAuth
            connection_params["authenticator"] = "oauth"
            # PATCH: Also add host from environment if not already specified
            # SNOWFLAKE_HOST is required for SPCS OAuth authentication
            snowflake_host = os.getenv("SNOWFLAKE_HOST")
            if snowflake_host:
                connection_params["host"] = snowflake_host
                # If account wasn't provided, derive it from host
                if "account" not in connection_params:
                    connection_params["account"] = snowflake_host.replace(".snowflakecomputing.com", "")
            logger.info(f"Using OAuth token authentication with host={snowflake_host}")
        if self.private_key_path:
            connection_params["private_key_path"] = self.private_key_path
        if self.private_key_passphrase:
            connection_params["private_key_passphrase"] = self.private_key_passphrase
        if self.warehouse:
            connection_params["warehouse"] = self.warehouse
        if self.database:
            connection_params["database"] = self.database
        if self.schema:
            connection_params["schema"] = self.schema

        try:
            from snowflake.snowpark import Session

            # If we have specific connection parameters, use them
            if connection_params:
                logger.info(f"Creating Snowflake session with parameters: {list(connection_params.keys())}")
                self.session = Session.builder.configs(connection_params).create()
                return self.session

            # Fallback: Use production session management
            try:
                from .. import get_default_session

                logger.info("Using default session creation from langchain_snowflake")
                self.session = get_default_session()
                return self.session
            except Exception as e:
                logger.warning(f"Failed to create default session: {e}")
                # Final fallback to basic session
                self.session = Session.builder.configs({}).create()
                return self.session

        except Exception as e:
            raise ValueError(f"Failed to create Snowflake session: {e}")

