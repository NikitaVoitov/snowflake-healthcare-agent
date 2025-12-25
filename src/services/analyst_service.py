"""SnowflakeCortexAnalyst service for healthcare agent - Cortex Analyst integration.

Provides factory function to create SnowflakeCortexAnalyst instances for
NL→SQL conversion using Snowflake Cortex Analyst API with semantic models.

Uses the same session management pattern as llm_service.py:
- SPCS mode: Session-only auth (OAuth token from SNOWFLAKE_HOST)
- Local mode: Explicit key-pair auth (account, user, private_key_path)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain_snowflake import SnowflakeCortexAnalyst

from src.config import settings

if TYPE_CHECKING:
    from snowflake.snowpark import Session

logger = logging.getLogger(__name__)


def get_snowflake_analyst(session: Session) -> SnowflakeCortexAnalyst:
    """Get a configured SnowflakeCortexAnalyst instance.

    Creates a SnowflakeCortexAnalyst tool for NL→SQL conversion using
    the healthcare semantic model.

    Authentication differs by environment:
    - SPCS mode: Uses session only (OAuth token from SNOWFLAKE_HOST)
    - Local mode: Passes explicit key-pair credentials for REST API

    Args:
        session: Active Snowpark Session with appropriate authentication.

    Returns:
        SnowflakeCortexAnalyst configured with healthcare semantic model.

    Example:
        >>> from src.services.analyst_service import get_snowflake_analyst
        >>> analyst = get_snowflake_analyst(session)
        >>> result = await analyst._arun("How many members do we have?")
        >>> # Returns JSON string with SQL statement
    """
    logger.debug(
        "Creating SnowflakeCortexAnalyst: semantic_model=%s, is_spcs=%s",
        settings.cortex_analyst_semantic_model,
        settings.is_spcs,
    )

    # Base config shared by both environments
    config: dict[str, Any] = {
        "session": session,
        "semantic_model_file": settings.cortex_analyst_semantic_model,
        "database": settings.snowflake_database,
        "schema": settings.snowflake_schema,
        "warehouse": settings.snowflake_warehouse,
        "use_rest_api": True,
        "enable_streaming": False,
    }

    # Local mode: Add explicit key-pair credentials for REST API
    # (SPCS mode uses OAuth token derived from session/SNOWFLAKE_HOST)
    if not settings.is_spcs:
        config.update(
            {
                "account": settings.snowflake_account,
                "user": settings.snowflake_user,
                "private_key_path": settings.snowflake_private_key_path,
                "private_key_passphrase": settings.snowflake_private_key_passphrase,
            }
        )
        logger.debug("Local mode: Added key-pair auth credentials")

    analyst = SnowflakeCortexAnalyst(**config)

    logger.info(
        "SnowflakeCortexAnalyst created: model=%s, is_spcs=%s",
        settings.cortex_analyst_semantic_model,
        settings.is_spcs,
    )

    return analyst

