"""SnowflakeCortexAnalyst service for healthcare agent - Cortex Analyst integration.

Provides factory function to create SnowflakeCortexAnalyst instances for
NL→SQL conversion using Snowflake Cortex Analyst API with semantic models.

Uses the same session management pattern as llm_service.py.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_snowflake import SnowflakeCortexAnalyst

from src.config import settings

if TYPE_CHECKING:
    from snowflake.snowpark import Session

logger = logging.getLogger(__name__)


def get_snowflake_analyst(session: Session) -> SnowflakeCortexAnalyst:
    """Get a configured SnowflakeCortexAnalyst instance.

    Creates a SnowflakeCortexAnalyst tool for NL→SQL conversion using
    the healthcare semantic model. Uses session-based auth which works
    for both SPCS (OAuth) and local (key-pair) environments.

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
        "Creating SnowflakeCortexAnalyst: semantic_model=%s",
        settings.cortex_analyst_semantic_model,
    )

    analyst = SnowflakeCortexAnalyst(
        session=session,
        semantic_model_file=settings.cortex_analyst_semantic_model,
        database=settings.snowflake_database,
        schema=settings.snowflake_schema,  # Required field
        warehouse=settings.snowflake_warehouse,
        use_rest_api=True,  # Use REST API (patched with SNOWFLAKE_HOST fix)
        enable_streaming=False,
    )

    logger.info(
        "SnowflakeCortexAnalyst created: model=%s, is_spcs=%s",
        settings.cortex_analyst_semantic_model,
        settings.is_spcs,
    )

    return analyst

