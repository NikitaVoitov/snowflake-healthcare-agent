"""LangGraph checkpointer setup for Snowflake persistence.

Uses CustomAsyncSnowflakeSaver for both local and SPCS environments.
"""

import logging
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from src.config import is_running_in_spcs

logger = logging.getLogger(__name__)


def create_checkpointer() -> Any:
    """Create checkpointer for LangGraph workflow.

    Returns:
        CustomAsyncSnowflakeSaver for persistent storage, or MemorySaver as fallback
    """
    try:
        from src.services.snowflake_checkpointer import CustomAsyncSnowflakeSaver

        conn = CustomAsyncSnowflakeSaver.create_connection()
        from src.config import settings

        checkpointer = CustomAsyncSnowflakeSaver(
            connection=conn,
            schema=f"{settings.snowflake_database}.CHECKPOINT_SCHEMA",
        )

        env = "SPCS" if is_running_in_spcs() else "local"
        logger.info(f"Using CustomAsyncSnowflakeSaver ({env}) for persistent checkpoints")
        return checkpointer

    except Exception as e:
        logger.warning(f"Failed to create Snowflake checkpointer: {e}")
        logger.info("Falling back to MemorySaver (in-memory, not persistent)")
        return MemorySaver()
