"""LangGraph checkpointer setup for Snowflake persistence."""

import logging
import os

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

logger = logging.getLogger(__name__)


def get_snowflake_connection_params() -> dict:
    """Build Snowflake connection parameters from environment.

    Returns:
        Dictionary with connection parameters including private_key bytes
    """
    private_key_path = os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH")
    passphrase = os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")

    if not private_key_path:
        raise ValueError("SNOWFLAKE_PRIVATE_KEY_PATH environment variable required")

    # Load and convert private key
    with open(private_key_path, "rb") as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(),
            password=passphrase.encode() if passphrase else None,
            backend=default_backend(),
        )

    private_key_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    return {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
        "private_key": private_key_bytes,
        "database": os.getenv("SNOWFLAKE_DATABASE", "HEALTHCARE_DB"),
        "schema": "CHECKPOINT_SCHEMA",
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "PAYERS_CC_WH"),
        "role": os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
    }


async def create_checkpointer():
    """Create AsyncSnowflakeSaver for LangGraph checkpointing.

    Returns:
        Configured AsyncSnowflakeSaver instance

    Note:
        Requires langgraph-checkpoint-snowflake package
    """
    try:
        from langgraph_checkpoint_snowflake import AsyncSnowflakeSaver

        conn_params = get_snowflake_connection_params()
        return AsyncSnowflakeSaver.from_conn_params(
            conn_params,
            namespace="healthcare_agents",
        )
    except ImportError:
        logger.warning(
            "langgraph-checkpoint-snowflake not installed, using MemorySaver fallback"
        )
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    except Exception as exc:
        logger.error(f"Failed to create Snowflake checkpointer: {exc}")
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()

