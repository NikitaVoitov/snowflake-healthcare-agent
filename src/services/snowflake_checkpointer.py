"""Custom AsyncSnowflakeSaver for LangGraph checkpointing with SPCS OAuth support.

Based on: https://medium.com/@siva_yetukuri/how-to-leverage-snowflake-as-a-checkpointer
Implements async checkpointing using native snowflake.connector with OAuth token auth.
"""

import asyncio
import json
import logging
import os
import random
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, Optional, Sequence, cast

import snowflake.connector
from snowflake.connector import DictCursor

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.types import TASKS, ChannelProtocol

from src.config import is_running_in_spcs, settings

logger = logging.getLogger(__name__)

MetadataInput = Optional[dict[str, Any]]

# =============================================================================
# SQL Statements
# =============================================================================

MIGRATIONS = [
    """
    CREATE TABLE IF NOT EXISTS {schema}.LANGGRAPH_CHECKPOINT_MIGRATIONS (
        v INTEGER PRIMARY KEY
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS {schema}.LANGGRAPH_CHECKPOINTS (
        thread_id STRING NOT NULL,
        checkpoint_ns STRING NOT NULL DEFAULT '',
        checkpoint_id STRING NOT NULL,
        parent_checkpoint_id STRING,
        type STRING,
        checkpoint VARIANT NOT NULL,
        metadata VARIANT NOT NULL DEFAULT PARSE_JSON('{{}}'),
        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS {schema}.LANGGRAPH_CHECKPOINT_BLOBS (
        thread_id STRING NOT NULL,
        checkpoint_ns STRING NOT NULL DEFAULT '',
        channel STRING NOT NULL,
        version STRING NOT NULL,
        type STRING NOT NULL,
        blob BINARY,
        PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS {schema}.LANGGRAPH_CHECKPOINT_WRITES (
        thread_id STRING NOT NULL,
        checkpoint_ns STRING NOT NULL DEFAULT '',
        checkpoint_id STRING NOT NULL,
        task_id STRING NOT NULL,
        idx INTEGER NOT NULL,
        channel STRING NOT NULL,
        type STRING,
        blob BINARY NOT NULL,
        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
    )
    """,
]

# Simplified SELECT - fetch checkpoint base data only, load related data separately
SELECT_CHECKPOINT_SQL = """
SELECT
    thread_id,
    checkpoint,
    checkpoint_ns,
    checkpoint_id,
    parent_checkpoint_id,
    metadata
FROM {schema}.LANGGRAPH_CHECKPOINTS c
"""

SELECT_BLOBS_SQL = """
SELECT channel, type, TO_VARCHAR(blob, 'HEX') as blob_hex
FROM {schema}.LANGGRAPH_CHECKPOINT_BLOBS
WHERE thread_id = '{thread_id}'
  AND checkpoint_ns = '{checkpoint_ns}'
  AND (channel, version) IN (
      SELECT key, value::STRING
      FROM TABLE(FLATTEN(INPUT => PARSE_JSON('{channel_versions}')))
  )
"""

SELECT_WRITES_SQL = """
SELECT task_id, channel, type, TO_VARCHAR(blob, 'HEX') as blob_hex
FROM {schema}.LANGGRAPH_CHECKPOINT_WRITES
WHERE thread_id = '{thread_id}'
  AND checkpoint_ns = '{checkpoint_ns}'
  AND checkpoint_id = '{checkpoint_id}'
ORDER BY task_id, idx
"""

SELECT_PENDING_SENDS_SQL = """
SELECT type, TO_VARCHAR(blob, 'HEX') as blob_hex
FROM {schema}.LANGGRAPH_CHECKPOINT_WRITES
WHERE thread_id = '{thread_id}'
  AND checkpoint_ns = '{checkpoint_ns}'
  AND checkpoint_id = '{checkpoint_id}'
  AND channel = '{tasks_channel}'
ORDER BY idx
"""

# Use inline SQL formatting - simpler and more compatible with SPCS
# Values are properly escaped before being inserted

UPSERT_CHECKPOINT_BLOBS_SQL = """
MERGE INTO {schema}.LANGGRAPH_CHECKPOINT_BLOBS AS target
USING (SELECT '{thread_id}' AS thread_id, '{checkpoint_ns}' AS checkpoint_ns,
              '{channel}' AS channel, '{version}' AS version, '{type}' AS type,
              TO_BINARY('{blob}', 'HEX') AS blob) AS source
ON target.thread_id = source.thread_id
   AND target.checkpoint_ns = source.checkpoint_ns
   AND target.channel = source.channel
   AND target.version = source.version
WHEN NOT MATCHED THEN
    INSERT (thread_id, checkpoint_ns, channel, version, type, blob)
    VALUES (source.thread_id, source.checkpoint_ns, source.channel,
            source.version, source.type, source.blob)
"""

UPSERT_CHECKPOINTS_SQL = """
MERGE INTO {schema}.LANGGRAPH_CHECKPOINTS AS target
USING (SELECT '{thread_id}' AS thread_id, '{checkpoint_ns}' AS checkpoint_ns,
              '{checkpoint_id}' AS checkpoint_id, {parent_checkpoint_id} AS parent_checkpoint_id,
              PARSE_JSON($${checkpoint}$$) AS checkpoint,
              PARSE_JSON($${metadata}$$) AS metadata) AS source
ON target.thread_id = source.thread_id
   AND target.checkpoint_ns = source.checkpoint_ns
   AND target.checkpoint_id = source.checkpoint_id
WHEN MATCHED THEN
    UPDATE SET checkpoint = source.checkpoint, metadata = source.metadata
WHEN NOT MATCHED THEN
    INSERT (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata)
    VALUES (source.thread_id, source.checkpoint_ns, source.checkpoint_id,
            source.parent_checkpoint_id, source.checkpoint, source.metadata)
"""

UPSERT_CHECKPOINT_WRITES_SQL = """
MERGE INTO {schema}.LANGGRAPH_CHECKPOINT_WRITES AS target
USING (SELECT '{thread_id}' AS thread_id, '{checkpoint_ns}' AS checkpoint_ns,
              '{checkpoint_id}' AS checkpoint_id, '{task_id}' AS task_id,
              {idx} AS idx, '{channel}' AS channel, '{type}' AS type,
              TO_BINARY('{blob}', 'HEX') AS blob) AS source
ON target.thread_id = source.thread_id
   AND target.checkpoint_ns = source.checkpoint_ns
   AND target.checkpoint_id = source.checkpoint_id
   AND target.task_id = source.task_id
   AND target.idx = source.idx
WHEN MATCHED THEN
    UPDATE SET channel = source.channel, type = source.type, blob = source.blob
WHEN NOT MATCHED THEN
    INSERT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob)
    VALUES (source.thread_id, source.checkpoint_ns, source.checkpoint_id,
            source.task_id, source.idx, source.channel, source.type, source.blob)
"""

INSERT_CHECKPOINT_WRITES_SQL = """
MERGE INTO {schema}.LANGGRAPH_CHECKPOINT_WRITES AS target
USING (SELECT '{thread_id}' AS thread_id, '{checkpoint_ns}' AS checkpoint_ns,
              '{checkpoint_id}' AS checkpoint_id, '{task_id}' AS task_id,
              {idx} AS idx, '{channel}' AS channel, '{type}' AS type,
              TO_BINARY('{blob}', 'HEX') AS blob) AS source
ON target.thread_id = source.thread_id
   AND target.checkpoint_ns = source.checkpoint_ns
   AND target.checkpoint_id = source.checkpoint_id
   AND target.task_id = source.task_id
   AND target.idx = source.idx
WHEN NOT MATCHED THEN
    INSERT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob)
    VALUES (source.thread_id, source.checkpoint_ns, source.checkpoint_id,
            source.task_id, source.idx, source.channel, source.type, source.blob)
"""


# =============================================================================
# Async Cursor Wrapper
# =============================================================================


class AsyncCursor:
    """Async wrapper for Snowflake sync cursor using asyncio.to_thread."""

    def __init__(self, sync_cursor: DictCursor) -> None:
        self.sync_cursor = sync_cursor

    async def execute(self, query: str, params: Any = None) -> Any:
        """Execute query in separate thread."""
        return await asyncio.to_thread(self.sync_cursor.execute, query, params)

    async def executemany(self, query: str, params: Any = None) -> Any:
        """Execute many in separate thread."""
        return await asyncio.to_thread(self.sync_cursor.executemany, query, params)

    async def fetchone(self) -> Any:
        """Fetch one row in separate thread."""
        return await asyncio.to_thread(self.sync_cursor.fetchone)

    async def fetchall(self) -> list[Any]:
        """Fetch all rows in separate thread."""
        return await asyncio.to_thread(self.sync_cursor.fetchall)

    async def close(self) -> None:
        """Close cursor in separate thread."""
        return await asyncio.to_thread(self.sync_cursor.close)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to underlying sync cursor."""
        return getattr(self.sync_cursor, name)


# =============================================================================
# Custom AsyncSnowflakeSaver
# =============================================================================


class CustomAsyncSnowflakeSaver(BaseCheckpointSaver):
    """Async Snowflake checkpointer with native connector and SPCS OAuth support."""

    def __init__(
        self,
        connection: snowflake.connector.SnowflakeConnection,
        schema: str = "CHECKPOINT_SCHEMA",
    ) -> None:
        super().__init__()
        self.connection = connection
        self.schema = schema
        self.lock = asyncio.Lock()

    @classmethod
    def create_connection(cls) -> snowflake.connector.SnowflakeConnection:
        """Create Snowflake connection based on environment (SPCS vs local)."""
        if is_running_in_spcs():
            return cls._create_spcs_connection()
        return cls._create_local_connection()

    @classmethod
    def _create_spcs_connection(cls) -> snowflake.connector.SnowflakeConnection:
        """Create connection using SPCS OAuth token."""
        token_path = Path("/snowflake/session/token")
        if not token_path.exists():
            raise ValueError("SPCS token file not found")

        token = token_path.read_text().strip()
        host = os.getenv("SNOWFLAKE_HOST")
        if not host:
            raise ValueError("SNOWFLAKE_HOST not set in SPCS environment")

        logger.info("Creating SPCS connection with OAuth token")
        return snowflake.connector.connect(
            account=host.replace(".snowflakecomputing.com", ""),
            host=host,
            authenticator="oauth",
            token=token,
            database=settings.snowflake_database,
            warehouse=settings.snowflake_warehouse,
        )

    @classmethod
    def _create_local_connection(cls) -> snowflake.connector.SnowflakeConnection:
        """Create connection using key-pair authentication."""
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import serialization

        key_path = settings.snowflake_private_key_path
        passphrase = settings.snowflake_private_key_passphrase

        if not key_path:
            raise ValueError("SNOWFLAKE_PRIVATE_KEY_PATH required for local")

        with open(key_path, "rb") as f:
            private_key = serialization.load_pem_private_key(
                f.read(),
                password=passphrase.encode() if passphrase else None,
                backend=default_backend(),
            )

        private_key_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        logger.info("Creating local connection with key-pair auth")
        return snowflake.connector.connect(
            account=settings.snowflake_account,
            user=settings.snowflake_user,
            private_key=private_key_bytes,
            database=settings.snowflake_database,
            warehouse=settings.snowflake_warehouse,
        )

    @asynccontextmanager
    async def _cursor(self) -> AsyncIterator[AsyncCursor]:
        """Async context manager for cursor."""
        async with self.lock:
            cursor = self.connection.cursor(DictCursor)
            async_cursor = AsyncCursor(cursor)
            try:
                # Ensure warehouse is active for all operations
                await async_cursor.execute(f"USE WAREHOUSE {settings.snowflake_warehouse}")
                yield async_cursor
            finally:
                await async_cursor.close()

    async def asetup(self) -> None:
        """Create checkpoint tables if they don't exist."""
        async with self._cursor() as cur:
            try:
                await cur.execute(
                    f"SELECT v FROM {self.schema}.LANGGRAPH_CHECKPOINT_MIGRATIONS "
                    "ORDER BY v DESC LIMIT 1"
                )
                row = await cur.fetchone()
                version = row["V"] if row else -1
            except Exception:
                version = -1

            for v, migration in enumerate(MIGRATIONS[version + 1 :], start=version + 1):
                await cur.execute(migration.format(schema=self.schema))
                await cur.execute(
                    f"INSERT INTO {self.schema}.LANGGRAPH_CHECKPOINT_MIGRATIONS (v) "
                    f"VALUES ({v})"
                )
                logger.info(f"Applied migration {v}")

    # -------------------------------------------------------------------------
    # Load/Dump helpers
    # -------------------------------------------------------------------------

    def _load_checkpoint(
        self,
        checkpoint: dict[str, Any],
        channel_values: list[list[str]] | None,
        pending_sends: list[list[str]] | None,
    ) -> Checkpoint:
        """Load checkpoint from database format."""
        return {
            **checkpoint,
            "pending_sends": [
                self.serde.loads_typed((c, bytes.fromhex(b)))
                for c, b in (pending_sends or [])
            ],
            "channel_values": self._load_blobs(channel_values),
        }

    def _load_blobs(self, blob_values: list[list[str]] | None) -> dict[str, Any]:
        """Load blobs from database format."""
        if not blob_values:
            return {}
        return {
            k: self.serde.loads_typed((t, bytes.fromhex(v)))
            for k, t, v in blob_values
            if t != "empty"
        }

    def _dump_checkpoint(self, checkpoint: Checkpoint) -> dict[str, Any]:
        """Dump checkpoint for database storage."""
        return {**checkpoint, "pending_sends": []}

    def _dump_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        values: dict[str, Any],
        versions: ChannelVersions,
    ) -> list[tuple[str, str, str, str, str, str]]:
        """Dump blobs for database storage as positional tuples."""
        if not versions:
            return []

        result = []
        for k, ver in versions.items():
            if k in values:
                type_str, blob_bytes = self.serde.dumps_typed(values[k])
                blob_hex = blob_bytes.hex() if blob_bytes else ""
            else:
                type_str, blob_hex = "empty", ""

            # Order: thread_id, checkpoint_ns, channel, version, type, blob
            result.append((
                thread_id,
                checkpoint_ns,
                k,
                cast(str, ver),
                type_str,
                blob_hex,
            ))
        return result

    def _load_writes(
        self, writes: list[list[str]] | None
    ) -> list[tuple[str, str, Any]]:
        """Load writes from database format."""
        if not writes:
            return []
        return [
            (tid, channel, self.serde.loads_typed((t, bytes.fromhex(v))))
            for tid, channel, t, v in writes
        ]

    def _dump_writes(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        writes: Sequence[tuple[str, Any]],
    ) -> list[tuple[str, str, str, str, int, str, str, str]]:
        """Dump writes for database storage as positional tuples."""
        result = []
        for idx, (channel, value) in enumerate(writes):
            type_str, blob_bytes = self.serde.dumps_typed(value)
            # Order: thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob
            result.append((
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                task_id,
                WRITES_IDX_MAP.get(channel, idx),
                channel,
                type_str,
                blob_bytes.hex() if blob_bytes else "",
            ))
        return result

    def _load_metadata(self, metadata: dict[str, Any]) -> CheckpointMetadata:
        """Load metadata from database format."""
        # Metadata is simple dict, just cast it
        return cast(CheckpointMetadata, metadata)

    def _dump_metadata(self, metadata: CheckpointMetadata) -> str:
        """Dump metadata for database storage."""
        # Serialize to JSON string, removing null characters
        return json.dumps(metadata).replace("\\u0000", "")

    def _escape(self, value: Any) -> str:
        """Escape string value for SQL - replace single quotes with doubled quotes."""
        if value is None:
            return ""
        return str(value).replace("'", "''")

    def get_next_version(
        self, current: str | None, channel: ChannelProtocol
    ) -> str:
        """Generate next version string."""
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"

    # -------------------------------------------------------------------------
    # Async methods (primary interface)
    # -------------------------------------------------------------------------

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get checkpoint tuple from database using simplified queries."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        # Build WHERE clause
        if checkpoint_id:
            where = (
                f"WHERE thread_id = '{self._escape(thread_id)}' "
                f"AND checkpoint_ns = '{self._escape(checkpoint_ns)}' "
                f"AND checkpoint_id = '{self._escape(checkpoint_id)}'"
            )
        else:
            where = (
                f"WHERE thread_id = '{self._escape(thread_id)}' "
                f"AND checkpoint_ns = '{self._escape(checkpoint_ns)}' "
                "ORDER BY checkpoint_id DESC LIMIT 1"
            )

        # Query 1: Get checkpoint base data
        query = SELECT_CHECKPOINT_SQL.format(schema=self.schema) + where

        async with self._cursor() as cur:
            await cur.execute(query, None)
            rows = await cur.fetchall()

            if not rows:
                return None

            row = rows[0]
            checkpoint_data = (
                json.loads(row["CHECKPOINT"])
                if isinstance(row["CHECKPOINT"], str)
                else row["CHECKPOINT"]
            )
            metadata_data = (
                json.loads(row["METADATA"])
                if isinstance(row["METADATA"], str)
                else row["METADATA"]
            )
            cp_id = row["CHECKPOINT_ID"]
            parent_cp_id = row["PARENT_CHECKPOINT_ID"]

            # Extract channel_versions from checkpoint for blob lookup
            channel_versions = checkpoint_data.get("channel_versions", {})

            # Query 2: Get blobs (if there are channel versions)
            channel_values: list[list[str]] = []
            if channel_versions:
                blob_query = SELECT_BLOBS_SQL.format(
                    schema=self.schema,
                    thread_id=self._escape(thread_id),
                    checkpoint_ns=self._escape(checkpoint_ns),
                    channel_versions=json.dumps(channel_versions),
                )
                await cur.execute(blob_query, None)
                blob_rows = await cur.fetchall()
                channel_values = [
                    [r["CHANNEL"], r["TYPE"], r["BLOB_HEX"]] for r in blob_rows
                ]

            # Query 3: Get pending writes
            writes_query = SELECT_WRITES_SQL.format(
                schema=self.schema,
                thread_id=self._escape(thread_id),
                checkpoint_ns=self._escape(checkpoint_ns),
                checkpoint_id=self._escape(cp_id),
            )
            await cur.execute(writes_query, None)
            writes_rows = await cur.fetchall()
            pending_writes = [
                [r["TASK_ID"], r["CHANNEL"], r["TYPE"], r["BLOB_HEX"]]
                for r in writes_rows
            ]

            # Query 4: Get pending sends (from parent checkpoint)
            pending_sends: list[list[str]] = []
            if parent_cp_id:
                sends_query = SELECT_PENDING_SENDS_SQL.format(
                    schema=self.schema,
                    thread_id=self._escape(thread_id),
                    checkpoint_ns=self._escape(checkpoint_ns),
                    checkpoint_id=self._escape(parent_cp_id),
                    tasks_channel=TASKS,
                )
                await cur.execute(sends_query, None)
                sends_rows = await cur.fetchall()
                pending_sends = [[r["TYPE"], r["BLOB_HEX"]] for r in sends_rows]

            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": cp_id,
                    }
                },
                checkpoint=self._load_checkpoint(
                    checkpoint_data, channel_values, pending_sends
                ),
                metadata=self._load_metadata(metadata_data),
                parent_config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_cp_id,
                    }
                }
                if parent_cp_id
                else None,
                pending_writes=self._load_writes(pending_writes),
            )

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: MetadataInput = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from database using simplified queries."""
        wheres: list[str] = []

        if config:
            thread_id = config["configurable"]["thread_id"]
            wheres.append(f"thread_id = '{self._escape(thread_id)}'")
            checkpoint_ns = config["configurable"].get("checkpoint_ns")
            if checkpoint_ns is not None:
                wheres.append(f"checkpoint_ns = '{self._escape(checkpoint_ns)}'")
            if checkpoint_id := get_checkpoint_id(config):
                wheres.append(f"checkpoint_id = '{self._escape(checkpoint_id)}'")

        if filter:
            wheres.append(
                f"OBJECT_CONTAINS(metadata, PARSE_JSON($${json.dumps(filter)}$$))"
            )

        if before:
            before_id = get_checkpoint_id(before)
            wheres.append(f"checkpoint_id < '{self._escape(before_id)}'")

        where_clause = "WHERE " + " AND ".join(wheres) if wheres else ""
        query = (
            SELECT_CHECKPOINT_SQL.format(schema=self.schema)
            + where_clause
            + " ORDER BY checkpoint_id DESC"
        )
        if limit:
            query += f" LIMIT {limit}"

        async with self._cursor() as cur:
            await cur.execute(query, None)
            rows = await cur.fetchall()

            for row in rows:
                checkpoint_data = (
                    json.loads(row["CHECKPOINT"])
                    if isinstance(row["CHECKPOINT"], str)
                    else row["CHECKPOINT"]
                )
                metadata_data = (
                    json.loads(row["METADATA"])
                    if isinstance(row["METADATA"], str)
                    else row["METADATA"]
                )
                tid = row["THREAD_ID"]
                cns = row["CHECKPOINT_NS"]
                cp_id = row["CHECKPOINT_ID"]
                parent_cp_id = row["PARENT_CHECKPOINT_ID"]

                # For listing, we use simplified data (no blobs/writes fetched)
                # This is for performance - full data loaded on get_tuple
                yield CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": tid,
                            "checkpoint_ns": cns,
                            "checkpoint_id": cp_id,
                        }
                    },
                    checkpoint=self._load_checkpoint(checkpoint_data, None, None),
                    metadata=self._load_metadata(metadata_data),
                    parent_config={
                        "configurable": {
                            "thread_id": tid,
                            "checkpoint_ns": cns,
                            "checkpoint_id": parent_cp_id,
                        }
                    }
                    if parent_cp_id
                    else None,
                    pending_writes=[],
                )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save checkpoint to database."""
        configurable = config["configurable"].copy()
        thread_id = configurable.pop("thread_id")
        checkpoint_ns = configurable.pop("checkpoint_ns", "")
        checkpoint_id = configurable.pop(
            "checkpoint_id", configurable.pop("thread_ts", None)
        )

        copy = checkpoint.copy()
        channel_values = copy.pop("channel_values")  # type: ignore[misc]

        next_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

        # Dump blobs
        blob_args = self._dump_blobs(
            thread_id, checkpoint_ns, channel_values, new_versions
        )

        async with self._cursor() as cur:
            # Insert blobs - use string formatting
            for blob_arg in blob_args:
                sql = UPSERT_CHECKPOINT_BLOBS_SQL.format(
                    schema=self.schema,
                    thread_id=self._escape(blob_arg[0]),
                    checkpoint_ns=self._escape(blob_arg[1]),
                    channel=self._escape(blob_arg[2]),
                    version=self._escape(blob_arg[3]),
                    type=self._escape(blob_arg[4]),
                    blob=blob_arg[5],  # Already hex-encoded
                )
                await cur.execute(sql, None)

            # Insert checkpoint - use string formatting
            parent_id = f"'{checkpoint_id}'" if checkpoint_id else "NULL"
            sql = UPSERT_CHECKPOINTS_SQL.format(
                schema=self.schema,
                thread_id=self._escape(thread_id),
                checkpoint_ns=self._escape(checkpoint_ns),
                checkpoint_id=self._escape(checkpoint["id"]),
                parent_checkpoint_id=parent_id,
                checkpoint=json.dumps(self._dump_checkpoint(copy)),
                metadata=self._dump_metadata(metadata),
            )
            await cur.execute(sql, None)

        logger.debug(f"Saved checkpoint {checkpoint['id']} for thread {thread_id}")
        return next_config

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store intermediate writes linked to checkpoint."""
        query_template = (
            UPSERT_CHECKPOINT_WRITES_SQL
            if all(w[0] in WRITES_IDX_MAP for w in writes)
            else INSERT_CHECKPOINT_WRITES_SQL
        )

        write_args = self._dump_writes(
            config["configurable"]["thread_id"],
            config["configurable"].get("checkpoint_ns", ""),
            config["configurable"]["checkpoint_id"],
            task_id,
            writes,
        )

        async with self._cursor() as cur:
            for write_arg in write_args:
                # Unpack tuple and format SQL
                sql = query_template.format(
                    schema=self.schema,
                    thread_id=self._escape(write_arg[0]),
                    checkpoint_ns=self._escape(write_arg[1]),
                    checkpoint_id=self._escape(write_arg[2]),
                    task_id=self._escape(write_arg[3]),
                    idx=write_arg[4],
                    channel=self._escape(write_arg[5]),
                    type=self._escape(write_arg[6]),
                    blob=write_arg[7],  # Already hex-encoded
                )
                await cur.execute(sql, None)

    # -------------------------------------------------------------------------
    # Sync methods (required by BaseCheckpointSaver)
    # -------------------------------------------------------------------------

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Sync wrapper for aget_tuple."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # If already in async context, create new loop in thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.aget_tuple(config))
                return future.result()
        return asyncio.run(self.aget_tuple(config))

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: MetadataInput = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """Sync wrapper for alist."""

        async def collect() -> list[CheckpointTuple]:
            results = []
            async for item in self.alist(config, filter=filter, before=before, limit=limit):
                results.append(item)
            return results

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, collect())
                yield from future.result()
        else:
            yield from asyncio.run(collect())

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Sync wrapper for aput."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.aput(config, checkpoint, metadata, new_versions),
                )
                return future.result()
        return asyncio.run(self.aput(config, checkpoint, metadata, new_versions))

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Sync wrapper for aput_writes."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run, self.aput_writes(config, writes, task_id)
                )
                future.result()
        else:
            asyncio.run(self.aput_writes(config, writes, task_id))


# =============================================================================
# Factory function
# =============================================================================


async def create_async_snowflake_checkpointer() -> CustomAsyncSnowflakeSaver:
    """Create and setup async Snowflake checkpointer."""
    conn = CustomAsyncSnowflakeSaver.create_connection()
    checkpointer = CustomAsyncSnowflakeSaver(
        connection=conn,
        schema=f"{settings.snowflake_database}.CHECKPOINT_SCHEMA",
    )
    await checkpointer.asetup()
    logger.info("AsyncSnowflakeSaver initialized with persistent Snowflake storage")
    return checkpointer

