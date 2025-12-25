"""Unit tests for CustomAsyncSnowflakeSaver checkpointer."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.services.snowflake_checkpointer import (
    MIGRATIONS,
    SELECT_CHECKPOINT_SQL,
    UPSERT_CHECKPOINT_BLOBS_SQL,
    UPSERT_CHECKPOINTS_SQL,
    AsyncCursor,
    CustomAsyncSnowflakeSaver,
)


class TestAsyncCursor:
    """Tests for AsyncCursor wrapper."""

    @pytest.fixture
    def mock_sync_cursor(self):
        """Create mock sync cursor."""
        cursor = MagicMock()
        cursor.execute.return_value = None
        cursor.executemany.return_value = None
        cursor.fetchone.return_value = {"V": 3}
        cursor.fetchall.return_value = []
        cursor.close.return_value = None
        cursor.sfqid = "test-query-id"
        return cursor

    @pytest.fixture
    def async_cursor(self, mock_sync_cursor):
        """Create AsyncCursor with mock."""
        return AsyncCursor(mock_sync_cursor)

    @pytest.mark.asyncio
    async def test_execute_delegates_to_sync(self, async_cursor, mock_sync_cursor):
        """Test execute runs in thread and delegates to sync cursor."""
        await async_cursor.execute("SELECT 1", None)
        mock_sync_cursor.execute.assert_called_once_with("SELECT 1", None)

    @pytest.mark.asyncio
    async def test_executemany_delegates_to_sync(self, async_cursor, mock_sync_cursor):
        """Test executemany runs in thread and delegates to sync cursor."""
        params = [{"a": 1}, {"a": 2}]
        await async_cursor.executemany("INSERT INTO t VALUES (%(a)s)", params)
        mock_sync_cursor.executemany.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetchone_returns_row(self, async_cursor, mock_sync_cursor):
        """Test fetchone returns single row."""
        result = await async_cursor.fetchone()
        assert result == {"V": 3}
        mock_sync_cursor.fetchone.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetchall_returns_rows(self, async_cursor, mock_sync_cursor):
        """Test fetchall returns all rows."""
        mock_sync_cursor.fetchall.return_value = [{"a": 1}, {"a": 2}]
        result = await async_cursor.fetchall()
        assert len(result) == 2
        mock_sync_cursor.fetchall.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_closes_cursor(self, async_cursor, mock_sync_cursor):
        """Test close delegates to sync cursor."""
        await async_cursor.close()
        mock_sync_cursor.close.assert_called_once()

    def test_getattr_delegates_to_sync(self, async_cursor, mock_sync_cursor):  # noqa: ARG002
        """Test attribute access delegates to sync cursor."""
        assert async_cursor.sfqid == "test-query-id"


class TestCustomAsyncSnowflakeSaver:
    """Tests for CustomAsyncSnowflakeSaver."""

    @pytest.fixture
    def mock_connection(self):
        """Create mock Snowflake connection."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = None
        mock_cursor.fetchone.return_value = None
        mock_cursor.fetchall.return_value = []
        mock_cursor.close.return_value = None
        conn.cursor.return_value = mock_cursor
        return conn

    @pytest.fixture
    def checkpointer(self, mock_connection):
        """Create checkpointer with mock connection."""
        return CustomAsyncSnowflakeSaver(
            connection=mock_connection,
            schema="TEST_DB.CHECKPOINT_SCHEMA",
        )

    def test_init_sets_properties(self, checkpointer, mock_connection):
        """Test initialization sets connection and schema."""
        assert checkpointer.connection == mock_connection
        assert checkpointer.schema == "TEST_DB.CHECKPOINT_SCHEMA"
        assert checkpointer.lock is not None

    def test_get_next_version_from_none(self, checkpointer):
        """Test version generation from None."""
        version = checkpointer.get_next_version(None, None)
        assert version.startswith("00000000000000000000000000000001.")

    def test_get_next_version_increments(self, checkpointer):
        """Test version generation increments."""
        version = checkpointer.get_next_version("00000000000000000000000000000005.123", None)
        assert version.startswith("00000000000000000000000000000006.")

    def test_dump_checkpoint_removes_pending_sends(self, checkpointer):
        """Test checkpoint dump removes pending_sends."""
        checkpoint = {
            "id": "test-id",
            "ts": "2024-01-01",
            "pending_sends": [{"msg": "test"}],
            "channel_values": {},
        }
        result = checkpointer._dump_checkpoint(checkpoint)
        assert result["pending_sends"] == []
        assert result["id"] == "test-id"

    def test_load_blobs_empty(self, checkpointer):
        """Test loading empty blobs."""
        result = checkpointer._load_blobs(None)
        assert result == {}

    def test_load_blobs_skips_empty_type(self, checkpointer):
        """Test loading blobs skips 'empty' type entries."""
        blob_values = [
            ["channel1", "empty", ""],
            ["channel2", "json", "7b2261223a317d"],  # {"a":1} in hex
        ]
        # This will fail without proper serde, but we're testing the skip logic
        with patch.object(checkpointer, "serde") as mock_serde:
            mock_serde.loads_typed.return_value = {"a": 1}
            result = checkpointer._load_blobs(blob_values)
            # Only channel2 should be loaded (channel1 is 'empty')
            assert "channel2" in result
            assert "channel1" not in result

    def test_dump_metadata_serializes(self, checkpointer):
        """Test metadata dump serializes to JSON string."""
        metadata = {"step": 1, "source": "test"}
        result = checkpointer._dump_metadata(metadata)
        assert isinstance(result, str)
        # Should be valid JSON
        parsed = json.loads(result)
        assert parsed["step"] == 1

    def test_load_metadata_deserializes(self, checkpointer):
        """Test metadata load deserializes from dict."""
        metadata = {"step": 1, "source": "test"}
        result = checkpointer._load_metadata(metadata)
        # Result should preserve the data
        assert result.get("step") == 1

    def test_dump_blobs_empty_versions(self, checkpointer):
        """Test dump blobs with empty versions returns empty list."""
        result = checkpointer._dump_blobs("thread1", "ns", {}, {})
        assert result == []

    def test_dump_writes_creates_records(self, checkpointer):
        """Test dump writes creates proper tuples."""
        with patch.object(checkpointer, "serde") as mock_serde:
            mock_serde.dumps_typed.return_value = ("json", b'{"test": 1}')
            writes = [("channel1", {"test": 1})]
            result = checkpointer._dump_writes("thread1", "ns", "cp1", "task1", writes)
            assert len(result) == 1
            # Tuple order: thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob
            assert result[0][0] == "thread1"  # thread_id
            assert result[0][1] == "ns"  # checkpoint_ns
            assert result[0][5] == "channel1"  # channel

    def test_load_writes_empty(self, checkpointer):
        """Test load writes with None returns empty list."""
        result = checkpointer._load_writes(None)
        assert result == []


class TestSQLStatements:
    """Tests for SQL statement templates."""

    def test_migrations_count(self):
        """Test correct number of migrations."""
        assert len(MIGRATIONS) == 4

    def test_migrations_contain_table_names(self):
        """Test migrations create expected tables."""
        migration_sql = " ".join(MIGRATIONS)
        assert "LANGGRAPH_CHECKPOINT_MIGRATIONS" in migration_sql
        assert "LANGGRAPH_CHECKPOINTS" in migration_sql
        assert "LANGGRAPH_CHECKPOINT_BLOBS" in migration_sql
        assert "LANGGRAPH_CHECKPOINT_WRITES" in migration_sql

    def test_select_sql_has_schema_placeholder(self):
        """Test SELECT SQL has schema placeholder."""
        assert "{schema}" in SELECT_CHECKPOINT_SQL

    def test_upsert_blobs_has_merge(self):
        """Test UPSERT blobs uses MERGE."""
        assert "MERGE INTO" in UPSERT_CHECKPOINT_BLOBS_SQL

    def test_upsert_checkpoints_has_merge(self):
        """Test UPSERT checkpoints uses MERGE."""
        assert "MERGE INTO" in UPSERT_CHECKPOINTS_SQL


class TestCheckpointerAsyncMethods:
    """Tests for async methods of CustomAsyncSnowflakeSaver."""

    @pytest.fixture
    def mock_connection(self):
        """Create mock Snowflake connection."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = None
        mock_cursor.fetchone.return_value = {"V": 3}
        mock_cursor.fetchall.return_value = []
        mock_cursor.close.return_value = None
        conn.cursor.return_value = mock_cursor
        return conn

    @pytest.fixture
    def checkpointer(self, mock_connection):
        """Create checkpointer with mock connection."""
        return CustomAsyncSnowflakeSaver(
            connection=mock_connection,
            schema="TEST_DB.CHECKPOINT_SCHEMA",
        )

    @pytest.mark.asyncio
    async def test_asetup_runs_migrations(self, checkpointer, mock_connection):
        """Test asetup executes migrations."""
        mock_cursor = mock_connection.cursor.return_value
        # Simulate no existing migrations
        mock_cursor.fetchone.return_value = None

        await checkpointer.asetup()

        # Should have executed migrations
        assert mock_cursor.execute.call_count > 0

    @pytest.mark.asyncio
    async def test_aget_tuple_returns_none_for_missing(self, checkpointer, mock_connection):
        """Test aget_tuple returns None when checkpoint not found."""
        mock_cursor = mock_connection.cursor.return_value
        mock_cursor.fetchall.return_value = []

        config = {"configurable": {"thread_id": "test-thread", "checkpoint_ns": ""}}
        result = await checkpointer.aget_tuple(config)

        assert result is None

    @pytest.mark.asyncio
    async def test_aput_saves_checkpoint(self, checkpointer, mock_connection):
        """Test aput saves checkpoint to database."""
        mock_cursor = mock_connection.cursor.return_value

        config = {
            "configurable": {
                "thread_id": "test-thread",
                "checkpoint_ns": "",
                "checkpoint_id": "parent-id",
            }
        }
        checkpoint = {
            "id": "new-id",
            "ts": "2024-01-01",
            "channel_values": {},
            "channel_versions": {},
            "pending_sends": [],
        }
        metadata = {"step": 1}
        new_versions = {}

        result = await checkpointer.aput(config, checkpoint, metadata, new_versions)

        assert result["configurable"]["checkpoint_id"] == "new-id"
        assert result["configurable"]["thread_id"] == "test-thread"
        # Should have called execute for checkpoint upsert
        assert mock_cursor.execute.call_count >= 1


class TestCheckpointerConnectionFactory:
    """Tests for connection factory methods."""

    @patch("src.services.snowflake_checkpointer.is_running_in_spcs")
    @patch("src.services.snowflake_checkpointer.snowflake.connector.connect")
    def test_create_connection_local(self, mock_connect, mock_is_spcs):
        """Test create_connection uses local method when not in SPCS."""
        mock_is_spcs.return_value = False
        mock_connect.return_value = MagicMock()

        with patch("src.services.snowflake_checkpointer.settings") as mock_settings:
            mock_settings.snowflake_private_key_path = "/path/to/key.pem"
            mock_settings.snowflake_private_key_passphrase = None
            mock_settings.snowflake_account = "test-account"
            mock_settings.snowflake_user = "test-user"
            mock_settings.snowflake_database = "TEST_DB"
            mock_settings.snowflake_warehouse = "TEST_WH"

            # Mock the cryptography imports inside the method
            mock_key = MagicMock()
            mock_key.private_bytes.return_value = b"key-bytes"
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=b"key-data")))
            mock_file.__exit__ = MagicMock(return_value=False)

            with (
                patch("builtins.open", return_value=mock_file),
                patch(
                    "cryptography.hazmat.primitives.serialization.load_pem_private_key",
                    return_value=mock_key,
                ),
            ):
                CustomAsyncSnowflakeSaver.create_connection()

                mock_connect.assert_called_once()
                call_kwargs = mock_connect.call_args[1]
                assert call_kwargs["account"] == "test-account"
                assert call_kwargs["user"] == "test-user"

    @patch("src.services.snowflake_checkpointer.is_running_in_spcs")
    @patch("src.services.snowflake_checkpointer.Path")
    @patch("src.services.snowflake_checkpointer.os.getenv")
    @patch("src.services.snowflake_checkpointer.snowflake.connector.connect")
    def test_create_connection_spcs(self, mock_connect, mock_getenv, mock_path, mock_is_spcs):
        """Test create_connection uses SPCS method when in SPCS."""
        mock_is_spcs.return_value = True
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.read_text.return_value = "oauth-token-123"
        mock_path.return_value = mock_path_instance
        mock_getenv.return_value = "test.snowflakecomputing.com"
        mock_connect.return_value = MagicMock()

        with patch("src.services.snowflake_checkpointer.settings") as mock_settings:
            mock_settings.snowflake_database = "TEST_DB"
            mock_settings.snowflake_warehouse = "TEST_WH"

            CustomAsyncSnowflakeSaver.create_connection()

            mock_connect.assert_called_once()
            call_kwargs = mock_connect.call_args[1]
            assert call_kwargs["authenticator"] == "oauth"
            assert call_kwargs["token"] == "oauth-token-123"
