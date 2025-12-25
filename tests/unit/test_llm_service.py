"""Unit tests for LLM service with ChatSnowflake - ASYNC VERSION."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.llm_service import (
    _create_local_chat_model,
    _create_spcs_chat_model,
    get_chat_snowflake,
    get_chat_snowflake_with_structured_output,
    get_session,
    set_session,
)


class TestGetChatSnowflake:
    """Tests for get_chat_snowflake factory function."""

    @pytest.mark.asyncio
    @patch("src.services.llm_service._get_or_create_session")
    @patch("src.services.llm_service.settings")
    @patch("src.services.llm_service.ChatSnowflake")
    async def test_creates_local_model_when_not_spcs(
        self,
        mock_chat_snowflake: MagicMock,
        mock_settings: MagicMock,
        mock_get_session: AsyncMock,
    ) -> None:
        """Should create local model when not in SPCS environment."""
        # Arrange
        mock_settings.is_spcs = False
        mock_settings.snowflake_account = "test_account"
        mock_settings.snowflake_user = "test_user"
        mock_settings.snowflake_private_key_path = "/path/to/key.pem"
        mock_settings.snowflake_private_key_passphrase = None
        mock_settings.snowflake_database = "HEALTHCARE_DB"
        mock_settings.snowflake_warehouse = "TEST_WH"
        mock_settings.snowflake_role = "TEST_ROLE"
        mock_settings.cortex_llm_model = "claude-3-5-sonnet"

        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        mock_llm = MagicMock()
        mock_chat_snowflake.return_value = mock_llm

        # Act
        result = await get_chat_snowflake()

        # Assert
        mock_get_session.assert_called_once()
        mock_chat_snowflake.assert_called_once()
        call_kwargs = mock_chat_snowflake.call_args.kwargs
        assert call_kwargs["session"] == mock_session
        assert call_kwargs["model"] == "claude-3-5-sonnet"
        assert result == mock_llm

    @pytest.mark.asyncio
    @patch("src.services.llm_service.settings")
    @patch("src.services.llm_service.ChatSnowflake")
    async def test_creates_spcs_model_when_in_spcs(
        self,
        mock_chat_snowflake: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Should create SPCS model using pre-set session when in SPCS."""
        # Import module to reset cached session
        import src.services.llm_service as llm_service

        # Arrange
        mock_settings.is_spcs = True
        mock_settings.cortex_llm_model = "claude-3-5-sonnet"

        # Set session like main.py does during startup
        mock_session = MagicMock()
        llm_service._cached_session = mock_session

        mock_llm = MagicMock()
        mock_chat_snowflake.return_value = mock_llm

        try:
            # Act
            result = await get_chat_snowflake()

            # Assert
            mock_chat_snowflake.assert_called_once()
            call_kwargs = mock_chat_snowflake.call_args.kwargs
            assert call_kwargs["session"] == mock_session
            assert call_kwargs["model"] == "claude-3-5-sonnet"
            assert result == mock_llm
        finally:
            # Cleanup
            llm_service._cached_session = None

    @pytest.mark.asyncio
    @patch("src.services.llm_service._get_or_create_session")
    @patch("src.services.llm_service.settings")
    @patch("src.services.llm_service.ChatSnowflake")
    async def test_binds_tools_when_provided(
        self,
        mock_chat_snowflake: MagicMock,
        mock_settings: MagicMock,
        mock_get_session: AsyncMock,
    ) -> None:
        """Should bind tools to model when tools are provided."""
        # Arrange
        mock_settings.is_spcs = False
        mock_settings.snowflake_account = "test_account"
        mock_settings.snowflake_user = "test_user"
        mock_settings.snowflake_private_key_path = "/path/to/key.pem"
        mock_settings.snowflake_private_key_passphrase = None
        mock_settings.snowflake_database = "HEALTHCARE_DB"
        mock_settings.snowflake_warehouse = "TEST_WH"
        mock_settings.snowflake_role = "TEST_ROLE"
        mock_settings.cortex_llm_model = "claude-3-5-sonnet"

        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        mock_llm = MagicMock()
        mock_llm_with_tools = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm_with_tools
        mock_chat_snowflake.return_value = mock_llm

        mock_tool = MagicMock()
        mock_tool.name = "test_tool"

        # Act
        result = await get_chat_snowflake(tools=[mock_tool])

        # Assert
        mock_llm.bind_tools.assert_called_once_with([mock_tool])
        assert result == mock_llm_with_tools

    @pytest.mark.asyncio
    @patch("src.services.llm_service._get_or_create_session")
    @patch("src.services.llm_service.settings")
    @patch("src.services.llm_service.ChatSnowflake")
    async def test_uses_custom_model_when_provided(
        self,
        mock_chat_snowflake: MagicMock,
        mock_settings: MagicMock,
        mock_get_session: AsyncMock,
    ) -> None:
        """Should use custom model name when provided."""
        # Arrange
        mock_settings.is_spcs = False
        mock_settings.snowflake_account = "test_account"
        mock_settings.snowflake_user = "test_user"
        mock_settings.snowflake_private_key_path = "/path/to/key.pem"
        mock_settings.snowflake_private_key_passphrase = None
        mock_settings.snowflake_database = "HEALTHCARE_DB"
        mock_settings.snowflake_warehouse = "TEST_WH"
        mock_settings.snowflake_role = "TEST_ROLE"
        mock_settings.cortex_llm_model = "claude-3-5-sonnet"  # Default

        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        mock_llm = MagicMock()
        mock_chat_snowflake.return_value = mock_llm

        # Act
        result = await get_chat_snowflake(model="mistral-large2")

        # Assert
        call_kwargs = mock_chat_snowflake.call_args.kwargs
        assert call_kwargs["model"] == "mistral-large2"

    @pytest.mark.asyncio
    @patch("src.services.llm_service._get_or_create_session")
    @patch("src.services.llm_service.settings")
    @patch("src.services.llm_service.ChatSnowflake")
    async def test_uses_custom_temperature_when_provided(
        self,
        mock_chat_snowflake: MagicMock,
        mock_settings: MagicMock,
        mock_get_session: AsyncMock,
    ) -> None:
        """Should use custom temperature when provided."""
        # Arrange
        mock_settings.is_spcs = False
        mock_settings.snowflake_account = "test_account"
        mock_settings.snowflake_user = "test_user"
        mock_settings.snowflake_private_key_path = "/path/to/key.pem"
        mock_settings.snowflake_private_key_passphrase = None
        mock_settings.snowflake_database = "HEALTHCARE_DB"
        mock_settings.snowflake_warehouse = "TEST_WH"
        mock_settings.snowflake_role = "TEST_ROLE"
        mock_settings.cortex_llm_model = "claude-3-5-sonnet"

        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        mock_llm = MagicMock()
        mock_chat_snowflake.return_value = mock_llm

        # Act
        result = await get_chat_snowflake(temperature=0.7)

        # Assert
        call_kwargs = mock_chat_snowflake.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7


class TestCreateLocalChatModel:
    """Tests for _create_local_chat_model helper."""

    @pytest.mark.asyncio
    @patch("src.services.llm_service.settings")
    async def test_raises_when_account_missing(self, mock_settings: MagicMock) -> None:
        """Should raise ValueError when account is missing."""
        mock_settings.snowflake_account = None

        with pytest.raises(ValueError, match="SNOWFLAKE_ACCOUNT required"):
            await _create_local_chat_model("claude-3-5-sonnet", 0)

    @pytest.mark.asyncio
    @patch("src.services.llm_service.settings")
    async def test_raises_when_user_missing(self, mock_settings: MagicMock) -> None:
        """Should raise ValueError when user is missing."""
        mock_settings.snowflake_account = "test_account"
        mock_settings.snowflake_user = None

        with pytest.raises(ValueError, match="SNOWFLAKE_USER required"):
            await _create_local_chat_model("claude-3-5-sonnet", 0)

    @pytest.mark.asyncio
    @patch("src.services.llm_service.settings")
    async def test_raises_when_key_path_missing(self, mock_settings: MagicMock) -> None:
        """Should raise ValueError when private key path is missing."""
        mock_settings.snowflake_account = "test_account"
        mock_settings.snowflake_user = "test_user"
        mock_settings.snowflake_private_key_path = None

        with pytest.raises(ValueError, match="SNOWFLAKE_PRIVATE_KEY_PATH required"):
            await _create_local_chat_model("claude-3-5-sonnet", 0)


class TestCreateSpcsChatModel:
    """Tests for _create_spcs_chat_model helper."""

    @pytest.mark.asyncio
    async def test_raises_when_session_not_set(self) -> None:
        """Should raise ValueError when no session has been set."""
        # Import module to ensure clean state
        import src.services.llm_service as llm_service

        # Ensure no session is cached
        original_session = llm_service._cached_session
        llm_service._cached_session = None

        try:
            with pytest.raises(ValueError, match="No Snowpark session available"):
                await _create_spcs_chat_model("claude-3-5-sonnet", 0)
        finally:
            # Restore original state
            llm_service._cached_session = original_session


class TestSetSession:
    """Tests for set_session and get_session functions."""

    def test_set_and_get_session(self) -> None:
        """Should set and retrieve the cached session."""
        import src.services.llm_service as llm_service

        # Save original
        original = llm_service._cached_session

        try:
            mock_session = MagicMock()
            set_session(mock_session)
            assert get_session() == mock_session
        finally:
            # Restore
            llm_service._cached_session = original


class TestGetChatSnowflakeWithStructuredOutput:
    """Tests for get_chat_snowflake_with_structured_output."""

    @pytest.mark.asyncio
    @patch("src.services.llm_service.get_chat_snowflake")
    async def test_applies_structured_output(self, mock_get_chat: AsyncMock) -> None:
        """Should apply structured output schema."""
        from pydantic import BaseModel

        class TestOutput(BaseModel):
            answer: str

        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_get_chat.return_value = mock_llm

        # Act
        result = await get_chat_snowflake_with_structured_output(TestOutput)

        # Assert
        mock_llm.with_structured_output.assert_called_once_with(TestOutput)
        assert result == mock_structured_llm
