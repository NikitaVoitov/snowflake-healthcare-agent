"""Unit tests for ReAct workflow with ChatSnowflake integration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.graphs.react_prompts import build_system_message, format_conversation_history
from src.graphs.react_state import ConversationTurn, HealthcareAgentState
from src.graphs.react_workflow import (
    _build_chat_messages,
    final_answer_node,
    model_node,
    route_after_model,
)


class TestBuildChatMessages:
    """Tests for _build_chat_messages helper function."""

    def test_creates_system_message_first(self, initial_state: HealthcareAgentState) -> None:
        """Should create system message as first message."""
        messages = _build_chat_messages(initial_state)

        assert len(messages) >= 1
        assert isinstance(messages[0], SystemMessage)
        assert "healthcare" in messages[0].content.lower()

    def test_includes_existing_messages(self, initial_state: HealthcareAgentState) -> None:
        """Should include existing messages from state after system message."""
        messages = _build_chat_messages(initial_state)

        # System message + HumanMessage from state
        assert len(messages) == 2
        assert isinstance(messages[1], HumanMessage)
        assert messages[1].content == "What are my benefits?"

    def test_includes_member_context_in_system_message(
        self, initial_state: HealthcareAgentState
    ) -> None:
        """Should include member ID in system message when provided."""
        messages = _build_chat_messages(initial_state)

        system_content = messages[0].content
        assert "106742775" in system_content

    def test_includes_conversation_history_in_system_message(
        self, state_with_history: HealthcareAgentState
    ) -> None:
        """Should include conversation history in system message."""
        messages = _build_chat_messages(state_with_history)

        system_content = messages[0].content
        assert "Gold PPO" in system_content or "106742775" in system_content


class TestModelNode:
    """Tests for model_node with ChatSnowflake."""

    @pytest.mark.asyncio
    @patch("src.graphs.react_workflow._ensure_snowpark_session_async")
    @patch("src.graphs.react_workflow.get_chat_snowflake")
    async def test_calls_chat_snowflake(
        self, mock_get_chat: AsyncMock, mock_ensure_session: AsyncMock, initial_state: HealthcareAgentState
    ) -> None:
        """Should call ChatSnowflake with correct messages."""
        # Arrange
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="Test response")
        mock_get_chat.return_value = mock_llm
        mock_ensure_session.return_value = None

        # Act
        result = await model_node(initial_state)

        # Assert
        mock_ensure_session.assert_called_once()
        mock_get_chat.assert_called_once()
        mock_llm.ainvoke.assert_called_once()
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)

    @pytest.mark.asyncio
    @patch("src.graphs.react_workflow._ensure_snowpark_session_async")
    @patch("src.graphs.react_workflow.get_chat_snowflake")
    async def test_returns_ai_message_with_tool_calls(
        self, mock_get_chat: AsyncMock, mock_ensure_session: AsyncMock, initial_state: HealthcareAgentState
    ) -> None:
        """Should return AIMessage with tool_calls when model decides to use tool."""
        # Arrange
        ai_message = AIMessage(
            content="I need to query the database",
            tool_calls=[
                {
                    "id": "call_123",
                    "name": "query_member_data",
                    "args": {"query": "benefits", "member_id": "106742775"},
                }
            ],
        )
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = ai_message
        mock_get_chat.return_value = mock_llm
        mock_ensure_session.return_value = None

        # Act
        result = await model_node(initial_state)

        # Assert
        assert result["messages"][0].tool_calls is not None
        assert len(result["messages"][0].tool_calls) == 1
        assert result["messages"][0].tool_calls[0]["name"] == "query_member_data"

    @pytest.mark.asyncio
    @patch("src.graphs.react_workflow._ensure_snowpark_session_async")
    @patch("src.graphs.react_workflow.get_chat_snowflake")
    async def test_increments_iteration(
        self, mock_get_chat: AsyncMock, mock_ensure_session: AsyncMock, initial_state: HealthcareAgentState
    ) -> None:
        """Should increment iteration count."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="Response")
        mock_get_chat.return_value = mock_llm
        mock_ensure_session.return_value = None

        result = await model_node(initial_state)

        assert result["iteration"] == 1

    @pytest.mark.asyncio
    @patch("src.graphs.react_workflow._ensure_snowpark_session_async")
    @patch("src.graphs.react_workflow.get_chat_snowflake")
    async def test_handles_max_iterations(
        self, mock_get_chat: AsyncMock, mock_ensure_session: AsyncMock, initial_state: HealthcareAgentState
    ) -> None:
        """Should synthesize answer when max iterations reached."""
        mock_ensure_session.return_value = None
        # Set iteration to max
        initial_state["iteration"] = 5
        initial_state["max_iterations"] = 5

        result = await model_node(initial_state)

        # Should not call LLM, should synthesize answer
        mock_get_chat.assert_not_called()
        assert "messages" in result
        assert "reached my processing limit" in result["messages"][0].content.lower()

    @pytest.mark.asyncio
    @patch("src.graphs.react_workflow._ensure_snowpark_session_async")
    @patch("src.graphs.react_workflow.get_chat_snowflake")
    async def test_handles_max_iterations_with_tool_data(
        self, mock_get_chat: AsyncMock, mock_ensure_session: AsyncMock
    ) -> None:
        """Should use tool data to synthesize answer at max iterations."""
        mock_ensure_session.return_value = None
        state: HealthcareAgentState = {
            "messages": [
                HumanMessage(content="What are my claims?"),
                AIMessage(
                    content="Querying",
                    tool_calls=[{"id": "call_1", "name": "query_member_data", "args": {}}],
                ),
                ToolMessage(content="Claims found: 3 claims totaling $500", tool_call_id="call_1"),
            ],
            "user_query": "What are my claims?",
            "member_id": "106742775",
            "tenant_id": "test",
            "conversation_history": [],
            "iteration": 5,
            "max_iterations": 5,
            "execution_id": "test",
            "final_answer": None,
            "error_count": 0,
            "last_error": None,
            "has_error": False,
        }

        result = await model_node(state)

        # Should include tool data in synthesized answer
        assert "Claims found" in result["messages"][0].content or "analysis" in result["messages"][0].content.lower()

    @pytest.mark.asyncio
    @patch("src.graphs.react_workflow._ensure_snowpark_session_async")
    @patch("src.graphs.react_workflow.get_chat_snowflake")
    async def test_handles_llm_error(
        self, mock_get_chat: AsyncMock, mock_ensure_session: AsyncMock, initial_state: HealthcareAgentState
    ) -> None:
        """Should handle LLM errors gracefully."""
        mock_ensure_session.return_value = None
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = Exception("API Error")
        mock_get_chat.return_value = mock_llm

        result = await model_node(initial_state)

        assert result["has_error"] is True
        assert result["last_error"]["error_type"] == "llm_error"
        assert "error" in result["messages"][0].content.lower()


class TestRouteAfterModel:
    """Tests for route_after_model routing function."""

    def test_routes_to_tools_when_tool_calls_present(self) -> None:
        """Should route to tools when AIMessage has tool_calls."""
        state: HealthcareAgentState = {
            "messages": [
                AIMessage(
                    content="Need to query",
                    tool_calls=[{"id": "call_1", "name": "query_member_data", "args": {}}],
                )
            ],
            "user_query": "test",
            "member_id": None,
            "tenant_id": "test",
            "conversation_history": [],
            "iteration": 1,
            "max_iterations": 5,
            "execution_id": "test",
            "final_answer": None,
            "error_count": 0,
            "last_error": None,
            "has_error": False,
        }

        result = route_after_model(state)
        assert result == "tools"

    def test_routes_to_final_answer_when_no_tool_calls(self) -> None:
        """Should route to final_answer when AIMessage has no tool_calls."""
        state: HealthcareAgentState = {
            "messages": [AIMessage(content="Here is your answer")],
            "user_query": "test",
            "member_id": None,
            "tenant_id": "test",
            "conversation_history": [],
            "iteration": 1,
            "max_iterations": 5,
            "execution_id": "test",
            "final_answer": None,
            "error_count": 0,
            "last_error": None,
            "has_error": False,
        }

        result = route_after_model(state)
        assert result == "final_answer"

    def test_routes_to_error_when_has_error(self) -> None:
        """Should route to error when has_error is True."""
        state: HealthcareAgentState = {
            "messages": [AIMessage(content="Error occurred")],
            "user_query": "test",
            "member_id": None,
            "tenant_id": "test",
            "conversation_history": [],
            "iteration": 1,
            "max_iterations": 5,
            "execution_id": "test",
            "final_answer": None,
            "error_count": 1,
            "last_error": {"error_type": "test"},
            "has_error": True,
        }

        result = route_after_model(state)
        assert result == "error"

    def test_routes_to_error_when_no_messages(self) -> None:
        """Should route to error when messages list is empty."""
        state: HealthcareAgentState = {
            "messages": [],
            "user_query": "test",
            "member_id": None,
            "tenant_id": "test",
            "conversation_history": [],
            "iteration": 1,
            "max_iterations": 5,
            "execution_id": "test",
            "final_answer": None,
            "error_count": 0,
            "last_error": None,
            "has_error": False,
        }

        result = route_after_model(state)
        assert result == "error"


class TestFinalAnswerNode:
    """Tests for final_answer_node."""

    @pytest.mark.asyncio
    async def test_extracts_final_answer(self) -> None:
        """Should extract final answer from last AIMessage without tool_calls."""
        state: HealthcareAgentState = {
            "messages": [
                HumanMessage(content="What are my benefits?"),
                AIMessage(
                    content="Querying",
                    tool_calls=[{"id": "call_1", "name": "query_member_data", "args": {}}],
                ),
                ToolMessage(content="Benefits data", tool_call_id="call_1"),
                AIMessage(content="Your benefits include full coverage for preventive care."),
            ],
            "user_query": "What are my benefits?",
            "member_id": "106742775",
            "tenant_id": "test",
            "conversation_history": [],
            "iteration": 2,
            "max_iterations": 5,
            "execution_id": "test",
            "final_answer": None,
            "error_count": 0,
            "last_error": None,
            "has_error": False,
        }

        result = await final_answer_node(state)

        assert result["final_answer"] == "Your benefits include full coverage for preventive care."

    @pytest.mark.asyncio
    async def test_collects_tools_used(self) -> None:
        """Should collect all tools used in conversation."""
        state: HealthcareAgentState = {
            "messages": [
                HumanMessage(content="Query"),
                AIMessage(
                    content="",
                    tool_calls=[{"id": "call_1", "name": "query_member_data", "args": {}}],
                ),
                ToolMessage(content="Data", tool_call_id="call_1"),
                AIMessage(
                    content="",
                    tool_calls=[{"id": "call_2", "name": "search_knowledge", "args": {}}],
                ),
                ToolMessage(content="Search results", tool_call_id="call_2"),
                AIMessage(content="Final answer"),
            ],
            "user_query": "Query",
            "member_id": None,
            "tenant_id": "test",
            "conversation_history": [],
            "iteration": 3,
            "max_iterations": 5,
            "execution_id": "test",
            "final_answer": None,
            "error_count": 0,
            "last_error": None,
            "has_error": False,
        }

        result = await final_answer_node(state)

        assert len(result["conversation_history"]) == 1
        turn = result["conversation_history"][0]
        assert "query_member_data" in turn["tools_used"]
        assert "search_knowledge" in turn["tools_used"]


class TestBuildSystemMessage:
    """Tests for build_system_message helper."""

    def test_includes_base_prompt(self) -> None:
        """Should include healthcare assistant base prompt."""
        message = build_system_message()

        assert "healthcare" in message.lower()
        assert "assistant" in message.lower()

    def test_includes_member_id_when_provided(self) -> None:
        """Should include member ID in context."""
        message = build_system_message(member_id="123456789")

        assert "123456789" in message

    def test_includes_member_id_from_history(self) -> None:
        """Should extract member ID from conversation history."""
        history = [
            ConversationTurn(
                user_query="Tell me about member 987654321",
                member_id="987654321",
                final_answer="Member info...",
                tools_used=["query_member_data"],
            )
        ]

        message = build_system_message(conversation_history=history)

        assert "987654321" in message

    def test_includes_conversation_history(self) -> None:
        """Should include formatted conversation history."""
        history = [
            ConversationTurn(
                user_query="What is my plan?",
                member_id="106742775",
                final_answer="You have the Gold PPO plan.",
                tools_used=["query_member_data"],
            )
        ]

        message = build_system_message(conversation_history=history)

        assert "Gold PPO" in message or "What is my plan" in message


class TestFormatConversationHistory:
    """Tests for format_conversation_history helper."""

    def test_returns_empty_for_no_history(self) -> None:
        """Should return empty string when no history."""
        result = format_conversation_history([])
        assert result == ""

    def test_formats_single_turn(self) -> None:
        """Should format a single conversation turn."""
        history = [
            ConversationTurn(
                user_query="What is my deductible?",
                member_id="106742775",
                final_answer="Your deductible is $500.",
                tools_used=["query_member_data"],
            )
        ]

        result = format_conversation_history(history)

        assert "deductible" in result.lower()
        assert "106742775" in result

    def test_truncates_long_answers(self) -> None:
        """Should truncate long answers."""
        long_answer = "A" * 300
        history = [
            ConversationTurn(
                user_query="Query",
                member_id=None,
                final_answer=long_answer,
                tools_used=[],
            )
        ]

        result = format_conversation_history(history)

        assert "..." in result
        assert len(result) < len(long_answer) + 100

    def test_limits_to_last_5_turns(self) -> None:
        """Should only show last 5 turns."""
        history = [
            ConversationTurn(
                user_query=f"Query {i}",
                member_id=None,
                final_answer=f"Answer {i}",
                tools_used=[],
            )
            for i in range(10)
        ]

        result = format_conversation_history(history)

        # Should not include early turns
        assert "Query 0" not in result
        assert "Query 4" not in result
        # Should include later turns
        assert "Query 9" in result

