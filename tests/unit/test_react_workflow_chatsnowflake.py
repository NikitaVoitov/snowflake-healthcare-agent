"""Unit tests for Healthcare Agent with create_agent architecture.

Tests the helper functions and prompt generation that support the
LangChain v1 create_agent-based healthcare agent.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.graphs.react_prompts import build_system_message, format_conversation_history
from src.graphs.react_state import ConversationTurn, HealthcareAgentState
from src.graphs.react_workflow import (
    build_conversation_turn,
    extract_final_answer,
    extract_tools_used,
)


class TestExtractFinalAnswer:
    """Tests for extract_final_answer helper function."""

    def test_extracts_from_final_answer_field(self) -> None:
        """Should extract from final_answer field if present."""
        state: HealthcareAgentState = {
            "messages": [HumanMessage(content="test")],
            "user_query": "test",
            "member_id": None,
            "tenant_id": "test",
            "conversation_history": [],
            "iteration": 1,
            "max_iterations": 5,
            "execution_id": "test",
            "final_answer": "Pre-stored answer",
        }

        result = extract_final_answer(state)
        assert result == "Pre-stored answer"

    def test_extracts_from_last_ai_message(self) -> None:
        """Should extract from last AIMessage without tool_calls."""
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
        }

        result = extract_final_answer(state)
        assert result == "Your benefits include full coverage for preventive care."

    def test_skips_ai_messages_with_tool_calls(self) -> None:
        """Should skip AIMessages that have tool_calls."""
        state: HealthcareAgentState = {
            "messages": [
                HumanMessage(content="test"),
                AIMessage(
                    content="I need to query the database",
                    tool_calls=[{"id": "call_1", "name": "query_member_data", "args": {}}],
                ),
            ],
            "user_query": "test",
            "member_id": None,
            "tenant_id": "test",
            "conversation_history": [],
            "iteration": 1,
            "max_iterations": 5,
            "execution_id": "test",
            "final_answer": None,
        }

        result = extract_final_answer(state)
        assert result == "I could not find an answer to your question."

    def test_returns_default_when_no_messages(self) -> None:
        """Should return default message when no messages present."""
        state: HealthcareAgentState = {
            "messages": [],
            "user_query": "test",
            "member_id": None,
            "tenant_id": "test",
            "conversation_history": [],
            "iteration": 0,
            "max_iterations": 5,
            "execution_id": "test",
            "final_answer": None,
        }

        result = extract_final_answer(state)
        assert "could not find" in result.lower()


class TestExtractToolsUsed:
    """Tests for extract_tools_used helper function."""

    def test_extracts_tools_from_ai_messages(self) -> None:
        """Should extract tool names from AIMessage tool_calls."""
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
        }

        result = extract_tools_used(state)
        assert "query_member_data" in result
        assert "search_knowledge" in result

    def test_returns_empty_when_no_tools(self) -> None:
        """Should return empty list when no tools used."""
        state: HealthcareAgentState = {
            "messages": [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi there!"),
            ],
            "user_query": "Hello",
            "member_id": None,
            "tenant_id": "test",
            "conversation_history": [],
            "iteration": 1,
            "max_iterations": 5,
            "execution_id": "test",
            "final_answer": None,
        }

        result = extract_tools_used(state)
        assert result == []


class TestBuildConversationTurn:
    """Tests for build_conversation_turn helper function."""

    def test_builds_turn_with_all_fields(self) -> None:
        """Should build ConversationTurn with all fields populated."""
        state: HealthcareAgentState = {
            "messages": [
                HumanMessage(content="What are my claims?"),
                AIMessage(
                    content="Querying",
                    tool_calls=[{"id": "call_1", "name": "query_member_data", "args": {}}],
                ),
                ToolMessage(content="Claims data", tool_call_id="call_1"),
                AIMessage(content="You have 3 claims totaling $500."),
            ],
            "user_query": "What are my claims?",
            "member_id": "106742775",
            "tenant_id": "test",
            "conversation_history": [],
            "iteration": 2,
            "max_iterations": 5,
            "execution_id": "test",
            "final_answer": None,
        }

        turn = build_conversation_turn(state)

        assert turn["user_query"] == "What are my claims?"
        assert turn["member_id"] == "106742775"
        assert turn["final_answer"] == "You have 3 claims totaling $500."
        assert "query_member_data" in turn["tools_used"]


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
