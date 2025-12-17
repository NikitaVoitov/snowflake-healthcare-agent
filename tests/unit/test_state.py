"""Unit tests for LangGraph state definitions."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.graphs.state import HealthcareAgentState


class TestHealthcareAgentState:
    """Tests for HealthcareAgentState TypedDict."""

    def test_minimal_state_creation(self) -> None:
        """Should create state with required fields."""
        state: HealthcareAgentState = {
            "user_query": "test query",
            "member_id": None,
            "messages": [],
            "plan": "",
            "analyst_results": None,
            "search_results": None,
            "final_response": "",
            "execution_id": "test-001",
            "thread_checkpoint_id": None,
            "current_step": "start",
            "error_count": 0,
            "max_steps": 5,
            "last_error": None,
            "has_error": False,
            "is_complete": False,
            "should_escalate": False,
        }
        assert state["user_query"] == "test query"

    def test_state_with_member_id(self) -> None:
        """Should accept member_id for member-specific queries."""
        state: HealthcareAgentState = {
            "user_query": "What are my claims?",
            "member_id": "106742775",
            "messages": [],
            "plan": "analyst",
            "analyst_results": None,
            "search_results": None,
            "final_response": "",
            "execution_id": "test-002",
            "thread_checkpoint_id": None,
            "current_step": "analyst",
            "error_count": 0,
            "max_steps": 5,
            "last_error": None,
            "has_error": False,
            "is_complete": False,
            "should_escalate": False,
        }
        assert state["member_id"] == "106742775"

    def test_state_update_immutability(self, initial_state: HealthcareAgentState) -> None:
        """State updates should create new dict, not mutate."""
        original_query = initial_state["user_query"]
        update = {"plan": "search", "current_step": "search"}
        new_state = {**initial_state, **update}

        assert initial_state["plan"] == ""
        assert new_state["plan"] == "search"
        assert new_state["user_query"] == original_query

    def test_messages_list_accumulation(self) -> None:
        """Messages should accumulate."""
        initial_messages = [HumanMessage(content="Hello")]
        new_messages = [AIMessage(content="Hi there!")]
        accumulated = initial_messages + new_messages

        state: HealthcareAgentState = {
            "user_query": "test",
            "member_id": None,
            "messages": accumulated,
            "plan": "",
            "analyst_results": None,
            "search_results": None,
            "final_response": "",
            "execution_id": "test",
            "thread_checkpoint_id": None,
            "current_step": "start",
            "error_count": 0,
            "max_steps": 5,
            "last_error": None,
            "has_error": False,
            "is_complete": False,
            "should_escalate": False,
        }
        assert len(state["messages"]) == 2

    def test_error_state_tracking(self) -> None:
        """Error tracking fields should work together."""
        state: HealthcareAgentState = {
            "user_query": "failing query",
            "member_id": None,
            "messages": [],
            "plan": "both",
            "analyst_results": None,
            "search_results": None,
            "final_response": "",
            "execution_id": "test-error",
            "thread_checkpoint_id": None,
            "current_step": "error_handler",
            "error_count": 2,
            "max_steps": 3,
            "last_error": {"error_type": "cortex_api", "message": "Rate limit"},
            "has_error": True,
            "is_complete": False,
            "should_escalate": False,
        }
        assert state["has_error"] is True
        assert state["error_count"] == 2

    def test_completion_state(self) -> None:
        """Completed state should have final_response and is_complete."""
        state: HealthcareAgentState = {
            "user_query": "What is my deductible?",
            "member_id": "106742775",
            "messages": [],
            "plan": "analyst",
            "analyst_results": {"deductible": 500.00},
            "search_results": None,
            "final_response": "Your deductible is $500.00",
            "execution_id": "test-complete",
            "thread_checkpoint_id": "chk-001",
            "current_step": "complete",
            "error_count": 0,
            "max_steps": 5,
            "last_error": None,
            "has_error": False,
            "is_complete": True,
            "should_escalate": False,
        }
        assert state["is_complete"] is True
        assert state["final_response"] != ""

    def test_parallel_execution_results(self) -> None:
        """Both analyst and search results can be populated."""
        state: HealthcareAgentState = {
            "user_query": "coverage and policy questions",
            "member_id": "XYZ789",
            "messages": [],
            "plan": "both",
            "analyst_results": {"claims": [{"amount": 100.0}]},
            "search_results": [{"results": [{"content": "Policy..."}]}],
            "final_response": "Based on your records...",
            "execution_id": "test-both",
            "thread_checkpoint_id": "chk-002",
            "current_step": "response",
            "error_count": 0,
            "max_steps": 5,
            "last_error": None,
            "has_error": False,
            "is_complete": False,
            "should_escalate": False,
        }
        assert state["analyst_results"] is not None
        assert state["search_results"] is not None


class TestStateFixtures:
    """Tests using conftest fixtures."""

    def test_initial_state_fixture(self, initial_state: HealthcareAgentState) -> None:
        """initial_state fixture should be valid."""
        assert initial_state["user_query"] == "What are my benefits?"

    def test_analyst_query_state_fixture(self, analyst_query_state: HealthcareAgentState) -> None:
        """analyst_query_state fixture should be routed to analyst."""
        assert analyst_query_state["plan"] == "analyst"

    def test_search_query_state_fixture(self, search_query_state: HealthcareAgentState) -> None:
        """search_query_state fixture should be routed to search."""
        assert search_query_state["plan"] == "search"

    def test_error_state_fixture(self, error_state: HealthcareAgentState) -> None:
        """error_state fixture should have error condition."""
        assert error_state["has_error"] is True

