"""Unit tests for individual graph node functions."""

import pytest

from src.graphs.state import HealthcareAgentState
from src.graphs.workflow import (
    async_planner_agent,
    async_analyst_agent,
    async_search_agent,
    async_response_agent,
    error_handler_node,
    route_to_agents,
)


class TestPlannerAgent:
    """Tests for async_planner_agent node."""

    @pytest.mark.asyncio
    async def test_routes_to_analyst_for_member_query(
        self, initial_state: HealthcareAgentState
    ) -> None:
        """Queries with member_id and claims keywords route to analyst."""
        state = {**initial_state, "user_query": "Show me my claims history"}
        result = await async_planner_agent(state)
        assert "plan" in result
        assert result["plan"] in ["analyst", "both"]

    @pytest.mark.asyncio
    async def test_routes_to_search_for_general_query(
        self, search_query_state: HealthcareAgentState
    ) -> None:
        """General policy questions without member context route to search."""
        state = {**search_query_state, "member_id": None}
        result = await async_planner_agent(state)
        assert "plan" in result
        assert result["plan"] in ["search", "both"]

    @pytest.mark.asyncio
    async def test_routes_both_for_complex_query(
        self, initial_state: HealthcareAgentState
    ) -> None:
        """Complex queries route to both."""
        state = {
            **initial_state,
            "user_query": "What are my coverage limits and policy details?",
        }
        result = await async_planner_agent(state)
        assert "plan" in result
        assert result["plan"] == "both"


class TestAnalystAgent:
    """Tests for async_analyst_agent node."""

    @pytest.mark.asyncio
    async def test_analyst_returns_results(
        self, analyst_query_state: HealthcareAgentState
    ) -> None:
        """Analyst agent should return structured results."""
        result = await async_analyst_agent(analyst_query_state)
        assert "analyst_results" in result
        assert result["analyst_results"] is None or isinstance(result["analyst_results"], dict)

    @pytest.mark.asyncio
    async def test_analyst_handles_missing_member_id(
        self, search_query_state: HealthcareAgentState
    ) -> None:
        """Analyst should handle missing member_id gracefully."""
        state = {**search_query_state, "member_id": None}
        result = await async_analyst_agent(state)
        assert isinstance(result, dict)


class TestSearchAgent:
    """Tests for async_search_agent node."""

    @pytest.mark.asyncio
    async def test_search_returns_results(
        self, search_query_state: HealthcareAgentState
    ) -> None:
        """Search agent should return list of results."""
        result = await async_search_agent(search_query_state)
        assert "search_results" in result
        assert result["search_results"] is None or isinstance(result["search_results"], list)


class TestResponseAgent:
    """Tests for async_response_agent node."""

    @pytest.mark.asyncio
    async def test_synthesizes_analyst_results(self) -> None:
        """Should synthesize response from analyst results."""
        state: HealthcareAgentState = {
            "user_query": "What are my claims?",
            "member_id": "106742775",
            "messages": [],
            "plan": "analyst",
            "analyst_results": {"claims": [{"claim_id": "CLM001"}]},
            "search_results": None,
            "final_response": "",
            "execution_id": "test",
            "thread_checkpoint_id": None,
            "current_step": "response",
            "error_count": 0,
            "max_steps": 5,
            "last_error": None,
            "has_error": False,
            "is_complete": False,
            "should_escalate": False,
        }
        result = await async_response_agent(state)
        assert "final_response" in result
        assert result["final_response"] != ""

    @pytest.mark.asyncio
    async def test_synthesizes_search_results(self) -> None:
        """Should synthesize response from search results."""
        state: HealthcareAgentState = {
            "user_query": "What is the copay policy?",
            "member_id": None,
            "messages": [],
            "plan": "search",
            "analyst_results": None,
            "search_results": [{"results": [{"answer": "Copay is $20"}]}],
            "final_response": "",
            "execution_id": "test",
            "thread_checkpoint_id": None,
            "current_step": "response",
            "error_count": 0,
            "max_steps": 5,
            "last_error": None,
            "has_error": False,
            "is_complete": False,
            "should_escalate": False,
        }
        result = await async_response_agent(state)
        assert "final_response" in result

    @pytest.mark.asyncio
    async def test_synthesizes_both_results(self) -> None:
        """Should combine analyst and search results."""
        state: HealthcareAgentState = {
            "user_query": "My claims and policy details",
            "member_id": "XYZ789",
            "messages": [],
            "plan": "both",
            "analyst_results": {"claims": [{"amount": 100.0}]},
            "search_results": [{"results": [{"content": "Policy info"}]}],
            "final_response": "",
            "execution_id": "test",
            "thread_checkpoint_id": None,
            "current_step": "response",
            "error_count": 0,
            "max_steps": 5,
            "last_error": None,
            "has_error": False,
            "is_complete": False,
            "should_escalate": False,
        }
        result = await async_response_agent(state)
        assert result["is_complete"] is True


class TestErrorHandler:
    """Tests for error_handler_node."""

    @pytest.mark.asyncio
    async def test_routes_to_retry_for_retriable_error(self) -> None:
        """Error handler should route to retry for retriable errors."""
        state: HealthcareAgentState = {
            "user_query": "test",
            "member_id": None,
            "messages": [],
            "plan": "both",
            "analyst_results": None,
            "search_results": None,
            "final_response": "",
            "execution_id": "test",
            "thread_checkpoint_id": None,
            "current_step": "error_handler",
            "error_count": 1,
            "max_steps": 5,
            "last_error": {"retriable": True, "message": "Timeout"},
            "has_error": True,
            "is_complete": False,
            "should_escalate": False,
        }
        result = await error_handler_node(state)
        assert result.get("current_step") == "retry"

    @pytest.mark.asyncio
    async def test_routes_to_escalate_for_non_retriable(self) -> None:
        """Error handler should escalate non-retriable errors."""
        state: HealthcareAgentState = {
            "user_query": "test",
            "member_id": None,
            "messages": [],
            "plan": "both",
            "analyst_results": None,
            "search_results": None,
            "final_response": "",
            "execution_id": "test",
            "thread_checkpoint_id": None,
            "current_step": "error_handler",
            "error_count": 3,
            "max_steps": 3,
            "last_error": {"retriable": False, "message": "Fatal error"},
            "has_error": True,
            "is_complete": False,
            "should_escalate": False,
        }
        result = await error_handler_node(state)
        assert result.get("should_escalate") is True

    @pytest.mark.asyncio
    async def test_escalates_after_max_retries(self) -> None:
        """Error handler should escalate after max retries."""
        state: HealthcareAgentState = {
            "user_query": "test",
            "member_id": None,
            "messages": [],
            "plan": "both",
            "analyst_results": None,
            "search_results": None,
            "final_response": "",
            "execution_id": "test",
            "thread_checkpoint_id": None,
            "current_step": "error_handler",
            "error_count": 10,
            "max_steps": 3,
            "last_error": {"retriable": True, "message": "Still failing"},
            "has_error": True,
            "is_complete": False,
            "should_escalate": False,
        }
        result = await error_handler_node(state)
        assert result.get("should_escalate") is True


class TestRouteToAgents:
    """Tests for route_to_agents conditional edge function."""

    def test_routes_to_analyst(self, analyst_query_state: HealthcareAgentState) -> None:
        """Should route to analyst node when plan is analyst."""
        result = route_to_agents(analyst_query_state)
        assert result == "analyst"

    def test_routes_to_search(self, search_query_state: HealthcareAgentState) -> None:
        """Should route to search node when plan is search."""
        result = route_to_agents(search_query_state)
        assert result == "search"

    def test_routes_to_parallel(self) -> None:
        """Should route to parallel execution when plan is both."""
        state: HealthcareAgentState = {
            "user_query": "test",
            "member_id": None,
            "messages": [],
            "plan": "both",
            "analyst_results": None,
            "search_results": None,
            "final_response": "",
            "execution_id": "test",
            "thread_checkpoint_id": None,
            "current_step": "planner",
            "error_count": 0,
            "max_steps": 5,
            "last_error": None,
            "has_error": False,
            "is_complete": False,
            "should_escalate": False,
        }
        result = route_to_agents(state)
        assert result == "parallel"

