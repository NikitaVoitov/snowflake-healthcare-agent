"""Integration tests for AgentService."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from src.models.requests import QueryRequest
from src.models.responses import AgentResponse
from src.services.react_agent_service import AgentService


class TestAgentServiceExecute:
    """Tests for AgentService.execute() method."""

    @pytest.fixture
    def agent_service(self, mock_compiled_graph: AsyncMock, mock_settings: MagicMock) -> AgentService:
        """Create AgentService with mocked dependencies."""
        return AgentService(mock_compiled_graph, mock_settings)

    @pytest.mark.asyncio
    async def test_execute_returns_agent_response(
        self, agent_service: AgentService, sample_query_request: QueryRequest
    ) -> None:
        """execute() should return AgentResponse."""
        result = await agent_service.execute(sample_query_request)
        assert isinstance(result, AgentResponse)
        assert result.output != ""

    @pytest.mark.asyncio
    async def test_execute_with_member_id(self, agent_service: AgentService) -> None:
        """execute() should handle member-specific queries."""
        request = QueryRequest(query="What are my claims?", member_id="106742775")
        result = await agent_service.execute(request)
        assert isinstance(result, AgentResponse)

    @pytest.mark.asyncio
    async def test_execute_without_member_id(self, agent_service: AgentService) -> None:
        """execute() should handle general queries without member_id."""
        request = QueryRequest(query="What is the policy for dental coverage?")
        result = await agent_service.execute(request)
        assert isinstance(result, AgentResponse)

    @pytest.mark.asyncio
    async def test_execute_empty_query_raises(self) -> None:
        """execute() should raise ValidationError for empty query."""
        with pytest.raises(ValidationError):
            QueryRequest(query="   ")

    @pytest.mark.asyncio
    async def test_execute_builds_correct_thread_id(
        self, mock_compiled_graph: AsyncMock, mock_settings: MagicMock
    ) -> None:
        """execute() should build thread_id from tenant and user."""
        service = AgentService(mock_compiled_graph, mock_settings)
        request = QueryRequest(query="test", tenant_id="acme", user_id="user123")
        await service.execute(request)
        call_kwargs = mock_compiled_graph.ainvoke.call_args.kwargs
        config = call_kwargs.get("config", {})
        thread_id = config.get("configurable", {}).get("thread_id", "")
        assert "acme" in thread_id
        assert "user123" in thread_id


class TestAgentServiceStream:
    """Tests for AgentService.stream() method."""

    @pytest.fixture
    def agent_service(self, mock_compiled_graph: AsyncMock, mock_settings: MagicMock) -> AgentService:
        """Create AgentService with mocked dependencies."""
        return AgentService(mock_compiled_graph, mock_settings)

    @pytest.mark.asyncio
    async def test_stream_yields_events(
        self, agent_service: AgentService, sample_query_request: QueryRequest
    ) -> None:
        """stream() should yield StreamEvent objects."""
        events = []
        async for event in agent_service.stream(sample_query_request):
            events.append(event)
        assert len(events) > 0
        assert events[0].event_type == "node_start"

    @pytest.mark.asyncio
    async def test_stream_emits_completion_event(
        self, agent_service: AgentService, sample_query_request: QueryRequest
    ) -> None:
        """stream() should emit completion event at end."""
        events = []
        async for event in agent_service.stream(sample_query_request):
            events.append(event)
        # Should have at least start event
        assert len(events) >= 1

    def test_empty_query_rejected_at_validation(self) -> None:
        """Empty query should be rejected by Pydantic validation before reaching stream()."""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(query="")

        # Verify the error is about query length
        error_dict = exc_info.value.errors()[0]
        assert error_dict["type"] == "string_too_short"
        assert error_dict["loc"] == ("query",)


class TestAgentServiceHelpers:
    """Tests for AgentService helper methods."""

    @pytest.fixture
    def agent_service(self, mock_compiled_graph: AsyncMock, mock_settings: MagicMock) -> AgentService:
        """Create AgentService with mocked dependencies."""
        return AgentService(mock_compiled_graph, mock_settings)

    def test_build_thread_id_format(self, agent_service: AgentService) -> None:
        """Thread ID should follow expected format."""
        thread_id = agent_service._build_thread_id("tenant1", "user1")
        assert "tenant1" in thread_id
        assert "user1" in thread_id
        assert len(thread_id) > len("tenant1:user1")

    def test_build_initial_state_has_messages(
        self, agent_service: AgentService, sample_query_request: QueryRequest
    ) -> None:
        """Initial state should include messages with HumanMessage."""
        state = agent_service._build_initial_state(sample_query_request, "test-exec-id")
        assert "messages" in state
        assert len(state["messages"]) == 1
        assert isinstance(state["messages"][0], HumanMessage)
        assert state["messages"][0].content == sample_query_request.query

    def test_build_initial_state_has_context(
        self, agent_service: AgentService, sample_query_request: QueryRequest
    ) -> None:
        """Initial state should include context fields."""
        state = agent_service._build_initial_state(sample_query_request, "test-exec-id")
        assert state["user_query"] == sample_query_request.query
        assert state["member_id"] == sample_query_request.member_id
        assert state["tenant_id"] == sample_query_request.tenant_id
        assert state["iteration"] == 0
        assert state["final_answer"] is None


class TestRoutingDetermination:
    """Tests for routing determination based on tools used."""

    @pytest.fixture
    def agent_service(self, mock_compiled_graph: AsyncMock, mock_settings: MagicMock) -> AgentService:
        """Create AgentService with mocked dependencies."""
        return AgentService(mock_compiled_graph, mock_settings)

    def test_routing_analyst_only(self, agent_service: AgentService) -> None:
        """Routing should be 'analyst' when only query_member_data used."""
        routing = agent_service._determine_routing(["query_member_data"])
        assert routing == "analyst"

    def test_routing_search_only(self, agent_service: AgentService) -> None:
        """Routing should be 'search' when only search_knowledge used."""
        routing = agent_service._determine_routing(["search_knowledge"])
        assert routing == "search"

    def test_routing_both(self, agent_service: AgentService) -> None:
        """Routing should be 'both' when both tools used."""
        routing = agent_service._determine_routing(["query_member_data", "search_knowledge"])
        assert routing == "both"

    def test_routing_no_tools(self, agent_service: AgentService) -> None:
        """Routing should be 'react' when no tools used."""
        routing = agent_service._determine_routing([])
        assert routing == "react"
