"""Unit tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from src.models.agent_types import ErrorDetail, ErrorType, RoutingDecision
from src.models.base import BaseSchema
from src.models.requests import QueryRequest, StreamRequest
from src.models.responses import AgentResponse, HealthResponse, StreamEvent


class TestBaseSchema:
    """Tests for BaseSchema configuration."""

    def test_camel_case_serialization(self) -> None:
        """BaseSchema should serialize to camelCase."""

        class TestModel(BaseSchema):
            user_name: str
            is_active: bool

        model = TestModel(user_name="test", is_active=True)
        data = model.model_dump(by_alias=True)

        assert "userName" in data
        assert "isActive" in data

    def test_populate_by_name(self) -> None:
        """BaseSchema should accept both snake_case and camelCase."""

        class TestModel(BaseSchema):
            user_name: str

        model1 = TestModel(user_name="test1")
        model2 = TestModel(userName="test2")
        assert model1.user_name == "test1"
        assert model2.user_name == "test2"

    def test_whitespace_stripping(self) -> None:
        """BaseSchema should strip whitespace from strings."""

        class TestModel(BaseSchema):
            value: str

        model = TestModel(value="  padded  ")
        assert model.value == "padded"


class TestQueryRequest:
    """Tests for QueryRequest validation."""

    def test_valid_query_request(self, sample_query_request: QueryRequest) -> None:
        """Valid request should pass validation."""
        assert sample_query_request.query is not None
        assert len(sample_query_request.query) > 0

    def test_query_min_length(self) -> None:
        """Query must have minimum length."""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(query="")
        assert "string_too_short" in str(exc_info.value).lower() or "1 character" in str(exc_info.value)

    def test_query_max_length(self) -> None:
        """Query must not exceed max length."""
        with pytest.raises(ValidationError):
            QueryRequest(query="x" * 5001)

    def test_member_id_pattern_valid(self) -> None:
        """Valid member ID patterns should pass."""
        request = QueryRequest(query="test", member_id="ABC123")
        assert request.member_id == "ABC123"

    def test_member_id_pattern_invalid(self) -> None:
        """Invalid member ID patterns should fail."""
        with pytest.raises(ValidationError):
            QueryRequest(query="test", member_id="123ABC")

    def test_member_id_optional(self) -> None:
        """Member ID should be optional."""
        request = QueryRequest(query="general question")
        assert request.member_id is None

    def test_default_values(self) -> None:
        """Default values should be applied."""
        request = QueryRequest(query="test query")
        assert request.tenant_id == "default"
        assert request.user_id == "anonymous"


class TestStreamRequest:
    """Tests for StreamRequest validation."""

    def test_inherits_query_request(self) -> None:
        """StreamRequest should inherit QueryRequest fields."""
        request = StreamRequest(query="test", member_id="ABC123")
        assert request.query == "test"

    def test_include_intermediate_default(self) -> None:
        """include_intermediate should default to True."""
        request = StreamRequest(query="test")
        assert request.include_intermediate is True


class TestAgentResponse:
    """Tests for AgentResponse model."""

    def test_valid_response(self, sample_agent_response: AgentResponse) -> None:
        """Valid response should serialize correctly."""
        data = sample_agent_response.model_dump()
        assert "output" in data

    def test_camel_case_output(self, sample_agent_response: AgentResponse) -> None:
        """Response should serialize to camelCase."""
        data = sample_agent_response.model_dump(by_alias=True)
        assert "executionId" in data


class TestStreamEvent:
    """Tests for StreamEvent model."""

    def test_node_start_event(self) -> None:
        """Node start event should include node name."""
        event = StreamEvent(event_type="node_start", node="planner", data={"step": 1})
        assert event.event_type == "node_start"

    def test_node_end_event(self) -> None:
        """Node end event should include results."""
        event = StreamEvent(event_type="node_end", node="analyst", data={"results": {}})
        assert event.data["results"] == {}


class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_healthy_response(self) -> None:
        """Healthy status should serialize correctly."""
        response = HealthResponse(status="healthy", version="0.1.0", service="API")
        assert response.status == "healthy"


class TestErrorType:
    """Tests for ErrorType enum."""

    def test_all_error_types(self) -> None:
        """All expected error types should exist."""
        assert ErrorType.VALIDATION.value == "validation"
        assert ErrorType.CORTEX_API.value == "cortex_api"
        assert ErrorType.TIMEOUT.value == "timeout"


class TestErrorDetail:
    """Tests for ErrorDetail dataclass."""

    def test_error_detail_creation(self) -> None:
        """ErrorDetail should store all fields."""
        error = ErrorDetail(
            error_type=ErrorType.CORTEX_API,
            message="Service unavailable",
            node="analyst",
            timestamp="2024-01-01T00:00:00Z",
            retriable=True,
            fallback_action="use_cache",
        )
        assert error.retriable is True

    def test_to_dict(self) -> None:
        """to_dict should return serializable dictionary."""
        error = ErrorDetail(
            error_type=ErrorType.TIMEOUT,
            message="Operation timed out",
            node="search",
            timestamp="2024-01-01T00:00:00Z",
            retriable=False,
            fallback_action=None,
        )
        assert error.to_dict()["error_type"] == "timeout"


class TestRoutingDecision:
    """Tests for RoutingDecision enum."""

    def test_routing_values(self) -> None:
        """All routing options should exist."""
        assert RoutingDecision.ANALYST.value == "analyst"
        assert RoutingDecision.SEARCH.value == "search"
        assert RoutingDecision.BOTH.value == "both"

