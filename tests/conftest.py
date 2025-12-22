"""Pytest configuration and fixtures for Healthcare Multi-Agent Lab tests."""

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.config import Settings
from src.graphs.react_state import ConversationTurn, HealthcareAgentState
from src.models.requests import QueryRequest
from src.models.responses import AgentResponse

# --- Pytest Configuration ---


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# --- Settings Fixtures ---


@pytest.fixture
def mock_settings() -> MagicMock:
    """Create mock settings for testing without .env file."""
    settings = MagicMock(spec=Settings)
    settings.snowflake_account = "test_account"
    settings.snowflake_user = "test_user"
    settings.snowflake_private_key_path = "/tmp/test_key.pem"
    settings.snowflake_private_key_passphrase = "test_pass"
    settings.snowflake_database = "HEALTHCARE_DB"
    settings.snowflake_schema = "PUBLIC"
    settings.snowflake_warehouse = "PAYERS_CC_WH"
    settings.snowflake_role = "ACCOUNTADMIN"
    settings.max_agent_steps = 5
    settings.agent_timeout_seconds = 30.0
    settings.cortex_llm_model = "llama3.1-70b"
    settings.cortex_search_limit = 5
    return settings


# --- Modern Message-Based State Fixtures ---


@pytest.fixture
def initial_state() -> HealthcareAgentState:
    """Create minimal initial state for testing."""
    return HealthcareAgentState(
        messages=[HumanMessage(content="What are my benefits?")],
        user_query="What are my benefits?",
        member_id="106742775",
        tenant_id="test_tenant",
        conversation_history=[],
        iteration=0,
        max_iterations=5,
        execution_id="test-exec-001",
        final_answer=None,
        error_count=0,
        last_error=None,
        has_error=False,
    )


@pytest.fixture
def state_with_history() -> HealthcareAgentState:
    """State with conversation history for continuation testing."""
    return HealthcareAgentState(
        messages=[HumanMessage(content="What is their plan type?")],
        user_query="What is their plan type?",
        member_id=None,  # Should be inferred from history
        tenant_id="test_tenant",
        conversation_history=[
            ConversationTurn(
                user_query="Tell me about member 106742775",
                member_id="106742775",
                final_answer="Member 106742775 is enrolled in the Gold PPO plan.",
                tools_used=["query_member_data"],
            ),
        ],
        iteration=0,
        max_iterations=5,
        execution_id="test-exec-002",
        final_answer=None,
        error_count=0,
        last_error=None,
        has_error=False,
    )


@pytest.fixture
def state_after_tool() -> HealthcareAgentState:
    """State after tool execution with messages."""
    return HealthcareAgentState(
        messages=[
            HumanMessage(content="What claims have I submitted?"),
            AIMessage(
                content="I need to query the member database for claims.",
                tool_calls=[
                    {
                        "id": "call_test123",
                        "name": "query_member_data",
                        "args": {"query": "claims for member", "member_id": "106742775"},
                    }
                ],
            ),
            ToolMessage(
                content="[Database Query Results] Claims found: 3",
                tool_call_id="call_test123",
            ),
        ],
        user_query="What claims have I submitted?",
        member_id="106742775",
        tenant_id="test_tenant",
        conversation_history=[],
        iteration=1,
        max_iterations=5,
        execution_id="test-exec-003",
        final_answer=None,
        error_count=0,
        last_error=None,
        has_error=False,
    )


@pytest.fixture
def error_state() -> HealthcareAgentState:
    """State with error condition."""
    return HealthcareAgentState(
        messages=[HumanMessage(content="trigger error")],
        user_query="trigger error",
        member_id=None,
        tenant_id="test_tenant",
        conversation_history=[],
        iteration=3,
        max_iterations=5,
        execution_id="test-exec-error",
        final_answer=None,
        error_count=2,
        last_error={"error_type": "cortex_api", "message": "Service unavailable"},
        has_error=True,
    )


# --- Request/Response Fixtures ---


@pytest.fixture
def sample_query_request() -> QueryRequest:
    """Create sample query request."""
    return QueryRequest(
        query="What are my coverage benefits for physical therapy?",
        member_id="106742775",
        tenant_id="default",
        user_id="test_user",
    )


@pytest.fixture
def sample_agent_response() -> AgentResponse:
    """Create sample agent response."""
    return AgentResponse(
        output="Based on your coverage, physical therapy is covered at 80% after deductible.",
        routing="analyst",
        execution_id="test-exec-001",
        checkpoint_id="chk-001",
        error_count=0,
        last_error=None,
        analyst_results={"coverage": {"physical_therapy": "80%"}},
        search_results=None,
    )


# --- Mock Cortex Tools ---


@pytest.fixture
def mock_analyst_tool() -> AsyncMock:
    """Mock AsyncCortexAnalystTool."""
    mock = AsyncMock()
    mock.execute.return_value = {
        "member_id": "106742775",
        "claims": [
            {"claim_id": "CLM001", "amount": 150.00, "status": "approved"},
            {"claim_id": "CLM002", "amount": 75.50, "status": "pending"},
        ],
        "coverage": {
            "deductible": 500.00,
            "deductible_met": 250.00,
            "out_of_pocket_max": 3000.00,
        },
        "raw_response": "Member 106742775 has 2 claims...",
    }
    return mock


@pytest.fixture
def mock_search_tool() -> AsyncMock:
    """Mock AsyncCortexSearchTool."""
    mock = AsyncMock()
    mock.execute.return_value = [
        {
            "text": "A copay is a fixed amount you pay for a covered healthcare service.",
            "score": 0.95,
            "source": "faqs",
            "metadata": {},
        },
        {
            "text": "Your plan covers preventive care at 100%.",
            "score": 0.87,
            "source": "policies",
            "metadata": {},
        },
    ]
    return mock


@pytest.fixture
def mock_snowflake_session() -> MagicMock:
    """Mock Snowflake Session."""
    session = MagicMock()
    session.sql.return_value.collect.return_value = []
    return session


# --- Graph Fixtures ---


@pytest.fixture
def mock_compiled_graph() -> AsyncMock:
    """Mock compiled LangGraph for testing without full graph execution."""
    mock = AsyncMock()
    mock.ainvoke.return_value = {
        "messages": [
            HumanMessage(content="What are my benefits?"),
            AIMessage(
                content="I need to query member data.",
                tool_calls=[
                    {
                        "id": "call_test",
                        "name": "query_member_data",
                        "args": {"query": "benefits", "member_id": "106742775"},
                    }
                ],
            ),
            ToolMessage(content="[Database Query] Benefits info found.", tool_call_id="call_test"),
            AIMessage(content="Based on your records, your benefits include..."),
        ],
        "user_query": "What are my benefits?",
        "member_id": "106742775",
        "tenant_id": "test_tenant",
        "conversation_history": [],
        "iteration": 2,
        "max_iterations": 5,
        "final_answer": "Based on your records, your benefits include...",
        "execution_id": "test-exec",
        "error_count": 0,
        "last_error": None,
        "has_error": False,
    }

    async def mock_stream(*_args: object, **_kwargs: object) -> AsyncIterator[dict]:
        """Mock streaming."""
        yield {
            "model": {
                "messages": [
                    AIMessage(
                        content="Querying...",
                        tool_calls=[{"id": "call_1", "name": "query_member_data", "args": {}}],
                    )
                ],
                "iteration": 1,
            }
        }
        yield {
            "tools": {
                "messages": [ToolMessage(content="Member data found", tool_call_id="call_1")]
            }
        }
        yield {"final_answer": {"final_answer": "Response..."}}

    mock.astream = mock_stream
    return mock


# --- Memory/Checkpointer Fixtures ---


@pytest.fixture
def mock_checkpointer() -> MagicMock:
    """Mock LangGraph checkpointer."""
    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver()


# --- Cleanup Fixtures ---


@pytest.fixture(autouse=True)
def reset_caches():
    """Reset LRU caches between tests."""
    from src.dependencies import get_checkpointer, get_compiled_graph, get_settings

    # Clear before test
    get_settings.cache_clear()
    get_compiled_graph.cache_clear()
    get_checkpointer.cache_clear()

    yield

    # Clear after test
    get_settings.cache_clear()
    get_compiled_graph.cache_clear()
    get_checkpointer.cache_clear()
