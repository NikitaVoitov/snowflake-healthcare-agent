"""Pytest configuration and fixtures for Healthcare Multi-Agent Lab tests."""

import asyncio
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import Settings
from src.graphs.state import HealthcareAgentState
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
    settings.max_agent_steps = 3
    settings.agent_timeout_seconds = 30.0
    settings.cortex_analyst_semantic_model = "@STAGING.MEMBER_SEMANTIC_MODEL"
    settings.cortex_search_services = ["FAQS_SEARCH", "POLICIES_SEARCH", "TRANSCRIPTS_SEARCH"]
    return settings


# --- State Fixtures ---


@pytest.fixture
def initial_state() -> HealthcareAgentState:
    """Create minimal initial state for testing."""
    return HealthcareAgentState(
        user_query="What are my benefits?",
        member_id="ABC123",
        messages=[],
        plan="",
        analyst_results=None,
        search_results=None,
        final_response="",
        execution_id="test-exec-001",
        thread_checkpoint_id=None,
        current_step="start",
        error_count=0,
        max_steps=5,
        last_error=None,
        has_error=False,
        is_complete=False,
        should_escalate=False,
    )


@pytest.fixture
def analyst_query_state() -> HealthcareAgentState:
    """State for analyst-routed query."""
    return HealthcareAgentState(
        user_query="What claims have I submitted this year?",
        member_id="MEM456",
        messages=[],
        plan="analyst",
        analyst_results=None,
        search_results=None,
        final_response="",
        execution_id="test-exec-002",
        thread_checkpoint_id=None,
        current_step="analyst",
        error_count=0,
        max_steps=5,
        last_error=None,
        has_error=False,
        is_complete=False,
        should_escalate=False,
    )


@pytest.fixture
def search_query_state() -> HealthcareAgentState:
    """State for search-routed query."""
    return HealthcareAgentState(
        user_query="What is the copay for specialist visits?",
        member_id=None,
        messages=[],
        plan="search",
        analyst_results=None,
        search_results=None,
        final_response="",
        execution_id="test-exec-003",
        thread_checkpoint_id=None,
        current_step="search",
        error_count=0,
        max_steps=5,
        last_error=None,
        has_error=False,
        is_complete=False,
        should_escalate=False,
    )


@pytest.fixture
def error_state() -> HealthcareAgentState:
    """State with error condition."""
    return HealthcareAgentState(
        user_query="trigger error",
        member_id=None,
        messages=[],
        plan="both",
        analyst_results=None,
        search_results=None,
        final_response="",
        execution_id="test-exec-error",
        thread_checkpoint_id=None,
        current_step="error_handler",
        error_count=2,
        max_steps=3,
        last_error={"error_type": "cortex_api", "message": "Service unavailable"},
        has_error=True,
        is_complete=False,
        should_escalate=False,
    )


# --- Request/Response Fixtures ---


@pytest.fixture
def sample_query_request() -> QueryRequest:
    """Create sample query request."""
    return QueryRequest(
        query="What are my coverage benefits for physical therapy?",
        member_id="ABC123",
        tenant_id="default",
        user_id="test_user",
    )


@pytest.fixture
def sample_agent_response() -> AgentResponse:
    """Create sample agent response."""
    return AgentResponse(
        output="Based on your coverage, physical therapy is covered at 80% after deductible.",
        routing="both",
        execution_id="test-exec-001",
        checkpoint_id="chk-001",
        error_count=0,
        last_error=None,
        analyst_results={"claims": [], "coverage": {"physical_therapy": "80%"}},
        search_results=[{"source": "policy", "content": "PT coverage details"}],
    )


# --- Mock Cortex Tools ---


@pytest.fixture
def mock_analyst_tool() -> AsyncMock:
    """Mock AsyncCortexAnalystTool."""
    mock = AsyncMock()
    mock.execute.return_value = {
        "member_id": "ABC123",
        "claims": [
            {"claim_id": "CLM001", "amount": 150.00, "status": "approved"},
            {"claim_id": "CLM002", "amount": 75.50, "status": "pending"},
        ],
        "coverage": {
            "deductible": 500.00,
            "deductible_met": 250.00,
            "out_of_pocket_max": 3000.00,
        },
        "raw_response": "Member ABC123 has 2 claims...",
    }
    return mock


@pytest.fixture
def mock_search_tool() -> AsyncMock:
    """Mock AsyncCortexSearchTool."""
    mock = AsyncMock()
    mock.execute.return_value = [
        {
            "service": "FAQS_SEARCH",
            "results": [
                {"question": "What is copay?", "answer": "A fixed amount you pay..."},
            ],
        },
        {
            "service": "POLICIES_SEARCH",
            "results": [
                {"policy_name": "Coverage Guide", "content": "Your plan covers..."},
            ],
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
        "user_query": "What are my benefits?",
        "member_id": "ABC123",
        "messages": [],
        "plan": "both",
        "analyst_results": {"coverage": {}},
        "search_results": [{"content": "FAQ answer"}],
        "final_response": "Based on your records...",
        "execution_id": "test-exec",
        "thread_checkpoint_id": "chk-001",
        "current_step": "complete",
        "error_count": 0,
        "max_steps": 5,
        "last_error": None,
        "has_error": False,
        "is_complete": True,
        "should_escalate": False,
    }
    
    async def mock_stream(*args, **kwargs) -> AsyncIterator[dict]:
        """Mock streaming."""
        yield {"planner": {"plan": "both"}}
        yield {"parallel_execution": {"analyst_results": {}, "search_results": []}}
        yield {"response_synthesizer": {"final_response": "Response..."}}
    
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
    from src.dependencies import get_compiled_graph, get_settings
    
    # Clear before test
    get_settings.cache_clear()
    get_compiled_graph.cache_clear()
    
    yield
    
    # Clear after test
    get_settings.cache_clear()
    get_compiled_graph.cache_clear()

