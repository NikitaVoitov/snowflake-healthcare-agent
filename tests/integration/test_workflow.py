"""Integration tests for LangGraph workflow execution."""

import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.graphs.state import HealthcareAgentState
from src.graphs.workflow import create_healthcare_graph


class TestGraphCompilation:
    """Tests for graph compilation."""

    def test_graph_compiles_successfully(self) -> None:
        """Graph should compile without errors."""
        graph = create_healthcare_graph()
        compiled = graph.compile()
        assert compiled is not None

    def test_graph_compiles_with_checkpointer(self) -> None:
        """Graph should compile with memory checkpointer."""
        graph = create_healthcare_graph()
        compiled = graph.compile(checkpointer=MemorySaver())
        assert compiled is not None

    def test_graph_has_expected_nodes(self) -> None:
        """Compiled graph should have all expected nodes."""
        graph = create_healthcare_graph()
        compiled = graph.compile()
        node_names = list(compiled.nodes.keys())
        expected = [
            "planner",
            "analyst",
            "search",
            "parallel",
            "response",
            "error_handler",
            "escalation_handler",
        ]
        for node in expected:
            assert node in node_names, f"Missing node: {node}"


class TestGraphExecution:
    """Tests for full graph execution with Snowflake Cortex."""

    @pytest.mark.asyncio
    async def test_analyst_path_execution(self, analyst_query_state: HealthcareAgentState) -> None:
        """Graph should execute analyst path for member queries."""
        graph = create_healthcare_graph()
        compiled = graph.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "test-analyst-001"}}
        result = await compiled.ainvoke(analyst_query_state, config=config)
        assert result["is_complete"] is True
        assert result["final_response"] != ""

    @pytest.mark.asyncio
    async def test_search_path_execution(self, search_query_state: HealthcareAgentState) -> None:
        """Graph should execute search path for general queries."""
        graph = create_healthcare_graph()
        compiled = graph.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "test-search-001"}}
        result = await compiled.ainvoke(search_query_state, config=config)
        assert result["is_complete"] is True
        assert result["final_response"] != ""

    @pytest.mark.asyncio
    async def test_parallel_path_execution(self, initial_state: HealthcareAgentState) -> None:
        """Graph should execute parallel path for 'both' routing."""
        state = {
            **initial_state,
            "plan": "both",
            "user_query": "Show my claims and explain deductible policy",
        }
        graph = create_healthcare_graph()
        compiled = graph.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "test-parallel-001"}}
        result = await compiled.ainvoke(state, config=config)
        assert result["is_complete"] is True

    @pytest.mark.asyncio
    async def test_error_handling_path(self, error_state: HealthcareAgentState) -> None:
        """Graph should handle errors gracefully."""
        graph = create_healthcare_graph()
        compiled = graph.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "test-error-001"}}
        result = await compiled.ainvoke(error_state, config=config)
        assert "is_complete" in result


class TestGraphStreaming:
    """Tests for graph streaming execution."""

    @pytest.mark.asyncio
    async def test_stream_emits_events(self, initial_state: HealthcareAgentState) -> None:
        """Stream should emit events for each node."""
        graph = create_healthcare_graph()
        compiled = graph.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "test-stream-001"}}
        events = []
        async for event in compiled.astream(initial_state, config=config):
            events.append(event)
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_stream_includes_node_names(self, initial_state: HealthcareAgentState) -> None:
        """Stream events should include node names as keys."""
        graph = create_healthcare_graph()
        compiled = graph.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "test-stream-002"}}
        node_names = set()
        async for event in compiled.astream(initial_state, config=config):
            node_names.update(event.keys())

        # Verify planner node executes (required entry point)
        assert "planner" in node_names, f"Expected 'planner' node in stream, got: {node_names}"
        # Verify at least one other node executes after planner
        assert len(node_names) > 1, f"Expected multiple nodes, got only: {node_names}"


class TestCheckpointing:
    """Tests for state persistence with checkpointer."""

    @pytest.mark.asyncio
    async def test_state_persists_across_invocations(
        self, initial_state: HealthcareAgentState
    ) -> None:
        """State should persist with same thread_id."""
        graph = create_healthcare_graph()
        checkpointer = MemorySaver()
        compiled = graph.compile(checkpointer=checkpointer)
        thread_id = "test-persistence-001"
        config = {"configurable": {"thread_id": thread_id}}
        await compiled.ainvoke(initial_state, config=config)
        state = await compiled.aget_state(config)
        assert state is not None

    @pytest.mark.asyncio
    async def test_different_threads_isolated(self, initial_state: HealthcareAgentState) -> None:
        """Different thread_ids should have isolated state."""
        graph = create_healthcare_graph()
        checkpointer = MemorySaver()
        compiled = graph.compile(checkpointer=checkpointer)
        config1 = {"configurable": {"thread_id": "thread-001"}}
        config2 = {"configurable": {"thread_id": "thread-002"}}
        state1 = {**initial_state, "user_query": "Query for thread 1"}
        state2 = {**initial_state, "user_query": "Query for thread 2"}
        result1 = await compiled.ainvoke(state1, config=config1)
        result2 = await compiled.ainvoke(state2, config=config2)
        assert result1["user_query"] != result2["user_query"]
