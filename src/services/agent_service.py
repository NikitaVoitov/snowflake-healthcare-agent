"""Agent service for orchestrating LangGraph healthcare agent execution."""

import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from langgraph.pregel import Pregel

from src.config import Settings
from src.graphs.state import HealthcareAgentState
from src.models.requests import QueryRequest
from src.models.responses import AgentResponse, StreamEvent

logger = logging.getLogger(__name__)


class AgentService:
    """Orchestrates LangGraph healthcare agent execution.

    Provides execute() for synchronous completion and stream()
    for real-time event streaming to clients.
    """

    def __init__(self, graph: Pregel, settings: Settings) -> None:
        """Initialize agent service.

        Args:
            graph: Compiled LangGraph workflow (Pregel instance)
            settings: Application settings
        """
        self.graph = graph
        self.settings = settings

    def _build_thread_id(self, tenant_id: str, user_id: str) -> str:
        """Build scoped thread ID for multi-tenancy and checkpointing.

        Format: tenant:{tenant_id}:user:{user_id}:ts:{timestamp}
        """
        ts = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
        return f"tenant:{tenant_id}:user:{user_id}:ts:{ts}"

    def _build_initial_state(self, request: QueryRequest) -> HealthcareAgentState:
        """Build initial state from request.

        Args:
            request: Validated query request

        Returns:
            Initial state dictionary for graph execution
        """
        return HealthcareAgentState(
            user_query=request.query.strip(),
            member_id=request.member_id,
            messages=[],
            max_steps=self.settings.max_agent_steps,
            error_count=0,
            is_complete=False,
            has_error=False,
            should_escalate=False,
        )

    async def execute(
        self,
        request: QueryRequest,
        checkpointer: Any | None = None,  # noqa: ARG002
    ) -> AgentResponse:
        """Execute agent workflow and return final result.

        Args:
            request: Validated query request
            checkpointer: Optional LangGraph checkpointer (handled at dependency level)

        Returns:
            AgentResponse with output and metadata

        Raises:
            ValueError: If query is empty
            TimeoutError: If execution exceeds timeout
        """
        # Guard clauses first (per coding standards)
        if not request.query.strip():
            raise ValueError("Query cannot be empty")

        thread_id = self._build_thread_id(request.tenant_id, request.user_id)
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = self._build_initial_state(request)

        logger.info(f"Executing agent workflow: {thread_id}")

        # Execute graph (checkpointer is handled at dependency level)
        result = await self.graph.ainvoke(initial_state, config=config)

        return AgentResponse(
            output=result.get("final_response", ""),
            routing=result.get("plan", "unknown"),
            execution_id=thread_id,
            checkpoint_id=result.get("thread_checkpoint_id"),
            error_count=result.get("error_count", 0),
            last_error=result.get("last_error"),
            analyst_results=result.get("analyst_results"),
            search_results=result.get("search_results"),
        )

    async def stream(
        self,
        request: QueryRequest,
        checkpointer: Any | None = None,  # noqa: ARG002
    ) -> AsyncIterator[StreamEvent]:
        """Stream agent execution events for real-time UI updates.

        Args:
            request: Validated query request
            checkpointer: Optional LangGraph checkpointer (handled at dependency level)

        Yields:
            StreamEvent for each node transition and completion

        Raises:
            ValueError: If query is empty
        """
        # Guard clause
        if not request.query.strip():
            yield StreamEvent(
                event_type="error",
                data={"message": "Query cannot be empty"},
            )
            return

        thread_id = self._build_thread_id(request.tenant_id, request.user_id)
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = self._build_initial_state(request)

        logger.info(f"Streaming agent workflow: {thread_id}")

        # Emit start event
        yield StreamEvent(
            event_type="node_start",
            node="workflow",
            data={"thread_id": thread_id, "query": request.query},
        )

        try:
            async for event in self.graph.astream(initial_state, config=config):
                for node_name, node_output in event.items():
                    yield StreamEvent(
                        event_type="node_end",
                        node=node_name,
                        data=node_output,
                    )

            # Emit completion event
            yield StreamEvent(
                event_type="complete",
                node="workflow",
                data={"thread_id": thread_id},
            )

        except Exception as exc:
            logger.error(f"Stream error: {exc}", exc_info=True)
            yield StreamEvent(
                event_type="error",
                node="workflow",
                data={"message": str(exc)},
            )

