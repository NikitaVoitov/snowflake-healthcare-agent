"""ReAct Agent service for orchestrating LangGraph ReAct workflow execution.

Supports conversation continuity via Snowflake checkpointer in SPCS deployments.
The conversation_history is persisted across turns, enabling context-aware
follow-up questions like "What plan are they on?" after discussing a member.
"""

import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.pregel import Pregel

from src.config import Settings
from src.graphs.react_state import HealthcareReActState
from src.graphs.react_workflow import build_initial_react_state
from src.models.requests import QueryRequest
from src.models.responses import AgentResponse, StreamEvent

logger = logging.getLogger(__name__)


class ReActAgentService:
    """Orchestrates LangGraph ReAct healthcare agent execution.

    Provides execute() for synchronous completion and stream()
    for real-time event streaming to clients.

    This service uses the ReAct (Reasoning + Acting) workflow pattern
    for intelligent, iterative tool selection.
    """

    def __init__(self, graph: Pregel, settings: Settings) -> None:
        """Initialize ReAct agent service.

        Args:
            graph: Compiled ReAct LangGraph workflow (Pregel instance).
            settings: Application settings.
        """
        self.graph = graph
        self.settings = settings

    def _build_thread_id(self, tenant_id: str, user_id: str) -> str:
        """Build scoped thread ID for multi-tenancy and checkpointing.

        Args:
            tenant_id: Tenant identifier for isolation.
            user_id: User identifier for tracking.

        Returns:
            Formatted thread ID.
        """
        ts = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
        return f"tenant:{tenant_id}:user:{user_id}:ts:{ts}"

    def _build_turn_state(self, request: QueryRequest, execution_id: str) -> HealthcareReActState:
        """Build state for a new conversation turn.

        Creates state for the CURRENT turn only. Per-turn fields (scratchpad,
        iteration, etc.) are reset. Conversation history is loaded separately
        from the checkpointer in execute() and merged into this state.

        Args:
            request: Validated QueryRequest with query, member_id, etc.
            execution_id: Unique identifier for this execution.

        Returns:
            HealthcareReActState for the new turn (without conversation_history).
        """
        # Note: conversation_history is NOT set here - it's explicitly loaded
        # from the checkpointer in execute() for conversation continuation.
        return HealthcareReActState(
            # Current turn input
            user_query=request.query.strip(),
            member_id=request.member_id,
            tenant_id=request.tenant_id,
            # Reset per-turn state
            scratchpad=[],
            current_step=None,
            iteration=0,
            max_iterations=self.settings.max_agent_steps,
            tool_result=None,
            final_answer=None,
            # Execution tracking
            execution_id=execution_id,
            thread_checkpoint_id=None,
            current_node="start",
            # Reset error state per turn
            error_count=0,
            last_error=None,
            has_error=False,
            # conversation_history is NOT set - preserved via checkpointer
        )

    async def _load_conversation_history(
        self,
        checkpointer: BaseCheckpointSaver | None,
        config: dict,
    ) -> list:
        """Load conversation history from checkpoint if available.

        Args:
            checkpointer: LangGraph checkpointer for state retrieval.
            config: Config with thread_id for checkpoint lookup.

        Returns:
            List of ConversationTurn dicts or empty list.
        """
        if not checkpointer:
            return []

        try:
            checkpoint_tuple = await checkpointer.aget_tuple(config)
            if checkpoint_tuple and checkpoint_tuple.checkpoint:
                channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
                history = channel_values.get("conversation_history", [])
                if history:
                    logger.info(f"Loaded {len(history)} previous turns from checkpoint")
                return history
        except Exception as exc:
            logger.warning(f"Could not load conversation history: {exc}")

        return []

    async def execute(
        self,
        request: QueryRequest,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> AgentResponse:
        """Execute ReAct agent workflow and return final result.

        Supports conversation continuity via checkpointer:
        - If thread_id is provided, loads existing conversation_history from checkpoint
        - Conversation context is included in prompts for follow-up questions
        - After execution, the completed turn is appended to history and saved

        Args:
            request: Validated query request.
            checkpointer: Optional LangGraph checkpointer for state retrieval.

        Returns:
            AgentResponse with output and metadata.

        Raises:
            ValueError: If query is empty.
            TimeoutError: If execution exceeds timeout.
        """
        # Guard clauses first
        if not request.query.strip():
            raise ValueError("Query cannot be empty")

        # Use provided thread_id for conversation continuation, or create new one
        is_continuation = request.thread_id is not None
        thread_id = request.thread_id or self._build_thread_id(request.tenant_id, request.user_id)
        config = {"configurable": {"thread_id": thread_id}}

        # Build state for new turn
        turn_state = self._build_turn_state(request, thread_id)

        # Load conversation history from previous checkpoint if continuing
        if is_continuation:
            history = await self._load_conversation_history(checkpointer, config)
            if history:
                turn_state["conversation_history"] = history

        logger.info(f"Executing ReAct agent workflow: {thread_id} (continuation: {is_continuation})")

        # Execute graph - checkpointer automatically:
        # 1. Loads existing state (including conversation_history) for this thread
        # 2. Merges turn_state with existing state using reducers
        # 3. Saves final state (with updated conversation_history) after execution
        result = await self.graph.ainvoke(turn_state, config=config)

        # Get the actual checkpoint_id from the checkpointer after execution
        checkpoint_id = None
        if checkpointer:
            try:
                checkpoint_tuple = await checkpointer.aget_tuple(config)
                if checkpoint_tuple and checkpoint_tuple.checkpoint:
                    checkpoint_id = checkpoint_tuple.checkpoint.get("id")
            except Exception as exc:
                logger.warning(f"Could not retrieve checkpoint_id: {exc}")

        # Extract final answer from ReAct state
        final_answer = result.get("final_answer", "")

        # Build scratchpad summary for debugging
        scratchpad = result.get("scratchpad", [])
        iterations = len(scratchpad)

        # Determine routing based on tools used
        tools_used = [step.get("action") for step in scratchpad if step.get("action")]
        routing = self._determine_routing(tools_used)

        return AgentResponse(
            output=final_answer,
            routing=routing,
            execution_id=thread_id,
            checkpoint_id=checkpoint_id,
            error_count=result.get("error_count", 0),
            last_error=result.get("last_error"),
            analyst_results=self._extract_analyst_results(scratchpad),
            search_results=self._extract_search_results(scratchpad),
        )

    def _determine_routing(self, tools_used: list[str]) -> str:
        """Determine routing label based on tools used.

        Args:
            tools_used: List of action names from scratchpad.

        Returns:
            Routing label (analyst, search, both, or react).
        """
        has_analyst = "query_member_data" in tools_used
        has_search = "search_knowledge" in tools_used

        if has_analyst and has_search:
            return "both"
        elif has_analyst:
            return "analyst"
        elif has_search:
            return "search"
        else:
            return "react"  # Indicates ReAct but no tools called

    def _extract_analyst_results(self, scratchpad: list) -> dict | None:
        """Extract analyst results from scratchpad for response.

        Args:
            scratchpad: List of ReActStep dictionaries.

        Returns:
            Analyst results dict or None.
        """
        for step in scratchpad:
            if step.get("action") == "query_member_data" and step.get("observation"):
                # Return the observation as raw result
                return {"raw_response": step.get("observation")}
        return None

    def _extract_search_results(self, scratchpad: list) -> list | None:
        """Extract search results from scratchpad for response.

        Args:
            scratchpad: List of ReActStep dictionaries.

        Returns:
            Search results list or None.
        """
        for step in scratchpad:
            if step.get("action") == "search_knowledge" and step.get("observation"):
                # Return the observation as raw result
                return [{"text": step.get("observation"), "source": "knowledge_base"}]
        return None

    async def stream(
        self,
        request: QueryRequest,
        checkpointer: BaseCheckpointSaver | None = None,  # noqa: ARG002
    ) -> AsyncIterator[StreamEvent]:
        """Stream ReAct agent execution events for real-time UI updates.

        Supports conversation continuity via checkpointer.

        Args:
            request: Validated query request.
            checkpointer: Optional LangGraph checkpointer.

        Yields:
            StreamEvent for each node transition and completion.
        """
        # Guard clause
        if not request.query.strip():
            yield StreamEvent(
                event_type="error",
                data={"message": "Query cannot be empty"},
            )
            return

        # Use provided thread_id for conversation continuation
        is_continuation = request.thread_id is not None
        thread_id = request.thread_id or self._build_thread_id(request.tenant_id, request.user_id)
        config = {"configurable": {"thread_id": thread_id}}
        turn_state = self._build_turn_state(request, thread_id)

        # Load conversation history from previous checkpoint if continuing
        # Note: For streaming, checkpointer access is limited; history loading is best-effort
        if is_continuation and hasattr(self.graph, "checkpointer"):
            try:
                history = await self._load_conversation_history(self.graph.checkpointer, config)
                if history:
                    turn_state["conversation_history"] = history
            except Exception as exc:
                logger.warning(f"Could not load history for streaming: {exc}")

        logger.info(f"Streaming ReAct agent workflow: {thread_id} (continuation: {is_continuation})")

        # Emit start event
        yield StreamEvent(
            event_type="node_start",
            node="react_workflow",
            data={"thread_id": thread_id, "query": request.query, "is_continuation": is_continuation},
        )

        try:
            async for event in self.graph.astream(turn_state, config=config):
                for node_name, node_output in event.items():
                    # Emit ReAct-specific events
                    if node_name == "reasoner":
                        current_step = node_output.get("current_step", {})
                        yield StreamEvent(
                            event_type="react_thought",
                            node=node_name,
                            data={
                                "thought": current_step.get("thought", ""),
                                "action": current_step.get("action", ""),
                                "iteration": node_output.get("iteration", 0),
                            },
                        )
                    elif node_name in ("analyst_tool", "search_tool"):
                        yield StreamEvent(
                            event_type="react_action",
                            node=node_name,
                            data={"result_preview": str(node_output.get("tool_result", ""))[:200]},
                        )
                    elif node_name == "observation":
                        scratchpad = node_output.get("scratchpad", [])
                        yield StreamEvent(
                            event_type="react_observation",
                            node=node_name,
                            data={"steps_completed": len(scratchpad)},
                        )
                    elif node_name == "final_answer":
                        yield StreamEvent(
                            event_type="react_answer",
                            node=node_name,
                            data={"answer": node_output.get("final_answer", "")},
                        )
                    else:
                        yield StreamEvent(
                            event_type="node_end",
                            node=node_name,
                            data=node_output,
                        )

            # Emit completion event
            yield StreamEvent(
                event_type="complete",
                node="react_workflow",
                data={"thread_id": thread_id},
            )

        except Exception as exc:
            logger.error(f"ReAct stream error: {exc}", exc_info=True)
            yield StreamEvent(
                event_type="error",
                node="react_workflow",
                data={"message": str(exc)},
            )

