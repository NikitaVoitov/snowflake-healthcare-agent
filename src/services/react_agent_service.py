"""ReAct Agent service for orchestrating LangGraph ReAct workflow execution.

Supports conversation continuity via Snowflake checkpointer in SPCS deployments.
Uses message-based state with ToolNode for automatic tool execution.
"""

import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.pregel import Pregel

from src.config import Settings
from src.graphs.react_state import HealthcareAgentState
from src.models.requests import QueryRequest
from src.models.responses import AgentResponse, StreamEvent

logger = logging.getLogger(__name__)


class AgentService:
    """Orchestrates LangGraph ReAct healthcare agent execution.

    Uses modern message-based state pattern with ToolNode.
    Provides execute() for synchronous completion and stream()
    for real-time event streaming to clients.
    """

    def __init__(self, graph: Pregel, settings: Settings) -> None:
        """Initialize agent service.

        Args:
            graph: Compiled LangGraph workflow (Pregel instance).
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

    def _build_initial_state(
        self,
        request: QueryRequest,
        execution_id: str,
    ) -> HealthcareAgentState:
        """Build initial state for a conversation turn.

        Args:
            request: Validated QueryRequest with query, member_id, etc.
            execution_id: Unique identifier for this execution.

        Returns:
            HealthcareAgentState for the new turn.
        """
        return HealthcareAgentState(
            # Messages - the core communication channel
            messages=[HumanMessage(content=request.query.strip())],
            # Context
            user_query=request.query.strip(),
            member_id=request.member_id,
            tenant_id=request.tenant_id,
            # Conversation history (loaded from checkpoint if available)
            conversation_history=[],
            # Execution control
            iteration=0,
            max_iterations=self.settings.max_agent_steps,
            execution_id=execution_id,
            # Output
            final_answer=None,
            # Error handling
            error_count=0,
            last_error=None,
            has_error=False,
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
        """Execute agent workflow and return final result.

        Supports conversation continuity via checkpointer.

        Args:
            request: Validated query request.
            checkpointer: Optional LangGraph checkpointer for state retrieval.

        Returns:
            AgentResponse with output and metadata.

        Raises:
            ValueError: If query is empty.
        """
        if not request.query.strip():
            raise ValueError("Query cannot be empty")

        # Use provided thread_id for conversation continuation, or create new one
        is_continuation = request.thread_id is not None
        thread_id = request.thread_id or self._build_thread_id(request.tenant_id, request.user_id)
        config = {"configurable": {"thread_id": thread_id}}

        # Build initial state for new turn
        initial_state = self._build_initial_state(request, thread_id)

        # Load conversation history from previous checkpoint if continuing
        if is_continuation:
            history = await self._load_conversation_history(checkpointer, config)
            if history:
                initial_state["conversation_history"] = history

        logger.info(f"Executing agent workflow: {thread_id} (continuation: {is_continuation})")

        # Execute graph
        result = await self.graph.ainvoke(initial_state, config=config)

        # Get the actual checkpoint_id from the checkpointer after execution
        checkpoint_id = None
        if checkpointer:
            try:
                checkpoint_tuple = await checkpointer.aget_tuple(config)
                if checkpoint_tuple and checkpoint_tuple.checkpoint:
                    checkpoint_id = checkpoint_tuple.checkpoint.get("id")
            except Exception as exc:
                logger.warning(f"Could not retrieve checkpoint_id: {exc}")

        # Extract results from message-based state
        messages = result.get("messages", [])
        final_answer = result.get("final_answer", "")

        # Analyze tools used from messages
        tools_used = self._extract_tools_from_messages(messages)
        routing = self._determine_routing(tools_used)

        return AgentResponse(
            output=final_answer,
            routing=routing,
            execution_id=thread_id,
            checkpoint_id=checkpoint_id,
            error_count=result.get("error_count", 0),
            last_error=result.get("last_error"),
            analyst_results=self._extract_analyst_results(messages),
            search_results=self._extract_search_results(messages),
        )

    def _extract_tools_from_messages(self, messages: list) -> list[str]:
        """Extract tool names from message sequence.

        Args:
            messages: List of messages.

        Returns:
            List of tool names used.
        """
        tools_used = []
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tools_used.append(tool_call.get("name", ""))
        return tools_used

    def _determine_routing(self, tools_used: list[str]) -> str:
        """Determine routing label based on tools used.

        Args:
            tools_used: List of tool names.

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
            return "react"

    def _extract_analyst_results(self, messages: list) -> dict | None:
        """Extract analyst results from messages.

        Searches for query_member_data tool calls and finds their corresponding
        ToolMessage results by matching tool_call_id (not position).

        Args:
            messages: List of messages.

        Returns:
            Analyst results dict or None.
        """
        # Debug: log message types and structure
        logger.debug(
            "Extracting analyst results from %d messages: %s",
            len(messages),
            [(type(m).__name__, getattr(m, "tool_call_id", None) if hasattr(m, "tool_call_id") else "N/A") for m in messages],
        )

        # Build a map of tool_call_id -> ToolMessage content
        tool_results: dict[str, str] = {}
        for msg in messages:
            if isinstance(msg, ToolMessage) and hasattr(msg, "tool_call_id"):
                tool_results[msg.tool_call_id] = msg.content
                logger.debug("Found ToolMessage with tool_call_id=%s", msg.tool_call_id)

        logger.debug("Tool results map: %s", list(tool_results.keys()))

        # Find query_member_data tool calls and get their results by tool_call_id
        for msg in messages:
            if not (isinstance(msg, AIMessage) and msg.tool_calls):
                continue
            for tool_call in msg.tool_calls:
                tool_name = tool_call.get("name", "")
                tool_id = tool_call.get("id", "")
                logger.debug("Checking tool_call: name=%s, id=%s", tool_name, tool_id)
                if tool_name == "query_member_data" and tool_id in tool_results:
                    logger.info("Found analyst results for tool_call_id=%s", tool_id)
                    return {"raw_response": tool_results[tool_id]}

        logger.warning("No analyst results found despite analyst routing")
        return None

    def _extract_search_results(self, messages: list) -> list | None:
        """Extract search results from messages.

        Searches for search_knowledge tool calls and finds their corresponding
        ToolMessage results by matching tool_call_id (not position).

        Args:
            messages: List of messages.

        Returns:
            Search results list or None.
        """
        # Build a map of tool_call_id -> ToolMessage content
        tool_results: dict[str, str] = {}
        for msg in messages:
            if isinstance(msg, ToolMessage) and hasattr(msg, "tool_call_id"):
                tool_results[msg.tool_call_id] = msg.content

        # Find search_knowledge tool calls and get their results by tool_call_id
        for msg in messages:
            if not (isinstance(msg, AIMessage) and msg.tool_calls):
                continue
            for tool_call in msg.tool_calls:
                tool_name = tool_call.get("name", "")
                tool_id = tool_call.get("id", "")
                if tool_name == "search_knowledge" and tool_id in tool_results:
                    return [{"text": tool_results[tool_id], "source": "knowledge_base"}]

        return None

    async def stream(
        self,
        request: QueryRequest,
        checkpointer: BaseCheckpointSaver | None = None,  # noqa: ARG002
    ) -> AsyncIterator[StreamEvent]:
        """Stream agent execution events for real-time UI updates.

        Args:
            request: Validated query request.
            checkpointer: Optional LangGraph checkpointer.

        Yields:
            StreamEvent for each node transition and completion.
        """
        if not request.query.strip():
            yield StreamEvent(
                event_type="error",
                data={"message": "Query cannot be empty"},
            )
            return

        is_continuation = request.thread_id is not None
        thread_id = request.thread_id or self._build_thread_id(request.tenant_id, request.user_id)
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = self._build_initial_state(request, thread_id)

        # Load conversation history if continuing
        if is_continuation and hasattr(self.graph, "checkpointer"):
            try:
                history = await self._load_conversation_history(self.graph.checkpointer, config)
                if history:
                    initial_state["conversation_history"] = history
            except Exception as exc:
                logger.warning(f"Could not load history for streaming: {exc}")

        logger.info(f"Streaming agent workflow: {thread_id} (continuation: {is_continuation})")

        # Emit start event
        yield StreamEvent(
            event_type="node_start",
            node="react_workflow",
            data={"thread_id": thread_id, "query": request.query, "is_continuation": is_continuation},
        )

        try:
            async for event in self.graph.astream(initial_state, config=config):
                for node_name, node_output in event.items():
                    if node_name == "model":
                        # Model produced output
                        messages = node_output.get("messages", [])
                        if messages and isinstance(messages[-1], AIMessage):
                            last_msg = messages[-1]
                            if last_msg.tool_calls:
                                yield StreamEvent(
                                    event_type="react_thought",
                                    node=node_name,
                                    data={
                                        "thought": last_msg.content[:200],
                                        "action": last_msg.tool_calls[0].get("name", ""),
                                        "iteration": node_output.get("iteration", 0),
                                    },
                                )
                            else:
                                yield StreamEvent(
                                    event_type="react_answer",
                                    node=node_name,
                                    data={"answer_preview": last_msg.content[:200]},
                                )
                    elif node_name == "tools":
                        # Tools executed
                        messages = node_output.get("messages", [])
                        tool_results = [m.content[:100] for m in messages if isinstance(m, ToolMessage)]
                        if tool_results:
                            yield StreamEvent(
                                event_type="react_action",
                                node=node_name,
                                data={"result_preview": str(tool_results)[:200]},
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
                            data={"node": node_name},
                        )

            yield StreamEvent(
                event_type="complete",
                node="react_workflow",
                data={"thread_id": thread_id},
            )

        except Exception as exc:
            logger.error(f"Stream error: {exc}", exc_info=True)
            yield StreamEvent(
                event_type="error",
                node="react_workflow",
                data={"message": str(exc)},
            )
