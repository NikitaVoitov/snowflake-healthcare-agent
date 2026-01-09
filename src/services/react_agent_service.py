"""Healthcare Agent service for orchestrating LangGraph agent execution.

Supports conversation continuity via Snowflake checkpointer in SPCS deployments.
Works with LangChain v1's create_agent which provides:
- LLM-controlled decision loop (true agent, not workflow)
- Automatic tool execution via built-in ToolNode
- Message-based state (HumanMessage, AIMessage, ToolMessage)
"""

import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from src.config import Settings
from src.graphs.react_state import HealthcareAgentState
from src.graphs.react_workflow import extract_final_answer
from src.models.requests import QueryRequest
from src.models.responses import AgentResponse, StreamEvent

logger = logging.getLogger(__name__)


class AgentService:
    """Orchestrates healthcare agent execution using LangChain v1 create_agent.

    Works with the compiled agent graph from create_agent() which provides
    an LLM-controlled decision loop with automatic tool execution.

    Provides execute() for synchronous completion and stream()
    for real-time event streaming to clients.
    """

    def __init__(self, graph: CompiledStateGraph, settings: Settings) -> None:
        """Initialize agent service.

        Args:
            graph: Compiled agent graph from create_agent().
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
        # Configure with thread_id and workflow metadata for observability
        # The metadata enables Agent Flow visualization in Splunk O11y trace views
        config = {
            "configurable": {"thread_id": thread_id},
            "metadata": {
                "workflow_type": "graph",  # Enables Agent Flow visualization
                "framework": "langgraph",
                "description": "Healthcare ReAct workflow with Cortex Analyst and Search tools",
            },
        }

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
        # Use extract_final_answer to find answer from messages (create_agent doesn't set final_answer field)
        final_answer = extract_final_answer(result)

        # Analyze tools used from messages
        tools_used = self._extract_tools_from_messages(messages)
        routing = self._determine_routing(tools_used)

        return AgentResponse(
            output=final_answer,
            routing=routing,
            execution_id=thread_id,
            checkpoint_id=checkpoint_id,
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

        # Collect all messages for final extraction
        all_messages: list = []
        final_answer = ""

        try:
            async for event in self.graph.astream(initial_state, config=config):
                for node_name, node_output in event.items():
                    # Accumulate messages from each node
                    node_messages = node_output.get("messages", [])
                    all_messages.extend(node_messages)

                    if node_name == "model":
                        # Model produced output
                        if node_messages and isinstance(node_messages[-1], AIMessage):
                            last_msg = node_messages[-1]
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
                        tool_results = [m.content[:100] for m in node_messages if isinstance(m, ToolMessage)]
                        if tool_results:
                            yield StreamEvent(
                                event_type="react_action",
                                node=node_name,
                                data={"result_preview": str(tool_results)[:200]},
                            )
                    elif node_name == "final_answer":
                        final_answer = node_output.get("final_answer", "")
                        yield StreamEvent(
                            event_type="react_answer",
                            node=node_name,
                            data={"answer": final_answer},
                        )
                    else:
                        yield StreamEvent(
                            event_type="node_end",
                            node=node_name,
                            data={"node": node_name},
                        )

            # Extract tools used and results from accumulated messages
            tools_used = self._extract_tools_from_messages(all_messages)
            routing = self._determine_routing(tools_used)
            analyst_results = self._extract_analyst_results(all_messages)
            search_results = self._extract_search_results(all_messages)

            yield StreamEvent(
                event_type="complete",
                node="react_workflow",
                data={
                    "thread_id": thread_id,
                    "routing": routing,
                    "analyst_results": analyst_results,
                    "search_results": search_results,
                },
            )

        except Exception as exc:
            logger.error(f"Stream error: {exc}", exc_info=True)
            yield StreamEvent(
                event_type="error",
                node="react_workflow",
                data={"message": str(exc)},
            )

    async def stream_tokens(
        self,
        request: QueryRequest,
        checkpointer: BaseCheckpointSaver | None = None,  # noqa: ARG002
    ) -> AsyncIterator[StreamEvent]:
        """Stream agent execution with token-level granularity.

        Uses LangGraph's astream_events API to capture:
        - Token-by-token LLM streaming (when patches applied)
        - Tool call progress (name, partial args)
        - Tool results
        - Final answer

        This requires the langchain-snowflake streaming patches to work properly.

        Args:
            request: Validated query request.
            checkpointer: Optional LangGraph checkpointer.

        Yields:
            StreamEvent for each token, tool call, and completion.
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
                logger.warning(f"Could not load history for token streaming: {exc}")

        logger.info(f"Token streaming agent workflow: {thread_id}")

        # Emit start event
        yield StreamEvent(
            event_type="node_start",
            node="react_workflow",
            data={"thread_id": thread_id, "query": request.query, "is_continuation": is_continuation},
        )

        all_messages: list = []
        accumulated_content = ""
        current_tool_name = ""
        current_tool_args = ""

        try:
            # Use astream_events for fine-grained streaming
            async for event in self.graph.astream_events(initial_state, config=config, version="v2"):
                event_kind = event.get("event", "")
                event_data = event.get("data", {})
                event_name = event.get("name", "")

                # Token-level streaming from LLM
                if event_kind == "on_chat_model_stream":
                    chunk = event_data.get("chunk")
                    if chunk:
                        # Text content token
                        if hasattr(chunk, "content") and chunk.content:
                            accumulated_content += chunk.content
                            yield StreamEvent(
                                event_type="token",
                                node="model",
                                data={"content": chunk.content, "accumulated": accumulated_content},
                            )

                        # Tool call chunks (from patched langchain-snowflake)
                        # Can be ToolCallChunk objects OR dicts depending on LangChain version
                        if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                            for tc_chunk in chunk.tool_call_chunks:
                                # Handle both dict and object formats
                                tc_name = tc_chunk.get("name") if isinstance(tc_chunk, dict) else getattr(tc_chunk, "name", None)
                                tc_args = tc_chunk.get("args") if isinstance(tc_chunk, dict) else getattr(tc_chunk, "args", None)
                                tc_id = tc_chunk.get("id") if isinstance(tc_chunk, dict) else getattr(tc_chunk, "id", None)
                                
                                if tc_name:
                                    current_tool_name = tc_name
                                    yield StreamEvent(
                                        event_type="tool_call_start",
                                        node="model",
                                        data={"tool_name": current_tool_name, "tool_id": tc_id},
                                    )
                                if tc_args:
                                    current_tool_args += tc_args
                                    yield StreamEvent(
                                        event_type="tool_call_args",
                                        node="model",
                                        data={
                                            "tool_name": current_tool_name,
                                            "partial_args": tc_args,
                                            "accumulated_args": current_tool_args,
                                        },
                                    )

                # Chat model finished (tool call complete or response done)
                elif event_kind == "on_chat_model_end":
                    output = event_data.get("output")
                    if output and hasattr(output, "tool_calls") and output.tool_calls:
                        # Tool call completed
                        for tool_call in output.tool_calls:
                            yield StreamEvent(
                                event_type="tool_call_complete",
                                node="model",
                                data={
                                    "tool_name": tool_call.get("name", ""),
                                    "tool_id": tool_call.get("id", ""),
                                    "args": tool_call.get("args", {}),
                                },
                            )
                        # Reset for next tool call
                        current_tool_name = ""
                        current_tool_args = ""
                    elif output and hasattr(output, "content") and output.content:
                        # Final answer
                        if not current_tool_name:  # Not during tool calling
                            yield StreamEvent(
                                event_type="react_answer",
                                node="model",
                                data={"answer": output.content},
                            )

                # Tool execution started
                elif event_kind == "on_tool_start":
                    yield StreamEvent(
                        event_type="tool_executing",
                        node="tools",
                        data={"tool_name": event_name, "input": str(event_data.get("input", ""))[:200]},
                    )

                # Tool execution completed
                elif event_kind == "on_tool_end":
                    tool_output = event_data.get("output", "")
                    if isinstance(tool_output, ToolMessage):
                        all_messages.append(tool_output)
                        tool_output = tool_output.content
                    yield StreamEvent(
                        event_type="tool_result",
                        node="tools",
                        data={"tool_name": event_name, "result_preview": str(tool_output)[:300]},
                    )

                # Node transitions
                elif event_kind == "on_chain_end" and event_name == "final_answer":
                    final_answer = event_data.get("output", {}).get("final_answer", "")
                    if final_answer:
                        yield StreamEvent(
                            event_type="react_answer",
                            node="final_answer",
                            data={"answer": final_answer},
                        )

            # Extract final results
            tools_used = self._extract_tools_from_messages(all_messages)
            routing = self._determine_routing(tools_used)
            analyst_results = self._extract_analyst_results(all_messages)
            search_results = self._extract_search_results(all_messages)

            yield StreamEvent(
                event_type="complete",
                node="react_workflow",
                data={
                    "thread_id": thread_id,
                    "routing": routing,
                    "analyst_results": analyst_results,
                    "search_results": search_results,
                },
            )

        except Exception as exc:
            logger.error(f"Token stream error: {exc}", exc_info=True)
            yield StreamEvent(
                event_type="error",
                node="react_workflow",
                data={"message": str(exc)},
            )
