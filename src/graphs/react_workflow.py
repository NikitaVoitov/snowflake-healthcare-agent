"""Modern ReAct workflow using LangGraph message-based state and ToolNode - FULLY ASYNC.

Implements the Reasoning + Acting pattern with:
- Message-based state (HumanMessage, AIMessage, ToolMessage)
- @tool decorated functions for automatic schema generation
- ToolNode for automatic tool execution
- Tool calls routing via AIMessage.tool_calls

Uses ChatSnowflake from langchain-snowflake for native Cortex integration
with automatic tool binding and async support.

ALL blocking operations are wrapped in asyncio.to_thread() to avoid
blocking the event loop.
"""

import asyncio
import logging
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.config import settings
from src.graphs.react_prompts import build_system_message
from src.graphs.react_state import (
    ConversationTurn,
    ErrorHandlerOutput,
    FinalAnswerOutput,
    HealthcareAgentState,
    ModelOutput,
)
from src.services.llm_service import get_chat_snowflake
from src.tools.healthcare_tools import get_healthcare_tools
from src.tools.healthcare_tools import set_snowpark_session as set_tools_session

logger = logging.getLogger(__name__)

# Module-level session for SPCS OAuth token auth (used by tools)
_snowpark_session = None
_session_initialized = False
_session_lock = asyncio.Lock()

# Get tools for ToolNode
HEALTHCARE_TOOLS = get_healthcare_tools()
TOOL_NODE = ToolNode(HEALTHCARE_TOOLS)

# Tool name to function mapping for schema generation
TOOLS_BY_NAME = {tool.name: tool for tool in HEALTHCARE_TOOLS}


def set_snowpark_session(session) -> None:
    """Set the Snowpark session for use by workflow nodes.

    In SPCS, this is called during app startup with the OAuth-authenticated session.
    The session is used by tools (Cortex Analyst, Cortex Search) not by ChatSnowflake.
    """
    global _snowpark_session, _session_initialized
    _snowpark_session = session
    _session_initialized = True
    # Also set session for tools module
    set_tools_session(session)
    logger.info("Snowpark session set for ReAct workflow")


async def _ensure_snowpark_session_async() -> None:
    """Ensure Snowpark session is initialized (ASYNC with locking).

    Initializes the session lazily on first use, avoiding blocking at module load.
    """
    global _snowpark_session, _session_initialized

    if _session_initialized:
        return

    async with _session_lock:
        # Double-check after acquiring lock
        if _session_initialized:
            return

        logger.info("Lazy-initializing Snowpark session...")
        await _initialize_snowpark_session_async()


async def _initialize_snowpark_session_async() -> None:
    """Initialize Snowpark session asynchronously.

    Creates connection using either:
    - SPCS OAuth token (when running in Snowpark Container Services)
    - Local credentials (when running in development)

    Note: This session is used by tools (Cortex Analyst, Cortex Search),
    not by ChatSnowflake which manages its own connection.
    """
    global _snowpark_session, _session_initialized

    if _session_initialized:
        return

    if settings.is_spcs:
        # SPCS mode - use OAuth token
        import os
        from pathlib import Path

        logger.info("Initializing Snowpark session with SPCS OAuth token")
        token_path = Path("/snowflake/session/token")

        # Read token asynchronously
        token = await asyncio.to_thread(token_path.read_text)

        connection_params = {
            "account": os.environ.get("SNOWFLAKE_ACCOUNT", ""),
            "host": os.environ.get("SNOWFLAKE_HOST", ""),
            "authenticator": "oauth",
            "token": token,
            "database": settings.snowflake_database,
            "warehouse": settings.snowflake_warehouse,
        }
    else:
        # Local mode - use key pair authentication
        logger.info("Initializing Snowpark session with key pair authentication")
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import serialization

        def _load_key_sync():
            with open(settings.snowflake_private_key_path, "rb") as key_file:
                private_key = serialization.load_pem_private_key(
                    key_file.read(),
                    password=settings.snowflake_private_key_passphrase.encode() if settings.snowflake_private_key_passphrase else None,
                    backend=default_backend(),
                )

            return private_key.private_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )

        # Load key asynchronously
        private_key_bytes = await asyncio.to_thread(_load_key_sync)

        connection_params = {
            "account": settings.snowflake_account,
            "user": settings.snowflake_user,
            "private_key": private_key_bytes,
            "database": settings.snowflake_database,
            "warehouse": settings.snowflake_warehouse,
            "role": settings.snowflake_role,
        }

    # Create session asynchronously (blocking network call)
    def _create_session_sync(params):
        from snowflake.snowpark import Session

        return Session.builder.configs(params).create()

    try:
        _snowpark_session = await asyncio.to_thread(_create_session_sync, connection_params)
        set_snowpark_session(_snowpark_session)
        _session_initialized = True
        logger.info("Snowpark session initialized successfully for ReAct workflow")
    except Exception as exc:
        logger.error("Failed to initialize Snowpark session: %s", exc)
        raise


# =============================================================================
# Message Construction for ChatSnowflake
# =============================================================================


def _build_chat_messages(state: HealthcareAgentState) -> list[BaseMessage]:
    """Convert state to LangChain messages for ChatSnowflake.

    Builds a message list with:
    1. System message containing healthcare context and tool guidance
    2. Existing messages from state (HumanMessage, AIMessage, ToolMessage)

    Args:
        state: Current graph state with messages and context.

    Returns:
        List of messages ready for ChatSnowflake.ainvoke().
    """
    messages: list[BaseMessage] = []

    # Build system message with context
    system_content = build_system_message(
        member_id=state.get("member_id"),
        conversation_history=state.get("conversation_history", []),
    )
    messages.append(SystemMessage(content=system_content))

    # Add existing messages (HumanMessage, AIMessage with tool_calls, ToolMessage)
    existing_messages = state.get("messages", [])
    messages.extend(existing_messages)

    return messages


# =============================================================================
# Graph Nodes
# =============================================================================


async def model_node(state: HealthcareAgentState) -> ModelOutput:
    """Call ChatSnowflake and return AIMessage (ASYNC).

    This node:
    1. Ensures Snowpark session is initialized (for tools)
    2. Checks iteration limits
    3. Builds messages from state
    4. Calls ChatSnowflake with bound tools
    5. Returns AIMessage (with tool_calls or final answer)
    """
    # Ensure session is initialized (lazy initialization)
    await _ensure_snowpark_session_async()

    iteration = state.get("iteration", 0) + 1
    max_iterations = state.get("max_iterations", settings.max_agent_steps)

    # Check iteration limit - synthesize answer from collected data
    if iteration > max_iterations:
        logger.warning("Max iterations (%d) reached", max_iterations)
        messages = state.get("messages", [])

        # Extract data from the last ToolMessage (most recent result)
        tool_data = None
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage) and msg.content:
                tool_data = msg.content
                break

        if tool_data:
            # Use the data to synthesize an answer
            answer = f"Based on my analysis:\n\n{tool_data}"
            logger.info("Synthesized answer from tool data: %d chars", len(tool_data))
        else:
            answer = "I've reached my processing limit. Based on what I found, please try a more specific question."

        return {
            "messages": [AIMessage(content=answer)],
            "iteration": iteration,
        }

    # Get ChatSnowflake with tools bound (ASYNC)

    llm = await get_chat_snowflake(tools=HEALTHCARE_TOOLS)
    logger.info("ChatSnowflake created, tools=%s", len(HEALTHCARE_TOOLS))

    # DEBUG: Log bound tools count
    if hasattr(llm, "_bound_tools"):
        tool_count = len(llm._bound_tools) if llm._bound_tools else 0
        logger.debug("Bound %d tools to LLM", tool_count)
    else:
        logger.debug("No _bound_tools attribute on LLM (SQL mode)")

    # Build messages for LLM
    messages = _build_chat_messages(state)

    # DEBUG: Log messages being sent
    logger.info("Messages count: %d, types: %s", len(messages), [type(m).__name__ for m in messages])

    try:
        # Call ChatSnowflake - returns AIMessage with tool_calls or content
        ai_message = await llm.ainvoke(messages)

        # Debug: Log detailed response info
        has_tool_calls = bool(ai_message.tool_calls) if hasattr(ai_message, "tool_calls") else False
        content_len = len(ai_message.content) if ai_message.content else 0
        response_metadata = getattr(ai_message, "response_metadata", {})
        usage_metadata = getattr(ai_message, "usage_metadata", {})

        logger.info(
            "ReAct iteration %d: tool_calls=%s, content_len=%d, content_preview=%s",
            iteration,
            has_tool_calls,
            content_len,
            (ai_message.content[:200] if ai_message.content else "EMPTY")[:200],
        )

        # Log additional debugging info if response is empty
        if content_len == 0 and not has_tool_calls:
            logger.warning(
                "Empty LLM response! response_metadata=%s, usage_metadata=%s, additional_kwargs=%s",
                response_metadata,
                usage_metadata,
                getattr(ai_message, "additional_kwargs", {}),
            )

        return {
            "messages": [ai_message],
            "iteration": iteration,
        }

    except Exception as e:
        logger.error("ChatSnowflake call failed: %s", e, exc_info=True)
        return {
            "messages": [AIMessage(content=f"I encountered an error: {e!s}")],
            "iteration": iteration,
            "has_error": True,
            "last_error": {"error_type": "llm_error", "message": str(e)},
        }


async def final_answer_node(state: HealthcareAgentState) -> FinalAnswerOutput:
    """Extract final answer and save to conversation history."""
    messages = state.get("messages", [])
    user_query = state.get("user_query", "")
    member_id = state.get("member_id")

    # Get final answer from last AIMessage and collect all tools used
    final_answer = "I could not find an answer to your question."
    tools_used = []

    # First pass: collect all tools from messages (in order)
    for msg in messages:
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_used.append(tc["name"])

    # Second pass: get final answer from last non-tool AIMessage
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not (hasattr(msg, "tool_calls") and msg.tool_calls) and msg.content:
            final_answer = msg.content
            break

    # Build conversation turn for history
    turn = ConversationTurn(
        user_query=user_query,
        member_id=member_id,
        final_answer=final_answer,
        tools_used=list(reversed(tools_used)),  # Chronological order
    )

    logger.info("Final answer: %d chars, tools: %s", len(final_answer), tools_used)

    return {
        "final_answer": final_answer,
        "conversation_history": [turn],
    }


async def error_handler_node(state: HealthcareAgentState) -> ErrorHandlerOutput:
    """Handle errors during graph execution."""
    error_count = state.get("error_count", 0) + 1
    last_error = state.get("last_error")

    logger.error("Error handler invoked (count: %d): %s", error_count, last_error)

    if error_count >= 3:
        # Too many errors, generate error response
        error_message = AIMessage(content="I apologize, but I encountered repeated errors. Please try again or rephrase your question.")
        return {
            "messages": [error_message],
            "has_error": False,  # Clear error to allow final_answer
            "error_count": error_count,
        }

    return {
        "has_error": True,
        "error_count": error_count,
    }


# =============================================================================
# Routing Logic
# =============================================================================


def route_after_model(state: HealthcareAgentState) -> Literal["tools", "final_answer", "error"]:
    """Route based on model output.

    - If AIMessage has tool_calls → route to tools
    - If AIMessage has content only → route to final_answer
    - If error → route to error_handler
    """
    if state.get("has_error"):
        return "error"

    messages = state.get("messages", [])
    if not messages:
        return "error"

    last_message = messages[-1]

    # Check if it's an AIMessage with tool_calls
    if isinstance(last_message, AIMessage):
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            logger.info("Routing to tools: %s", last_message.tool_calls[0]["name"])
            return "tools"
        else:
            logger.info("Routing to final_answer")
            return "final_answer"

    # Unexpected message type
    logger.warning("Unexpected message type: %s", type(last_message))
    return "error"


def route_after_error(state: HealthcareAgentState) -> Literal["model", "final_answer"]:
    """Route after error handling."""
    error_count = state.get("error_count", 0)

    if error_count >= 3:
        return "final_answer"

    return "model"


# =============================================================================
# Graph Construction
# =============================================================================


def build_react_graph() -> StateGraph:
    """Build the modern ReAct graph with ToolNode.

    Graph structure:
        START → model → [tools → model] (loop) → final_answer → END

    The loop continues until model returns AIMessage without tool_calls.
    """
    graph = StateGraph(HealthcareAgentState)

    # Add nodes
    graph.add_node("model", model_node)
    graph.add_node("tools", TOOL_NODE)  # ToolNode handles tool execution
    graph.add_node("final_answer", final_answer_node)
    graph.add_node("error_handler", error_handler_node)

    # Set entry point
    graph.set_entry_point("model")

    # Add edges
    graph.add_conditional_edges(
        "model",
        route_after_model,
        {
            "tools": "tools",
            "final_answer": "final_answer",
            "error": "error_handler",
        },
    )

    # Tools loop back to model (the ReAct loop!)
    graph.add_edge("tools", "model")

    # Final answer ends the graph
    graph.add_edge("final_answer", END)

    # Error handling
    graph.add_conditional_edges(
        "error_handler",
        route_after_error,
        {
            "model": "model",
            "final_answer": "final_answer",
        },
    )

    return graph


def compile_react_graph(checkpointer=None):
    """Compile the ReAct graph with optional checkpointer.

    Args:
        checkpointer: Optional LangGraph checkpointer for state persistence.

    Returns:
        Compiled graph ready for execution.
    """
    graph = build_react_graph()

    if checkpointer is None:
        checkpointer = MemorySaver()

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("ReAct healthcare workflow graph compiled")

    return compiled


# =============================================================================
# Module Initialization for langgraph dev
# =============================================================================

# NOTE: Session is NOT initialized at module load time.
# It is lazy-initialized on first request via _ensure_snowpark_session_async()
# This allows the event loop to be properly running when we make async calls.

# Export compiled graph for langgraph.json
react_healthcare_graph = build_react_graph().compile()
