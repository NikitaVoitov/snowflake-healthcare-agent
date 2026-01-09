"""Modern Healthcare Agent using LangChain v1 create_agent - FULLY ASYNC.

Implements a TRUE AGENT using LangChain v1's create_agent which provides:
- LLM-controlled decision loop (not hardcoded workflow)
- Dynamic prompt generation via @dynamic_prompt middleware
- Automatic tool binding and execution
- Built-in retry and error handling
- Integration with checkpointers for conversation persistence

Uses ChatSnowflake from langchain-snowflake for native Cortex integration
with automatic tool binding and async support.

ALL blocking operations are wrapped in asyncio.to_thread() to avoid
blocking the event loop.

Migration from StateGraph-based workflow:
- model_node → replaced by create_agent internal model loop
- route_after_model → replaced by agent's automatic tool/answer routing
- final_answer_node → handled via middleware/state
- ToolNode → automatically created by create_agent
- Dynamic system prompt → @dynamic_prompt middleware
"""

# Initialize OpenTelemetry instrumentation BEFORE any LangChain imports
# This enables telemetry capture when running via langgraph dev
from src.otel_setup import setup_opentelemetry

setup_opentelemetry()

import logging

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from src.config import settings
from src.graphs.react_state import ConversationTurn, HealthcareAgentState
from src.middleware.healthcare_prompts import create_healthcare_middleware

# Centralized session management - ONE source of truth
from src.services.llm_service import (
    get_chat_snowflake,
    init_session_sync,
    set_session,
)
from src.tools.healthcare_tools import get_healthcare_tools

logger = logging.getLogger(__name__)

# Get tools for agent
HEALTHCARE_TOOLS = get_healthcare_tools()

# Tool name to function mapping for schema generation
TOOLS_BY_NAME = {tool.name: tool for tool in HEALTHCARE_TOOLS}

# Extract tool names for agentic attribution in traces
_TOOL_NAMES = [tool.name for tool in HEALTHCARE_TOOLS]
_TOOL_NAMES_STR = ",".join(_TOOL_NAMES)


# Backward compatibility alias - delegates to centralized llm_service
def set_snowpark_session(session) -> None:
    """Set the Snowpark session (delegates to llm_service.set_session).

    In SPCS, this is called during app startup with the OAuth-authenticated session.
    The session is shared by both ChatSnowflake and tools.
    """
    set_session(session)
    logger.info("Snowpark session set for healthcare agent (via llm_service)")


# =============================================================================
# Agent Factory
# =============================================================================


async def build_healthcare_agent(
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Build the healthcare agent using create_agent.

    This is an async factory that:
    1. Gets ChatSnowflake with centralized session from llm_service
    2. Uses create_agent with dynamic prompt middleware

    Args:
        checkpointer: Optional checkpointer for state persistence.

    Returns:
        Compiled agent graph ready for execution.
    """
    # Get ChatSnowflake - llm_service handles session creation/caching
    # Tools also use the same session via llm_service.get_session()
    llm = await get_chat_snowflake()

    # Create the agent using LangChain v1 create_agent
    # This replaces the manual StateGraph construction
    agent = create_agent(
        model=llm,
        tools=HEALTHCARE_TOOLS,
        middleware=create_healthcare_middleware(),
        state_schema=HealthcareAgentState,
        checkpointer=checkpointer,
        name="healthcare_agent",
        # Max iterations handled via state.max_iterations
    )

    logger.info("Healthcare agent created with %d tools", len(HEALTHCARE_TOOLS))
    return agent


def build_healthcare_agent_sync(
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Build the healthcare agent synchronously (for module-level init).

    This is a synchronous wrapper for build_healthcare_agent that:
    1. Initializes Snowpark session via centralized llm_service
    2. Creates ChatSnowflake with session
    3. Uses create_agent with dynamic prompt middleware

    Args:
        checkpointer: Optional checkpointer for state persistence.

    Returns:
        Compiled agent graph ready for execution.
    """
    from langchain_snowflake import ChatSnowflake

    from src.services.llm_service import ENABLE_LLM_STREAMING, get_cached_session

    # Check if we're in SPCS mode
    if settings.is_spcs:
        # CRITICAL FIX: Read fresh token from file, NOT use cached session
        # SPCS refreshes /snowflake/session/token automatically, but if we
        # use a cached session, it holds the OLD token and fails after ~1 hour
        import os
        from src.services.snowflake_session import read_spcs_token
        
        token = read_spcs_token()
        host = os.getenv("SNOWFLAKE_HOST")
        if not host:
            raise ValueError("SNOWFLAKE_HOST not set in SPCS environment")
        account = host.replace(".snowflakecomputing.com", "")
        
        # SPCS: Use fresh token directly, NOT session
        # The patched auth.py adds authenticator='oauth' and host automatically
        # when token is provided, so no user parameter is needed
        llm = ChatSnowflake(
            model=settings.cortex_llm_model,
            temperature=0,
            disable_streaming=not ENABLE_LLM_STREAMING,
            token=token,
            account=account,
        )
        logger.info("ChatSnowflake created for SPCS (fresh token from file)")
    else:
        # Local: Initialize session with key-pair credentials
        session = init_session_sync()
        
        # Create ChatSnowflake WITH session AND key-pair credentials
        # Session is used for SQL mode, key-pair credentials for REST API (when tools are bound)
        llm = ChatSnowflake(
            model=settings.cortex_llm_model,
            temperature=0,
            disable_streaming=not ENABLE_LLM_STREAMING,
            session=session,
            # Key-pair credentials for REST API JWT auth (used when create_agent binds tools)
            account=settings.snowflake_account,
            user=settings.snowflake_user,
            private_key_path=settings.snowflake_private_key_path,
            private_key_passphrase=settings.snowflake_private_key_passphrase,
        )
        logger.info("ChatSnowflake created for local (key-pair auth)")

    # Create the agent using LangChain v1 create_agent
    agent = create_agent(
        model=llm,
        tools=HEALTHCARE_TOOLS,
        middleware=create_healthcare_middleware(),
        state_schema=HealthcareAgentState,
        checkpointer=checkpointer,
        name="healthcare_agent",
    )

    logger.info("Healthcare agent created (sync) with %d tools", len(HEALTHCARE_TOOLS))
    return agent


# =============================================================================
# Backward Compatibility: Compile Functions
# =============================================================================


def compile_react_graph(
    checkpointer: BaseCheckpointSaver | None = None,
    use_checkpointer: bool = True,
) -> CompiledStateGraph:
    """Compile the healthcare agent with optional checkpointer.

    This provides backward compatibility with existing code that uses
    compile_react_graph(). It creates an agent using create_agent.

    Args:
        checkpointer: Optional LangGraph checkpointer for state persistence.
        use_checkpointer: If False, compile without any checkpointer (for langgraph dev).

    Returns:
        Compiled agent graph ready for execution.
    """
    if not use_checkpointer:
        # For langgraph dev mode - platform handles persistence
        agent = build_healthcare_agent_sync(checkpointer=None)
    else:
        if checkpointer is None:
            checkpointer = MemorySaver()
        agent = build_healthcare_agent_sync(checkpointer=checkpointer)

    logger.info("Healthcare agent compiled (backward compat)")
    return agent


# =============================================================================
# Result Extraction Helpers (for AgentService compatibility)
# =============================================================================


def extract_final_answer(state: HealthcareAgentState) -> str:
    """Extract final answer from agent state.

    Looks for the last AIMessage without tool_calls.

    Args:
        state: Agent state after execution.

    Returns:
        Final answer string.
    """
    messages = state.get("messages", [])
    final_answer = state.get("final_answer", "")

    if final_answer:
        return final_answer

    # Find last AIMessage without tool_calls
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not (hasattr(msg, "tool_calls") and msg.tool_calls) and msg.content:
            return msg.content

    return "I could not find an answer to your question."


def extract_tools_used(state: HealthcareAgentState) -> list[str]:
    """Extract list of tools used from agent state.

    Args:
        state: Agent state after execution.

    Returns:
        List of tool names used.
    """
    messages = state.get("messages", [])
    tools_used = []

    for msg in messages:
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_used.append(tc.get("name", ""))

    return tools_used


def build_conversation_turn(state: HealthcareAgentState) -> ConversationTurn:
    """Build a ConversationTurn from agent state.

    Args:
        state: Agent state after execution.

    Returns:
        ConversationTurn for history persistence.
    """
    return ConversationTurn(
        user_query=state.get("user_query", ""),
        member_id=state.get("member_id"),
        final_answer=extract_final_answer(state),
        tools_used=extract_tools_used(state),
    )


# =============================================================================
# Module Initialization for langgraph dev
# =============================================================================

# NOTE: Session is NOT initialized at module load time.
# It is lazy-initialized on first request via _ensure_snowpark_session_async()
# This allows the event loop to be properly running when we make async calls.

# NOTE: For langgraph dev mode, do NOT pass any checkpointer.
# LangGraph API handles persistence automatically - custom checkpointers are REJECTED.
# To customize persistence, set POSTGRES_URI environment variable.
# See: https://langchain-ai.github.io/langgraph/cloud/reference/env_var/#postgres_uri_custom


# =============================================================================
# Graph Export for langgraph dev
# =============================================================================
#
# AGENT MARKERS FOR SPLUNK O11Y AGENT FLOW VISUALIZATION
# -------------------------------------------------------
# The "agent" tag triggers invoke_agent span (not workflow) in OTel instrumentation.
# This changes the trace from:
#   workflow react_healthcare → step model → chat
# To:
#   invoke_agent healthcare_agent → step model → chat
#
# Execution logic is UNCHANGED - only trace visualization changes.
#
# IMPORTANT: LangGraph dev server REPLACES custom metadata with its own system metadata
# but PRESERVES tags. Therefore, we pass agent attributes via tags using the format:
#   - agent_type:<type>
#   - agent_description:<description>
#   - agent_tools:<tool1,tool2,...>
#
# This ensures the OTel instrumentation can extract gen_ai.agent.* attributes
# even when running under LangGraph dev server.

# Lazy initialization - don't create at import time (SPCS session not available yet)
_react_healthcare_graph: CompiledStateGraph | None = None


def get_react_healthcare_graph() -> CompiledStateGraph:
    """Get the pre-configured healthcare agent graph (lazy initialization).

    In SPCS mode, this must be called AFTER main.py sets the session.
    """
    global _react_healthcare_graph
    if _react_healthcare_graph is None:
        _react_healthcare_graph = compile_react_graph(use_checkpointer=False).with_config(
            {
                "run_name": "healthcare_agent",
                # Tags: Agent markers and attributes (preserved by LangGraph dev server)
                # Format: "key:value" parsed by OTel instrumentation callback handler
                "tags": [
                    "agent",  # Triggers invoke_agent span type
                    "agent:healthcare_agent",  # Agent name (gen_ai.agent.name)
                    "agent_type:react",  # Agent type (gen_ai.agent.type)
                    "agent_description:Healthcare ReAct agent for member inquiries using Snowflake Cortex Analyst and Search",
                    f"agent_tools:{','.join(t.name for t in HEALTHCARE_TOOLS)}",
                ],
                # Metadata: Additional context (may be overwritten by LangGraph dev)
                "metadata": {
                    "agent_name": "healthcare_agent",
                    "agent_type": "react",
                },
            }
        )
    return _react_healthcare_graph


# NOTE: For backward compatibility, expose as module-level variable
# This will be lazily initialized on first access
# DEPRECATED: Use get_react_healthcare_graph() instead
react_healthcare_graph = None  # Set by first call to get_react_healthcare_graph()


# For LangGraph dev server compatibility
def _get_lazy_graph():
    """Get graph for langgraph dev server."""
    return get_react_healthcare_graph()


# Export the lazy getter for langgraph.json
__all__ = [
    "compile_react_graph",
    "build_healthcare_agent",
    "build_healthcare_agent_sync",
    "get_react_healthcare_graph",
    "react_healthcare_graph",
    # Helper functions
    "build_conversation_turn",
    "extract_final_answer",
    "extract_tools_used",
]
