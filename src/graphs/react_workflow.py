"""Modern ReAct workflow using LangGraph message-based state and ToolNode.

Implements the Reasoning + Acting pattern with:
- Message-based state (HumanMessage, AIMessage, ToolMessage)
- @tool decorated functions for automatic schema generation
- ToolNode for automatic tool execution
- Tool calls routing via AIMessage.tool_calls

Uses Snowflake Cortex COMPLETE with structured output, wrapped to return
AIMessage objects with tool_calls for LangGraph compatibility.
"""

import asyncio
import logging
import uuid
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.config import settings
from src.graphs.react_prompts import build_react_prompt
from src.graphs.react_state import (
    ConversationTurn,
    ErrorHandlerOutput,
    FinalAnswerOutput,
    HealthcareAgentState,
    ModelOutput,
)
from src.tools.healthcare_tools import get_healthcare_tools
from src.tools.healthcare_tools import set_snowpark_session as set_tools_session

logger = logging.getLogger(__name__)

# Module-level session for SPCS OAuth token auth
_snowpark_session = None

# Get tools for ToolNode
HEALTHCARE_TOOLS = get_healthcare_tools()
TOOL_NODE = ToolNode(HEALTHCARE_TOOLS)

# Tool name to function mapping for schema generation
TOOLS_BY_NAME = {tool.name: tool for tool in HEALTHCARE_TOOLS}


def set_snowpark_session(session) -> None:
    """Set the Snowpark session for use by workflow nodes.

    In SPCS, this is called during app startup with the OAuth-authenticated session.
    """
    global _snowpark_session
    _snowpark_session = session
    # Also set session for tools module
    set_tools_session(session)
    logger.info("Snowpark session set for ReAct workflow")


# =============================================================================
# Cortex COMPLETE Wrapper - Returns AIMessage with tool_calls
# =============================================================================


# Response format schema for AI_COMPLETE structured output
# Following Snowflake docs: https://docs.snowflake.com/en/user-guide/snowflake-cortex/complete-structured-outputs
RESPONSE_FORMAT_SQL = """{
    'type': 'json',
    'schema': {
        'type': 'object',
        'properties': {
            'thought': {'type': 'string'},
            'action': {'type': 'string'},
            'action_input': {
                'type': 'object',
                'properties': {
                    'query': {'type': 'string'},
                    'member_id': {'type': ['string', 'null']},
                    'answer': {'type': 'string'}
                },
                'required': ['query']
            }
        },
        'required': ['thought', 'action', 'action_input']
    }
}"""


async def call_cortex_complete(prompt: str, model: str = "llama3.1-70b") -> AIMessage:
    """Call Snowflake Cortex AI_COMPLETE with structured output and return AIMessage.

    Uses SQL-based AI_COMPLETE with named parameters:
    - model_parameters for temperature
    - response_format for JSON schema

    Args:
        prompt: Full prompt including system message and conversation.
        model: Cortex model to use.

    Returns:
        AIMessage with either tool_calls or content.
    """
    if _snowpark_session is None:
        logger.error("No Snowpark session available")
        return AIMessage(content="I apologize, but I cannot process your request at this time.")

    # Escape single quotes for SQL string
    escaped_prompt = prompt.replace("'", "''")

    # Build SQL with AI_COMPLETE using named parameters (per Snowflake docs)
    sql = f"""
    SELECT AI_COMPLETE(
        model => '{model}',
        prompt => '{escaped_prompt}',
        model_parameters => {{'temperature': 0}},
        response_format => {RESPONSE_FORMAT_SQL}
    ) AS response
    """

    def _execute_sql():
        """Execute SQL (blocking call)."""
        return _snowpark_session.sql(sql).collect()

    try:
        # Run blocking SQL in thread pool
        result = await asyncio.to_thread(_execute_sql)

        if not result:
            logger.error("Empty result from AI_COMPLETE")
            return AIMessage(content="I couldn't process your request.")

        # Get the response - AI_COMPLETE returns the result directly
        response_text = result[0]["RESPONSE"]
        logger.info(f"Raw LLM response type: {type(response_text)}, value: {str(response_text)[:300]}")

        # Parse the structured JSON response
        import orjson

        if response_text is None:
            logger.error("AI_COMPLETE returned None")
            return AIMessage(content="I couldn't generate a response.")

        # Handle both string and dict responses
        if isinstance(response_text, str):
            parsed = orjson.loads(response_text)
        elif isinstance(response_text, dict):
            parsed = response_text
        else:
            logger.error(f"Unexpected response type: {type(response_text)}")
            return AIMessage(content="I encountered an unexpected response format.")

        thought = parsed.get("thought", "")
        action = parsed.get("action", "FINAL_ANSWER")
        action_input = parsed.get("action_input", {})

        logger.info(f"Cortex COMPLETE: action={action}, thought={thought[:50]}...")

        # Convert to AIMessage
        if action == "FINAL_ANSWER":
            # Final answer - return AIMessage with content
            answer = action_input.get("answer", thought)
            return AIMessage(content=answer)

        # Tool call - return AIMessage with tool_calls
        tool_call_id = f"call_{uuid.uuid4().hex[:8]}"

        # Build tool arguments based on tool schema
        tool_args = {}
        if action == "query_member_data":
            tool_args["query"] = action_input.get("query", "")
            if action_input.get("member_id"):
                tool_args["member_id"] = action_input["member_id"]
        elif action == "search_knowledge":
            tool_args["query"] = action_input.get("query", "")

        return AIMessage(
            content=thought,  # Store thought in content for debugging
            tool_calls=[
                {
                    "id": tool_call_id,
                    "name": action,
                    "args": tool_args,
                }
            ],
        )

    except Exception as e:
        logger.error(f"Cortex COMPLETE failed: {e}")
        return AIMessage(content=f"I encountered an error: {str(e)}")


# =============================================================================
# Graph Nodes
# =============================================================================


async def model_node(state: HealthcareAgentState) -> ModelOutput:
    """Call Cortex COMPLETE and return AIMessage.

    This node:
    1. Builds the prompt from conversation history and messages
    2. Calls Cortex COMPLETE
    3. Returns AIMessage (with tool_calls or final answer)
    """
    iteration = state.get("iteration", 0) + 1
    max_iterations = state.get("max_iterations", 5)

    # Check iteration limit
    if iteration > max_iterations:
        logger.warning(f"Max iterations ({max_iterations}) reached")
        return {
            "messages": [AIMessage(content="I've reached my processing limit. Based on what I found, please try a more specific question.")],
            "iteration": iteration,
        }

    # Get conversation context
    user_query = state.get("user_query", "")
    member_id = state.get("member_id")
    conversation_history = state.get("conversation_history", [])
    messages = state.get("messages", [])

    # Build member context string
    member_context = f"Current member ID: {member_id}" if member_id else "No specific member context"

    # Build prompt with full context
    prompt = build_react_prompt(
        user_query=user_query,
        member_id=member_id,
        conversation_history=conversation_history,
        scratchpad=_messages_to_scratchpad(messages),
        member_context=member_context,
    )

    # Get model from settings
    model = getattr(settings, "cortex_llm_model", "llama3.1-70b")

    # Call Cortex COMPLETE
    ai_message = await call_cortex_complete(prompt, model)

    logger.info(f"ReAct iteration {iteration}: tool_calls={bool(ai_message.tool_calls)}")

    return {
        "messages": [ai_message],
        "iteration": iteration,
    }


def _messages_to_scratchpad(messages: list) -> list[dict]:
    """Convert messages to scratchpad format for prompt building.

    Extracts thought/action/observation from message sequence.
    """
    scratchpad = []
    i = 0

    while i < len(messages):
        msg = messages[i]

        if isinstance(msg, AIMessage) and msg.tool_calls:
            # AI decided to call a tool
            tool_call = msg.tool_calls[0]
            step = {
                "thought": msg.content,  # Thought stored in content
                "action": tool_call["name"],
                "action_input": tool_call["args"],
                "observation": "",
            }

            # Look for corresponding ToolMessage
            if i + 1 < len(messages) and isinstance(messages[i + 1], ToolMessage):
                step["observation"] = messages[i + 1].content
                i += 1

            scratchpad.append(step)

        i += 1

    return scratchpad


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
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_used.append(tc["name"])

    # Second pass: get final answer from last non-tool AIMessage
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
            final_answer = msg.content
            break

    # Build conversation turn for history
    turn = ConversationTurn(
        user_query=user_query,
        member_id=member_id,
        final_answer=final_answer,
        tools_used=list(reversed(tools_used)),  # Chronological order
    )

    logger.info(f"Final answer: {len(final_answer)} chars, tools: {tools_used}")

    return {
        "final_answer": final_answer,
        "conversation_history": [turn],
    }


async def error_handler_node(state: HealthcareAgentState) -> ErrorHandlerOutput:
    """Handle errors during graph execution."""
    error_count = state.get("error_count", 0) + 1
    last_error = state.get("last_error")

    logger.error(f"Error handler invoked (count: {error_count}): {last_error}")

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
        if last_message.tool_calls:
            logger.info(f"Routing to tools: {last_message.tool_calls[0]['name']}")
            return "tools"
        else:
            logger.info("Routing to final_answer")
            return "final_answer"

    # Unexpected message type
    logger.warning(f"Unexpected message type: {type(last_message)}")
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
# Initial State Builder
# =============================================================================


def build_initial_state(
    user_query: str,
    member_id: str | None = None,
    tenant_id: str = "default",
    execution_id: str = "",
    max_iterations: int = 5,
) -> HealthcareAgentState:
    """Build initial state for graph execution.

    Args:
        user_query: The user's question.
        member_id: Optional member ID for context.
        tenant_id: Tenant identifier.
        execution_id: Unique execution identifier.
        max_iterations: Maximum ReAct iterations (default: 5).

    Returns:
        Initial HealthcareAgentState ready for graph.ainvoke().
    """
    return HealthcareAgentState(
        messages=[HumanMessage(content=user_query.strip())],
        user_query=user_query.strip(),
        member_id=member_id,
        tenant_id=tenant_id,
        conversation_history=[],
        iteration=0,
        max_iterations=max_iterations,
        execution_id=execution_id or f"exec_{uuid.uuid4().hex[:8]}",
        final_answer=None,
        error_count=0,
        last_error=None,
        has_error=False,
    )


# =============================================================================
# Module Initialization for langgraph dev
# =============================================================================


def _initialize_snowpark_session() -> None:
    """Initialize Snowpark session at module load time.

    Creates connection using either:
    - SPCS OAuth token (when running in Snowpark Container Services)
    - Local credentials (when running in development)
    """
    global _snowpark_session

    if _snowpark_session is not None:
        return

    from snowflake.snowpark import Session

    if settings.is_spcs:
        # SPCS mode - use OAuth token
        import os
        from pathlib import Path

        logger.info("Initializing Snowpark session with SPCS OAuth token")
        token_path = Path("/snowflake/session/token")
        connection_params = {
            "account": os.environ.get("SNOWFLAKE_ACCOUNT", ""),
            "host": os.environ.get("SNOWFLAKE_HOST", ""),
            "authenticator": "oauth",
            "token": token_path.read_text(),
            "database": settings.snowflake_database,
            "warehouse": settings.snowflake_warehouse,
        }
    else:
        # Local mode - use key pair authentication
        logger.info("Initializing Snowpark session with key pair authentication")
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import serialization

        with open(settings.snowflake_private_key_path, "rb") as key_file:
            private_key = serialization.load_pem_private_key(
                key_file.read(),
                password=settings.snowflake_private_key_passphrase.encode() if settings.snowflake_private_key_passphrase else None,
                backend=default_backend(),
            )

        private_key_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        connection_params = {
            "account": settings.snowflake_account,
            "user": settings.snowflake_user,
            "private_key": private_key_bytes,
            "database": settings.snowflake_database,
            "warehouse": settings.snowflake_warehouse,
            "role": settings.snowflake_role,
        }

    try:
        _snowpark_session = Session.builder.configs(connection_params).create()
        set_snowpark_session(_snowpark_session)
        logger.info("Snowpark session initialized successfully for ReAct workflow")
    except Exception as exc:
        logger.error(f"Failed to initialize Snowpark session: {exc}")
        raise


# Initialize Snowpark session when module is loaded
_initialize_snowpark_session()

# Export compiled graph for langgraph.json
react_healthcare_graph = build_react_graph().compile()
