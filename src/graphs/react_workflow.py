"""ReAct workflow implementation for Healthcare Agent.

Implements the Reasoning + Acting pattern with iterative tool execution.
The agent alternates between thinking, acting, and observing until
it can deliver a final answer.
"""

import asyncio
import logging
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.config import settings
from src.graphs.react_parser import format_observation, parse_react_response
from src.graphs.react_prompts import build_react_prompt
from src.graphs.react_state import (
    ConversationTurn,
    ErrorHandlerOutput,
    FinalAnswerOutput,
    HealthcareReActState,
    ObservationOutput,
    ReActStep,
    ReasonerOutput,
    ToolOutput,
)
from src.services.cortex_tools import AsyncCortexAnalystTool, AsyncCortexSearchTool

logger = logging.getLogger(__name__)

# Module-level session for SPCS OAuth token auth
_snowpark_session = None


def set_snowpark_session(session) -> None:
    """Set the Snowpark session for use by workflow nodes.

    In SPCS, this is called during app startup with the OAuth-authenticated session.
    """
    global _snowpark_session
    _snowpark_session = session
    logger.info("Snowpark session set for ReAct workflow")


def get_snowpark_session():
    """Get the current Snowpark session.

    Returns:
        Active Snowpark Session or None if not set.
    """
    return _snowpark_session


# =============================================================================
# Helper Functions
# =============================================================================


def _synthesize_fallback_answer(state: HealthcareReActState) -> str:
    """Synthesize a user-friendly fallback answer from scratchpad observations.

    When the LLM fails to provide a proper FINAL_ANSWER, this function
    extracts useful information from tool observations to provide a
    meaningful response instead of leaking internal reasoning.

    Args:
        state: Current ReAct state with scratchpad.

    Returns:
        User-friendly answer string.
    """
    scratchpad = state.get("scratchpad", [])
    user_query = state.get("user_query", "your question")

    if not scratchpad:
        return f"I apologize, but I couldn't find the information to answer: {user_query}"

    # Collect observations from tools
    observations = []
    for step in scratchpad:
        obs = step.get("observation", "")
        if obs and not obs.startswith("[Error]"):
            observations.append(obs)

    if not observations:
        return f"I apologize, but I encountered issues while searching for information about: {user_query}"

    # Build a response from observations
    response_parts = ["Based on my search, here's what I found:\n"]

    for obs in observations:
        # Clean up the observation for user presentation
        # Remove internal prefixes like "[Database Query Results]"
        cleaned = obs
        if cleaned.startswith("["):
            # Find the closing bracket and skip the prefix
            bracket_end = cleaned.find("]")
            if bracket_end > 0:
                cleaned = cleaned[bracket_end + 1:].strip()

        if cleaned:
            response_parts.append(cleaned)

    if len(response_parts) == 1:
        # No useful observations extracted
        return f"I searched for information but couldn't find a definitive answer to: {user_query}"

    response_parts.append("\n\nIf you need more specific information, please let me know.")
    return "\n".join(response_parts)


# =============================================================================
# ReAct Nodes
# =============================================================================


async def reasoner_node(state: HealthcareReActState) -> ReasonerOutput:
    """Reasoning node that decides what action to take next.

    Uses Cortex COMPLETE to analyze the query and scratchpad,
    then outputs a Thought/Action/Action Input.

    Args:
        state: Current ReAct state with user_query and scratchpad.

    Returns:
        ReasonerOutput with current_step containing the reasoning decision.
    """
    session = get_snowpark_session()
    if not session:
        logger.error("No Snowpark session available for reasoner")
        return ReasonerOutput(
            current_step=ReActStep(
                thought="System error - no database connection",
                action="FINAL_ANSWER",
                action_input={"answer": "I apologize, but I'm unable to process your request due to a system issue."},
                observation="",
            ),
            iteration=state.get("iteration", 0) + 1,
            has_error=True,
            last_error={"message": "No Snowpark session", "node": "reasoner"},
        )

    # Build the ReAct prompt with conversation history for context
    prompt = build_react_prompt(
        user_query=state.get("user_query", ""),
        member_id=state.get("member_id"),
        scratchpad=state.get("scratchpad", []),
        conversation_history=state.get("conversation_history", []),
    )

    try:
        # Call Cortex COMPLETE for reasoning
        escaped_prompt = prompt.replace("'", "''")
        model = getattr(settings, "cortex_llm_model", "llama3.1-70b")

        sql = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            '{model}',
            '{escaped_prompt}'
        ) AS response
        """

        # Execute in thread pool to avoid blocking the event loop
        def _execute_cortex_complete():
            return session.sql(sql).collect()

        result = await asyncio.to_thread(_execute_cortex_complete)
        response_text = result[0][0] if result else ""

        logger.debug(f"Reasoner LLM response: {response_text[:500]}...")

        # Parse the response
        parsed = parse_react_response(response_text)

        current_step = ReActStep(
            thought=parsed["thought"],
            action=parsed["action"],
            action_input=parsed["action_input"],
            observation="",  # Will be filled by tool execution
        )

        iteration = state.get("iteration", 0) + 1
        logger.info(f"ReAct iteration {iteration}: action={parsed['action']}")

        return ReasonerOutput(
            current_step=current_step,
            iteration=iteration,
            has_error=False,
        )

    except Exception as exc:
        logger.error(f"Reasoner error: {exc}", exc_info=True)
        return ReasonerOutput(
            current_step=ReActStep(
                thought="Error during reasoning",
                action="FINAL_ANSWER",
                action_input={"answer": "I apologize, but I encountered an error while processing your request."},
                observation="",
            ),
            iteration=state.get("iteration", 0) + 1,
            has_error=True,
            last_error={"message": str(exc), "node": "reasoner"},
        )


async def analyst_tool_node(state: HealthcareReActState) -> ToolOutput:
    """Execute Cortex Analyst for database queries.

    Args:
        state: Current state with action_input from reasoner.

    Returns:
        ToolOutput with formatted observation.
    """
    session = get_snowpark_session()
    if not session:
        return ToolOutput(
            tool_result="[Error] Database connection unavailable",
            has_error=True,
            last_error={"message": "No Snowpark session", "node": "analyst_tool"},
            error_count=state.get("error_count", 0) + 1,
        )

    current_step = state.get("current_step")
    if not current_step:
        return ToolOutput(
            tool_result="[Error] No action specified",
            has_error=True,
            last_error={"message": "Missing current_step", "node": "analyst_tool"},
            error_count=state.get("error_count", 0) + 1,
        )

    action_input = current_step.get("action_input", {})
    query = action_input.get("query", state.get("user_query", ""))
    member_id = action_input.get("member_id", state.get("member_id"))

    try:
        analyst_tool = AsyncCortexAnalystTool(session)
        result = await analyst_tool.execute(query, member_id)

        # Format as observation
        observation = format_observation("query_member_data", result)

        logger.info(f"Analyst tool completed: {len(observation)} chars")
        return ToolOutput(
            tool_result=observation,
            has_error=False,
        )

    except Exception as exc:
        logger.error(f"Analyst tool error: {exc}", exc_info=True)
        return ToolOutput(
            tool_result=f"[Error] Database query failed: {exc}",
            has_error=True,
            last_error={"message": str(exc), "node": "analyst_tool"},
            error_count=state.get("error_count", 0) + 1,
        )


async def search_tool_node(state: HealthcareReActState) -> ToolOutput:
    """Execute Cortex Search for knowledge base queries.

    Args:
        state: Current state with action_input from reasoner.

    Returns:
        ToolOutput with formatted observation.
    """
    session = get_snowpark_session()
    if not session:
        return ToolOutput(
            tool_result="[Error] Database connection unavailable",
            has_error=True,
            last_error={"message": "No Snowpark session", "node": "search_tool"},
            error_count=state.get("error_count", 0) + 1,
        )

    current_step = state.get("current_step")
    if not current_step:
        return ToolOutput(
            tool_result="[Error] No action specified",
            has_error=True,
            last_error={"message": "Missing current_step", "node": "search_tool"},
            error_count=state.get("error_count", 0) + 1,
        )

    action_input = current_step.get("action_input", {})
    query = action_input.get("query", state.get("user_query", ""))

    try:
        search_tool = AsyncCortexSearchTool(session)
        results = await search_tool.execute(query)

        # Format as observation
        observation = format_observation("search_knowledge", results)

        logger.info(f"Search tool completed: {len(results)} results")
        return ToolOutput(
            tool_result=observation,
            has_error=False,
        )

    except Exception as exc:
        logger.error(f"Search tool error: {exc}", exc_info=True)
        return ToolOutput(
            tool_result=f"[Error] Knowledge base search failed: {exc}",
            has_error=True,
            last_error={"message": str(exc), "node": "search_tool"},
            error_count=state.get("error_count", 0) + 1,
        )


async def observation_node(state: HealthcareReActState) -> ObservationOutput:
    """Capture tool result and append to scratchpad.

    This node closes the ReAct loop by recording the observation
    and preparing for the next reasoning iteration.

    Args:
        state: Current state with tool_result and current_step.

    Returns:
        ObservationOutput with updated scratchpad.
    """
    current_step = state.get("current_step")
    tool_result = state.get("tool_result", "")

    if not current_step:
        logger.warning("Observation node called without current_step")
        return ObservationOutput(
            scratchpad=state.get("scratchpad", []),
            current_step=None,
            tool_result=None,
        )

    # Complete the step with the observation
    completed_step = ReActStep(
        thought=current_step.get("thought", ""),
        action=current_step.get("action", ""),
        action_input=current_step.get("action_input", {}),
        observation=tool_result,
    )

    # Append to scratchpad
    updated_scratchpad = list(state.get("scratchpad", []))
    updated_scratchpad.append(completed_step)

    logger.info(f"Observation recorded, scratchpad now has {len(updated_scratchpad)} steps")

    return ObservationOutput(
        scratchpad=updated_scratchpad,
        current_step=None,  # Clear for next iteration
        tool_result=None,
    )


async def final_answer_node(state: HealthcareReActState) -> FinalAnswerOutput:
    """Extract final answer and append completed turn to conversation history.

    This node:
    1. Extracts the final answer from the last reasoning step
    2. Creates a ConversationTurn capturing the query and answer
    3. Appends the turn to conversation_history for persistence via checkpointer

    Args:
        state: Current state with current_step containing FINAL_ANSWER.

    Returns:
        FinalAnswerOutput with the answer text and updated conversation_history.
    """
    current_step = state.get("current_step")

    if not current_step:
        return FinalAnswerOutput(
            final_answer="I apologize, but I was unable to formulate a response.",
            current_node="final_answer",
            conversation_history=[],  # No turn to add
        )

    action_input = current_step.get("action_input", {})
    answer = action_input.get("answer", "")

    if not answer:
        # IMPORTANT: Do NOT use raw "thought" - it contains internal reasoning, not a user-facing answer
        # Instead, try to synthesize a response from observations in the scratchpad
        answer = _synthesize_fallback_answer(state)

    logger.info(f"Final answer generated: {len(answer)} chars")

    # Extract tools used from scratchpad for context in history
    scratchpad = state.get("scratchpad", [])
    tools_used = [step.get("action") for step in scratchpad if step.get("action") and step.get("action") != "FINAL_ANSWER"]

    # Create the completed conversation turn to persist
    completed_turn = ConversationTurn(
        user_query=state.get("user_query", ""),
        member_id=state.get("member_id"),
        final_answer=answer,
        tools_used=tools_used,
    )

    return FinalAnswerOutput(
        final_answer=answer,
        current_node="final_answer",
        # This uses the _append_turns reducer to accumulate turns
        conversation_history=[completed_turn],
    )


async def error_handler_node(state: HealthcareReActState) -> ErrorHandlerOutput:
    """Handle errors and decide whether to retry or fail.

    Args:
        state: Current state with error information.

    Returns:
        ErrorHandlerOutput with error status.
    """
    error_count = state.get("error_count", 0)
    max_errors = 3

    if error_count >= max_errors:
        logger.warning(f"Max errors ({max_errors}) reached, ending execution")
        return ErrorHandlerOutput(
            has_error=True,
            error_count=error_count,
            last_error=state.get("last_error"),
            current_node="error_handler",
        )

    # Allow retry
    logger.info(f"Error {error_count}/{max_errors}, allowing retry")
    return ErrorHandlerOutput(
        has_error=False,
        error_count=error_count,
        current_node="error_handler",
    )


# =============================================================================
# Routing Functions
# =============================================================================


def route_by_action(
    state: HealthcareReActState,
) -> Literal["analyst_tool", "search_tool", "final_answer", "error", "max_iterations"]:
    """Route based on the reasoner's action decision.

    Args:
        state: Current state with current_step containing the action.

    Returns:
        Next node name to execute.
    """
    # Check max iterations first
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 5)

    if iteration >= max_iterations:
        logger.warning(f"Max iterations ({max_iterations}) reached")
        return "max_iterations"

    # Check for errors
    if state.get("has_error"):
        return "error"

    current_step = state.get("current_step")
    if not current_step:
        logger.error("No current_step in state")
        return "error"

    action = current_step.get("action", "").lower()

    # Map actions to nodes
    if action == "query_member_data":
        return "analyst_tool"
    elif action == "search_knowledge":
        return "search_tool"
    elif action == "final_answer":
        return "final_answer"
    else:
        logger.warning(f"Unknown action: {action}, defaulting to final_answer")
        return "final_answer"


def route_after_error(
    state: HealthcareReActState,
) -> Literal["reasoner", "end"]:
    """Route after error handling.

    Args:
        state: Current state with error information.

    Returns:
        Next node: retry (reasoner) or end.
    """
    error_count = state.get("error_count", 0)
    max_errors = 3

    if error_count >= max_errors:
        return "end"

    # Retry
    return "reasoner"


# =============================================================================
# Graph Builder
# =============================================================================


def build_react_graph() -> StateGraph:
    """Build the ReAct workflow graph.

    Returns:
        Compiled StateGraph with ReAct loop structure.
    """
    graph = StateGraph(HealthcareReActState)

    # Add nodes
    graph.add_node("reasoner", reasoner_node)
    graph.add_node("analyst_tool", analyst_tool_node)
    graph.add_node("search_tool", search_tool_node)
    graph.add_node("observation", observation_node)
    graph.add_node("final_answer", final_answer_node)
    graph.add_node("error_handler", error_handler_node)

    # Entry point
    graph.set_entry_point("reasoner")

    # Routing from reasoner based on action
    graph.add_conditional_edges(
        "reasoner",
        route_by_action,
        {
            "analyst_tool": "analyst_tool",
            "search_tool": "search_tool",
            "final_answer": "final_answer",
            "error": "error_handler",
            "max_iterations": "final_answer",  # Force final answer on max iterations
        },
    )

    # Tools feed to observation (the ReAct loop!)
    graph.add_edge("analyst_tool", "observation")
    graph.add_edge("search_tool", "observation")

    # Observation loops back to reasoner
    graph.add_edge("observation", "reasoner")

    # Final answer ends the graph
    graph.add_edge("final_answer", END)

    # Error handling
    graph.add_conditional_edges(
        "error_handler",
        route_after_error,
        {
            "reasoner": "reasoner",
            "end": "final_answer",
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
# Convenience function for AgentService compatibility
# =============================================================================


def build_initial_react_state(
    user_query: str,
    member_id: str | None = None,
    tenant_id: str = "default",
    execution_id: str = "",
    max_iterations: int = 5,
) -> HealthcareReActState:
    """Build initial state for ReAct workflow execution.

    Args:
        user_query: The user's question.
        member_id: Optional member ID for context.
        tenant_id: Tenant identifier.
        execution_id: Unique execution identifier.
        max_iterations: Maximum ReAct iterations (default: 5).

    Returns:
        Initial HealthcareReActState ready for graph.ainvoke().
    """
    return HealthcareReActState(
        user_query=user_query.strip(),
        member_id=member_id,
        tenant_id=tenant_id,
        scratchpad=[],
        current_step=None,
        iteration=0,
        max_iterations=max_iterations,
        tool_result=None,
        final_answer=None,
        execution_id=execution_id,
        thread_checkpoint_id=None,
        current_node="start",
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

        logger.info("Initializing Snowpark session with SPCS OAuth token")
        connection_params = {
            "account": os.environ.get("SNOWFLAKE_ACCOUNT", ""),
            "host": os.environ.get("SNOWFLAKE_HOST", ""),
            "authenticator": "oauth",
            "token": open("/snowflake/session/token").read(),
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
                password=settings.snowflake_private_key_passphrase.encode()
                if settings.snowflake_private_key_passphrase
                else None,
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
        logger.info("Snowpark session initialized successfully for ReAct workflow")
    except Exception as exc:
        logger.error(f"Failed to initialize Snowpark session: {exc}")
        raise


# Initialize Snowpark session when module is loaded
_initialize_snowpark_session()

# Export compiled graph for langgraph.json
# Note: LangGraph API handles persistence automatically, so we compile WITHOUT checkpointer
# The checkpointer is only added when deployed to SPCS via dependencies.py
react_healthcare_graph = build_react_graph().compile()

