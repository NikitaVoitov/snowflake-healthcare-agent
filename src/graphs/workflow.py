"""LangGraph workflow definition for Healthcare Multi-Agent system.

Implements async multi-agent orchestration with:
- Planner agent for routing decisions
- Analyst agent for member data queries (Cortex Analyst)
- Search agent for knowledge base queries (Cortex Search)
- Parallel execution using asyncio.TaskGroup
- Circuit breaker pattern for error handling
- Response synthesis with Cortex LLM
"""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Literal

from langgraph.graph import END, START, StateGraph
from snowflake.snowpark import Session

from src.config import settings
from src.graphs.state import (
    AgentOutput,
    ErrorOutput,
    EscalationOutput,
    HealthcareAgentState,
    ParallelOutput,
    PlannerOutput,
    ResponseOutput,
)
from src.models.agent_types import (
    AnalystResultModel,
    ErrorDetail,
    ErrorType,
    SearchResultModel,
)
from src.services.cortex_tools import AsyncCortexAnalystTool, AsyncCortexSearchTool

logger = logging.getLogger(__name__)

# Module-level Snowpark session - Snowpark sessions are thread-safe and can be shared
# Set via set_snowpark_session() at app startup
_snowpark_session: Session | None = None


def get_snowpark_session() -> Session | None:
    """Get the current Snowpark session.

    Returns:
        Current Snowpark session or None if not initialized.
    """
    return _snowpark_session


def set_snowpark_session(session: Session) -> None:
    """Set the Snowpark session for Cortex tools (called at app startup).

    Args:
        session: Initialized Snowpark Session instance.
    """
    global _snowpark_session  # noqa: PLW0603
    _snowpark_session = session
    logger.info("Snowpark session configured for workflow nodes")


# --- Agent Functions ---


async def async_planner_agent(state: HealthcareAgentState) -> PlannerOutput:
    """Route query to appropriate agent(s) based on intent.

    Analyzes query to determine if it needs:
    - analyst: Member-specific data (claims, coverage, demographics)
    - search: Knowledge base lookup (policies, FAQs, transcripts)
    - both: Combined member data + policy context

    Args:
        state: Current workflow state with user_query and member_id.

    Returns:
        PlannerOutput with plan routing decision and current_step.
    """
    query = state.get("user_query", "").lower()
    member_id = state.get("member_id")

    # Keywords indicating member data needs
    member_keywords = ["claim", "coverage", "balance", "deductible", "copay", "member", "my"]
    # Keywords indicating knowledge base needs
    knowledge_keywords = ["policy", "appeal", "procedure", "faq", "how to", "what is", "explain"]

    has_member_query = member_id and any(kw in query for kw in member_keywords)
    has_knowledge_query = any(kw in query for kw in knowledge_keywords)

    if has_member_query and has_knowledge_query:
        plan = "both"
    elif has_member_query:
        plan = "analyst"
    else:
        plan = "search"

    logger.info(f"Planner routing to: {plan} (member_id={member_id})")
    return {"plan": plan, "current_step": "planner"}


async def async_analyst_agent(state: HealthcareAgentState) -> AgentOutput:
    """Execute Cortex Analyst for member queries.

    Queries member data from MEMBER_SCHEMA tables using
    real Snowflake Cortex tools when session is available.

    Args:
        state: Current workflow state with plan, member_id, and user_query.

    Returns:
        AgentOutput with analyst_results or error information.
    """
    plan = state.get("plan", "")
    if plan not in ["analyst", "both"]:
        return {}

    member_id = state.get("member_id")
    query = state.get("user_query", "")

    try:
        # Use real Cortex tools if session is available
        session = get_snowpark_session()
        if session is not None:
            analyst_tool = AsyncCortexAnalystTool(session)
            results = await analyst_tool.execute(query, member_id)
            logger.info(f"Analyst executed real query for member: {member_id}")
        else:
            # Fallback to mock for testing/development
            logger.warning("No Snowpark session - using mock analyst results")
            results = {
                "member_id": member_id,
                "claims": [],
                "coverage": {"plan": "Gold", "deductible": 500, "copay_office": 25},
                "raw_response": None,
            }

        # Validate with Pydantic model
        validated = AnalystResultModel(**results)
        return {"analyst_results": validated.model_dump(), "current_step": "analyst"}

    except TimeoutError:
        error = ErrorDetail(
            error_type=ErrorType.TIMEOUT,
            message="Cortex Analyst timed out",
            node="analyst_agent",
            timestamp=datetime.now(UTC).isoformat(),
            retriable=True,
            fallback_action="retry_with_backoff",
        )
        return {
            "current_step": "error",
            "last_error": error.to_dict(),
            "error_count": state.get("error_count", 0) + 1,
            "has_error": True,
        }
    except Exception as exc:
        logger.error(f"Analyst agent error: {exc}", exc_info=True)
        error = ErrorDetail(
            error_type=ErrorType.CORTEX_API,
            message=str(exc),
            node="analyst_agent",
            timestamp=datetime.now(UTC).isoformat(),
            retriable=False,
            fallback_action="escalate",
        )
        return {
            "current_step": "error",
            "last_error": error.to_dict(),
            "error_count": state.get("error_count", 0) + 1,
            "has_error": True,
        }


async def async_search_agent(state: HealthcareAgentState) -> AgentOutput:
    """Execute Cortex Search for knowledge queries.

    Searches across FAQs, policies, and call transcripts
    using real Snowflake Cortex Search when session is available.

    Args:
        state: Current workflow state with plan and user_query.

    Returns:
        AgentOutput with search_results or error information.
    """
    plan = state.get("plan", "")
    if plan not in ["search", "both"]:
        return {}

    query = state.get("user_query", "")

    try:
        # Use real Cortex Search if session is available
        session = get_snowpark_session()
        if session is not None:
            search_tool = AsyncCortexSearchTool(session)
            results = await search_tool.execute(query, limit=5)
            logger.info(f"Search executed real query with {len(results)} results")
        else:
            # Fallback to mock for testing/development
            logger.warning("No Snowpark session - using mock search results")
            results = [
                {
                    "text": f"Policy content relevant to: {query[:50]}...",
                    "score": 0.95,
                    "source": "policies",
                },
                {
                    "text": "FAQ: How to file an appeal...",
                    "score": 0.88,
                    "source": "faqs",
                },
            ]

        # Validate with Pydantic models
        validated = [SearchResultModel(**r) for r in results]
        return {
            "search_results": [r.model_dump() for r in validated],
            "current_step": "search",
        }

    except TimeoutError:
        error = ErrorDetail(
            error_type=ErrorType.TIMEOUT,
            message="Cortex Search timed out",
            node="search_agent",
            timestamp=datetime.now(UTC).isoformat(),
            retriable=True,
            fallback_action="retry_with_backoff",
        )
        return {
            "current_step": "error",
            "last_error": error.to_dict(),
            "error_count": state.get("error_count", 0) + 1,
            "has_error": True,
        }
    except Exception as exc:
        logger.error(f"Search agent error: {exc}", exc_info=True)
        error = ErrorDetail(
            error_type=ErrorType.CORTEX_API,
            message=str(exc),
            node="search_agent",
            timestamp=datetime.now(UTC).isoformat(),
            retriable=False,
            fallback_action="escalate",
        )
        return {
            "current_step": "error",
            "last_error": error.to_dict(),
            "error_count": state.get("error_count", 0) + 1,
            "has_error": True,
        }


async def parallel_agent_execution(state: HealthcareAgentState) -> ParallelOutput:
    """Execute Analyst and Search in parallel using asyncio.TaskGroup.

    Used when planner routes to "both" - executes both agents
    concurrently and merges results.

    Args:
        state: Current workflow state with plan set to "both".

    Returns:
        ParallelOutput with merged analyst_results and search_results.
    """
    if state.get("plan") != "both":
        return {}

    analyst_result: dict = {}
    search_result: dict = {}
    errors: list[ErrorDetail] = []

    try:
        async with asyncio.TaskGroup() as tg:
            analyst_task = tg.create_task(
                asyncio.wait_for(
                    async_analyst_agent(state),
                    timeout=settings.agent_timeout_seconds,
                )
            )
            search_task = tg.create_task(
                asyncio.wait_for(
                    async_search_agent(state),
                    timeout=settings.agent_timeout_seconds,
                )
            )
        analyst_result = analyst_task.result()
        search_result = search_task.result()

    except* TimeoutError as eg:
        for exc in eg.exceptions:
            errors.append(
                ErrorDetail(
                    error_type=ErrorType.TIMEOUT,
                    message=f"Agent execution timed out: {exc}",
                    node="parallel_execution",
                    timestamp=datetime.now(UTC).isoformat(),
                    retriable=True,
                    fallback_action="retry_with_backoff",
                )
            )
    except* Exception as eg:
        for exc in eg.exceptions:
            errors.append(
                ErrorDetail(
                    error_type=ErrorType.CORTEX_API,
                    message=str(exc),
                    node="parallel_execution",
                    timestamp=datetime.now(UTC).isoformat(),
                    retriable=False,
                    fallback_action="escalate",
                )
            )

    # Merge results
    merged = {
        **analyst_result,
        **search_result,
        "current_step": "parallel",
        "error_count": state.get("error_count", 0) + len(errors),
        "has_error": len(errors) > 0,
    }
    if errors:
        merged["last_error"] = errors[-1].to_dict()

    return merged


async def async_response_agent(state: HealthcareAgentState) -> ResponseOutput:
    """Synthesize final response using Cortex LLM.

    Uses Cortex COMPLETE to generate contextual responses from
    analyst and search results.

    Args:
        state: Current workflow state with analyst_results and/or search_results.

    Returns:
        ResponseOutput with final_response and is_complete flag.
    """
    analyst_results = state.get("analyst_results", {})
    search_results = state.get("search_results", [])
    query = state.get("user_query", "")

    # Build context for LLM synthesis
    context_parts = []

    if analyst_results:
        member_id = analyst_results.get("member_id", "Unknown")
        raw_response = analyst_results.get("raw_response", "")
        coverage = analyst_results.get("coverage", {})
        claims = analyst_results.get("claims", [])

        context_parts.append(f"Member ID: {member_id}")
        if raw_response:
            context_parts.append(f"Member Data: {raw_response}")
        if coverage:
            context_parts.append(f"Coverage: {coverage}")
        if claims:
            context_parts.append(f"Recent Claims: {claims[:3]}")

    if search_results:
        top_results = search_results[:3]
        search_context = "\n".join([r.get("text", "")[:500] for r in top_results])
        context_parts.append(f"Knowledge Base Results:\n{search_context}")

    context = "\n\n".join(context_parts)

    # Use Cortex COMPLETE to generate contextual response
    session = get_snowpark_session()
    if session is not None and context:
        try:
            final_response = await _synthesize_with_cortex(session, query, context)
            logger.info("Response synthesized with Cortex LLM")
        except Exception as exc:
            logger.warning(f"Cortex synthesis failed, using fallback: {exc}")
            final_response = _fallback_response(query, analyst_results, search_results)
    else:
        final_response = _fallback_response(query, analyst_results, search_results)

    logger.info("Response synthesis complete")
    return {
        "final_response": final_response,
        "is_complete": True,
        "current_step": "response",
    }


async def _synthesize_with_cortex(session: Session, query: str, context: str) -> str:
    """Use Cortex COMPLETE to synthesize a response.

    Args:
        session: Active Snowpark session for Cortex calls.
        query: User's original query.
        context: Accumulated context from analyst/search results.

    Returns:
        Synthesized response from Cortex LLM.
    """

    def _call_cortex() -> str:
        escaped_query = query.replace("'", "''")
        escaped_context = context.replace("'", "''")[:8000]

        prompt = f"""You are a helpful healthcare contact center assistant.
Answer the user's question based ONLY on the provided context.
Be concise and accurate. Extract the specific information requested.

User Question: {escaped_query}

Context:
{escaped_context}

Answer:"""

        sql = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            'llama3.1-70b',
            '{prompt.replace("'", "''")}'
        ) AS response
        """
        result = session.sql(sql).collect()
        return str(result[0][0]) if result else "I couldn't generate a response."

    return await asyncio.to_thread(_call_cortex)


def _fallback_response(query: str, analyst_results: dict, search_results: list) -> str:
    """Generate fallback response when Cortex LLM is unavailable."""
    response_parts = []

    if analyst_results:
        member_id = analyst_results.get("member_id")
        raw_response = analyst_results.get("raw_response", "")

        if raw_response and "NAME" in raw_response:
            try:
                import ast

                member_data = ast.literal_eval(raw_response)
                if member_data and isinstance(member_data, list) and len(member_data) > 0:
                    m = member_data[0]
                    name = m.get("NAME", "Unknown")
                    plan = m.get("PLAN_NAME", "N/A")
                    pcp = m.get("PCP", "N/A")
                    response_parts.append(f"Member {member_id}: {name}, Plan: {plan}, PCP: {pcp}")
            except (SyntaxError, ValueError):
                response_parts.append(f"Member ID: {member_id}")
        elif member_id:
            response_parts.append(f"Member ID: {member_id}")

        coverage = analyst_results.get("coverage", {})
        if coverage:
            # Handle various field name formats (from different data sources)
            plan_name = coverage.get(
                "PLAN_NAME", coverage.get("plan_name", coverage.get("plan", "N/A"))
            )
            plan_type = coverage.get("PLAN_TYPE", coverage.get("plan_type", ""))
            deductible = coverage.get("DEDUCTIBLE", coverage.get("deductible", ""))
            copay = coverage.get("COPAY_OFFICE", coverage.get("copay_office", ""))

            if plan_name != "N/A":
                plan_info = f"Plan: {plan_name}"
                if plan_type:
                    plan_info += f" ({plan_type})"
                if deductible:
                    plan_info += f", Deductible: ${deductible}"
                if copay:
                    plan_info += f", Office Copay: ${copay}"
                response_parts.append(plan_info)

    if search_results:
        top_result = search_results[0] if search_results else {}
        response_parts.append(
            f"From knowledge base ({len(search_results)} results): "
            f"{top_result.get('text', 'No content')[:300]}..."
        )

    return " | ".join(response_parts) if response_parts else f"No info found for: {query}"


async def error_handler_node(state: HealthcareAgentState) -> ErrorOutput:
    """Handle errors with retry/escalate logic.

    Examines last_error to determine if operation should be
    retried (retriable=True) or escalated to human.

    Args:
        state: Current workflow state with last_error and error_count.

    Returns:
        ErrorOutput with has_error cleared for retry or should_escalate set.
    """
    last_error = state.get("last_error", {})
    error_count = state.get("error_count", 0)

    if last_error.get("retriable", False) and error_count < settings.max_agent_steps:
        logger.info(f"Retrying after error (attempt {error_count})")
        return {"current_step": "retry", "has_error": False}

    logger.warning(f"Escalating error after {error_count} attempts")
    return {"current_step": "escalate", "should_escalate": True}


async def escalation_handler(state: HealthcareAgentState) -> EscalationOutput:  # noqa: ARG001
    """Handle escalated errors with graceful degradation.

    Args:
        state: Current workflow state (unused, error already captured).

    Returns:
        EscalationOutput with apology response and escalation flags.
    """
    return {
        "final_response": (
            "I apologize, but I encountered an error processing your request. "
            "Please try again or contact our support team for assistance."
        ),
        "is_complete": True,
        "should_escalate": True,
        "current_step": "escalated",
    }


# --- Routing Functions ---


def route_to_agents(state: HealthcareAgentState) -> Literal["analyst", "search", "parallel"]:
    """Route based on planner decision."""
    plan = state.get("plan", "search")
    if plan == "both":
        return "parallel"
    if plan == "analyst":
        return "analyst"
    return "search"


def should_continue_or_halt(
    state: HealthcareAgentState,
) -> Literal["continue", "error_handler", "halt"]:
    """Circuit breaker: determine if workflow should continue or halt."""
    error_count = state.get("error_count", 0)
    max_steps = state.get("max_steps", settings.max_agent_steps)
    has_error = state.get("has_error", False)

    if has_error:
        return "error_handler"
    if error_count >= max_steps:
        return "halt"
    return "continue"


def route_after_error(
    state: HealthcareAgentState,
) -> Literal["planner", "escalation"]:
    """Route after error handler based on should_escalate."""
    if state.get("should_escalate", False):
        return "escalation"
    return "planner"


# --- Graph Construction ---


def create_healthcare_graph() -> StateGraph:
    """Build the healthcare agent workflow graph.

    Graph topology:
        START -> planner -> (analyst | search | parallel) -> response -> END
                    |                                            |
                    +<--------- error_handler <------------------+
                                     |
                                     v
                              escalation_handler -> END
    """
    workflow = StateGraph(HealthcareAgentState)

    # Add nodes
    workflow.add_node("planner", async_planner_agent)
    workflow.add_node("analyst", async_analyst_agent)
    workflow.add_node("search", async_search_agent)
    workflow.add_node("parallel", parallel_agent_execution)
    workflow.add_node("response", async_response_agent)
    workflow.add_node("error_handler", error_handler_node)
    workflow.add_node("escalation_handler", escalation_handler)

    # Add edges
    workflow.add_edge(START, "planner")

    workflow.add_conditional_edges(
        "planner",
        route_to_agents,
        {
            "analyst": "analyst",
            "search": "search",
            "parallel": "parallel",
        },
    )

    # All agent paths lead to response
    workflow.add_edge("analyst", "response")
    workflow.add_edge("search", "response")
    workflow.add_edge("parallel", "response")

    # Response node with circuit breaker
    workflow.add_conditional_edges(
        "response",
        should_continue_or_halt,
        {
            "continue": END,
            "error_handler": "error_handler",
            "halt": "escalation_handler",
        },
    )

    # Error handler routing
    workflow.add_conditional_edges(
        "error_handler",
        route_after_error,
        {
            "planner": "planner",
            "escalation": "escalation_handler",
        },
    )

    workflow.add_edge("escalation_handler", END)

    return workflow


# Initialize Snowpark session at module load time for langgraph dev
def _initialize_snowpark_session() -> None:
    """Initialize Snowpark session from environment variables.

    Creates a Snowpark session using either:
    - SPCS OAuth token (if SNOWFLAKE_TOKEN is set)
    - Key-pair authentication (if snowflake_private_key_path is configured)

    Sets the session via contextvars for thread-safe access.
    """
    if get_snowpark_session() is not None:
        return  # Already initialized

    try:
        import os
        from pathlib import Path

        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import serialization

        from src.config import settings

        session: Session | None = None

        # Check if running in SPCS (OAuth token available)
        if os.environ.get("SNOWFLAKE_TOKEN"):
            logger.info("Creating Snowpark session with SPCS OAuth token")
            session = Session.builder.configs(
                {
                    "account": settings.snowflake_account,
                    "token": os.environ.get("SNOWFLAKE_TOKEN"),
                    "authenticator": "oauth",
                    "database": settings.snowflake_database,
                    "schema": settings.snowflake_schema,
                    "warehouse": settings.snowflake_warehouse,
                }
            ).create()
        elif settings.snowflake_private_key_path:
            logger.info("Creating Snowpark session with key-pair authentication")

            # Load private key from file
            key_path = Path(settings.snowflake_private_key_path).expanduser()
            with open(key_path, "rb") as key_file:
                passphrase = settings.snowflake_private_key_passphrase
                p_key = serialization.load_pem_private_key(
                    key_file.read(),
                    password=passphrase.encode() if passphrase else None,
                    backend=default_backend(),
                )

            # Serialize to DER format for Snowpark
            pkb = p_key.private_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )

            session = Session.builder.configs(
                {
                    "account": settings.snowflake_account,
                    "user": settings.snowflake_user,
                    "private_key": pkb,
                    "database": settings.snowflake_database,
                    "schema": settings.snowflake_schema,
                    "warehouse": settings.snowflake_warehouse,
                    "role": settings.snowflake_role,
                }
            ).create()
        else:
            logger.warning("No Snowflake credentials configured - using mock mode")
            return

        # Set session via contextvars
        set_snowpark_session(session)
        logger.info("Snowpark session initialized successfully")

    except Exception as exc:
        logger.warning(f"Failed to initialize Snowpark session: {exc}")


# Initialize session on module load
_initialize_snowpark_session()

# Export compiled graph for langgraph.json
healthcare_graph = create_healthcare_graph().compile()
