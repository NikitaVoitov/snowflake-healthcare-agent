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

from src.config import settings
from src.graphs.state import HealthcareAgentState
from src.models.agent_types import (
    AnalystResultModel,
    ErrorDetail,
    ErrorType,
    SearchResultModel,
)

logger = logging.getLogger(__name__)

# --- Agent Functions ---


async def async_planner_agent(state: HealthcareAgentState) -> dict:
    """Route query to appropriate agent(s) based on intent.

    Analyzes query to determine if it needs:
    - analyst: Member-specific data (claims, coverage, demographics)
    - search: Knowledge base lookup (policies, FAQs, transcripts)
    - both: Combined member data + policy context
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


async def async_analyst_agent(state: HealthcareAgentState) -> dict:
    """Execute Cortex Analyst for member queries.

    Queries member data from MEMBER_SCHEMA tables using
    Cortex Analyst for natural language to SQL translation.
    """
    plan = state.get("plan", "")
    if plan not in ["analyst", "both"]:
        return {}

    try:
        # Mock implementation - in production, use AsyncCortexAnalystTool
        member_id = state.get("member_id")
        results = {
            "member_id": member_id,
            "claims": [],
            "coverage": {"plan": "Gold", "deductible": 500, "copay_office": 25},
        }

        # Validate with Pydantic model
        validated = AnalystResultModel(**results)
        logger.info(f"Analyst completed for member: {member_id}")
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


async def async_search_agent(state: HealthcareAgentState) -> dict:
    """Execute Cortex Search for knowledge queries.

    Searches across FAQs, policies, and call transcripts
    using Cortex Search for semantic similarity.
    """
    plan = state.get("plan", "")
    if plan not in ["search", "both"]:
        return {}

    try:
        # Mock implementation - in production, use AsyncCortexSearchTool
        query = state.get("user_query", "")
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
        logger.info(f"Search completed with {len(validated)} results")
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


async def parallel_agent_execution(state: HealthcareAgentState) -> dict:
    """Execute Analyst and Search in parallel using asyncio.TaskGroup.

    Used when planner routes to "both" - executes both agents
    concurrently and merges results.
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


async def async_response_agent(state: HealthcareAgentState) -> dict:
    """Synthesize final response from agent results.

    Combines analyst and search results into a coherent
    response using Cortex LLM for natural language generation.
    """
    analyst_results = state.get("analyst_results", {})
    search_results = state.get("search_results", [])
    query = state.get("user_query", "")

    # Build response (in production, use Cortex COMPLETE for synthesis)
    response_parts = []

    if analyst_results:
        member_id = analyst_results.get("member_id", "Unknown")
        coverage = analyst_results.get("coverage", {})
        response_parts.append(
            f"Based on your member information ({member_id}), "
            f"your plan has a ${coverage.get('deductible', 'N/A')} deductible "
            f"and ${coverage.get('copay_office', 'N/A')} office copay."
        )

    if search_results:
        top_result = search_results[0] if search_results else {}
        response_parts.append(
            f"From our knowledge base ({len(search_results)} relevant documents found): "
            f"{top_result.get('text', 'No content')[:200]}..."
        )

    final_response = (
        " ".join(response_parts)
        or f"I couldn't find specific information for: {query}"
    )

    logger.info("Response synthesis complete")
    return {
        "final_response": final_response,
        "is_complete": True,
        "current_step": "response",
    }


async def error_handler_node(state: HealthcareAgentState) -> dict:
    """Handle errors with retry/escalate logic.

    Examines last_error to determine if operation should be
    retried (retriable=True) or escalated to human.
    """
    last_error = state.get("last_error", {})
    error_count = state.get("error_count", 0)

    if last_error.get("retriable", False) and error_count < settings.max_agent_steps:
        logger.info(f"Retrying after error (attempt {error_count})")
        return {"current_step": "retry", "has_error": False}

    logger.warning(f"Escalating error after {error_count} attempts")
    return {"current_step": "escalate", "should_escalate": True}


async def escalation_handler(state: HealthcareAgentState) -> dict:  # noqa: ARG001
    """Handle escalated errors with graceful degradation."""
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


# Export compiled graph for langgraph.json
healthcare_graph = create_healthcare_graph().compile()

