"""LangGraph state definition for Healthcare Multi-Agent workflow."""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class HealthcareAgentState(TypedDict, total=False):
    """Healthcare agent workflow state with typed fields.

    Uses TypedDict for LangGraph compatibility with optional fields.
    Reducers are applied via Annotated types for automatic state updates.
    """

    # Input fields
    user_query: str
    member_id: str | None

    # Messages with reducer for accumulation
    messages: Annotated[list, add_messages]

    # Planning
    plan: str  # analyst | search | both

    # Results from agents
    analyst_results: dict | None
    search_results: list | None
    final_response: str

    # Execution tracking (auxiliary verbs per coding standards)
    execution_id: str
    thread_checkpoint_id: str | None
    current_step: str

    # Circuit breaker (auxiliary verbs)
    error_count: int
    max_steps: int
    last_error: dict | None
    has_error: bool
    is_complete: bool
    should_escalate: bool


# =============================================================================
# Typed Output Dictionaries for Node Return Values
# =============================================================================
# These provide strict type hints for partial state updates returned by nodes.


class PlannerOutput(TypedDict, total=False):
    """Output from planner node with routing decision."""

    plan: str
    current_step: str


class AgentOutput(TypedDict, total=False):
    """Output from analyst or search agent nodes."""

    analyst_results: dict | None
    search_results: list | None
    current_step: str
    last_error: dict | None
    error_count: int
    has_error: bool


class ParallelOutput(TypedDict, total=False):
    """Output from parallel execution node (combined agent results)."""

    analyst_results: dict | None
    search_results: list | None
    current_step: str
    last_error: dict | None
    error_count: int
    has_error: bool


class ResponseOutput(TypedDict, total=False):
    """Output from response synthesis node."""

    final_response: str
    is_complete: bool
    current_step: str


class ErrorOutput(TypedDict, total=False):
    """Output from error handler node."""

    current_step: str
    has_error: bool
    should_escalate: bool


class EscalationOutput(TypedDict, total=False):
    """Output from escalation handler node."""

    final_response: str
    is_complete: bool
    should_escalate: bool
    current_step: str

