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

