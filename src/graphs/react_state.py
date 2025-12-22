"""ReAct state definitions for Healthcare Agent workflow.

Implements the Reasoning + Acting pattern with iterative tool execution.
The agent alternates between thinking, acting, and observing until
it can deliver a final answer.

Conversation history is persisted via the Snowflake checkpointer,
enabling context continuity across turns in SPCS deployments.
"""

from typing import Annotated, TypedDict


class ReActStep(TypedDict):
    """A single step in the ReAct loop.

    Each step captures the model's reasoning, its decision to act,
    the action parameters, and the resulting observation.
    """

    thought: str  # Model's reasoning about what to do
    action: str  # Tool name or "FINAL_ANSWER"
    action_input: dict  # JSON arguments for the tool
    observation: str  # Tool result (filled after execution)


class ConversationTurn(TypedDict):
    """A single turn in the conversation history.

    Captures the user's query and the agent's final answer for context.
    """

    user_query: str
    member_id: str | None  # Member context if provided
    final_answer: str
    tools_used: list[str]  # Tools used in this turn (for context)


def _append_turns(existing: list[ConversationTurn], new: list[ConversationTurn]) -> list[ConversationTurn]:
    """Reducer to append new conversation turns to existing history.

    LangGraph uses this to merge state updates. We keep a sliding window
    of the last N turns to prevent unbounded growth.
    """
    max_turns = 10  # Keep last 10 turns for context
    combined = (existing or []) + (new or [])
    return combined[-max_turns:]


class HealthcareReActState(TypedDict, total=False):
    """State for ReAct-based healthcare agent workflow.

    The scratchpad accumulates all reasoning steps within a single turn.
    The conversation_history persists across turns via the checkpointer,
    enabling the agent to maintain context in multi-turn conversations.
    """

    # Input fields (current turn)
    user_query: str
    member_id: str | None
    tenant_id: str

    # Conversation History (persisted across turns via checkpointer)
    # Uses Annotated with reducer to properly accumulate turns
    conversation_history: Annotated[list[ConversationTurn], _append_turns]

    # ReAct Loop State (current turn only)
    scratchpad: list[ReActStep]  # History of thought/action/observation
    current_step: ReActStep | None  # Active step being processed
    iteration: int  # Current iteration count
    max_iterations: int  # Safety limit (default: 5)

    # Tool execution
    tool_result: str | None  # Result from last tool execution

    # Output
    final_answer: str | None

    # Execution tracking
    execution_id: str
    thread_checkpoint_id: str | None
    current_node: str  # Track which node we're in

    # Error handling
    error_count: int
    last_error: dict | None
    has_error: bool


# =============================================================================
# Typed Output Dictionaries for Node Return Values
# =============================================================================


class ReasonerOutput(TypedDict, total=False):
    """Output from reasoner node."""

    current_step: ReActStep | None
    iteration: int
    has_error: bool
    last_error: dict | None


class ToolOutput(TypedDict, total=False):
    """Output from tool execution nodes."""

    tool_result: str
    has_error: bool
    last_error: dict | None
    error_count: int


class ObservationOutput(TypedDict, total=False):
    """Output from observation node."""

    scratchpad: list[ReActStep]
    current_step: None  # Always None after observation
    tool_result: None  # Cleared after capture


class FinalAnswerOutput(TypedDict, total=False):
    """Output when agent produces final answer.

    Also appends the completed turn to conversation_history for persistence.
    """

    final_answer: str
    current_node: str
    conversation_history: list[ConversationTurn]  # Append completed turn


class ErrorHandlerOutput(TypedDict, total=False):
    """Output from error handler node."""

    has_error: bool
    error_count: int
    last_error: dict | None
    current_node: str

