"""ReAct state definitions for Healthcare Agent workflow.

Uses LangGraph's message-based state pattern for compatibility with
ToolNode and modern agent patterns.

The messages field accumulates the conversation within a single turn:
- HumanMessage: User query
- AIMessage with tool_calls: Model decides to call a tool
- ToolMessage: Tool execution result
- AIMessage with content: Final answer

Conversation history persists across turns via the Snowflake checkpointer.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


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


class HealthcareAgentState(TypedDict, total=False):
    """Message-based state for healthcare agent workflow.

    Uses LangGraph's add_messages reducer for the messages field,
    enabling automatic handling of message deduplication and updates.

    The messages field replaces the old scratchpad pattern:
    - Old: scratchpad with ReActStep dicts
    - New: messages with HumanMessage, AIMessage (tool_calls), ToolMessage
    """

    # ==========================================================================
    # Core Message Stream (LangGraph standard pattern)
    # ==========================================================================
    # Messages accumulate via add_messages reducer:
    # [HumanMessage] → [HumanMessage, AIMessage(tool_calls)] →
    # [HumanMessage, AIMessage(tool_calls), ToolMessage] → ...
    messages: Annotated[list[AnyMessage], add_messages]

    # ==========================================================================
    # Healthcare Context (custom fields)
    # ==========================================================================
    user_query: str  # Original query (kept for reference)
    member_id: str | None  # Member context from session
    tenant_id: str  # Multi-tenant identifier

    # ==========================================================================
    # Conversation History (persisted across turns via checkpointer)
    # ==========================================================================
    conversation_history: Annotated[list[ConversationTurn], _append_turns]

    # ==========================================================================
    # Execution Control
    # ==========================================================================
    iteration: int  # Current iteration count in ReAct loop
    max_iterations: int  # Safety limit (default: 5)
    execution_id: str  # Unique ID for this execution

    # ==========================================================================
    # Output
    # ==========================================================================
    final_answer: str | None  # Final response to user


# =============================================================================
# Typed Output Dictionaries for Node Return Values
# =============================================================================


class ModelOutput(TypedDict, total=False):
    """Output from model node (call_model).

    Returns AIMessage which gets added to messages via reducer.
    """

    messages: list[AnyMessage]  # Single AIMessage to append
    iteration: int


class FinalAnswerOutput(TypedDict, total=False):
    """Output when agent produces final answer.

    Also appends the completed turn to conversation_history for persistence.
    """

    final_answer: str
    conversation_history: list[ConversationTurn]
