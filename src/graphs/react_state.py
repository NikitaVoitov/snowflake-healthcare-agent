"""ReAct state definitions for Healthcare Agent workflow.

Updated for LangChain v1 / LangGraph v1:
- Extends `AgentState` from `langchain.agents` (new in v1)
- `messages` field is inherited from AgentState with `add_messages` reducer
- Custom healthcare fields added for member context and conversation history

The messages field accumulates the conversation within a single turn:
- HumanMessage: User query
- AIMessage with tool_calls: Model decides to call a tool
- ToolMessage: Tool execution result
- AIMessage with content: Final answer

Conversation history persists across turns via the Snowflake checkpointer.
"""

from typing import Annotated, TypedDict

from langchain.agents import AgentState
from langchain_core.messages import AnyMessage


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


class HealthcareAgentState(AgentState, total=False):
    """Message-based state for healthcare agent workflow.

    Extends LangChain v1's AgentState which provides:
    - messages: Annotated[list[AnyMessage], add_messages]  # Core message stream
    - jump_to: JumpTo | None  # Internal control flow
    - structured_response: ResponseT | None  # Structured output

    The `messages` field is inherited and uses add_messages reducer for
    automatic handling of message deduplication and updates.

    Custom healthcare fields extend the base state for:
    - Member context (member_id, tenant_id)
    - Conversation history for multi-turn context
    - Execution control (iteration limits, execution tracking)
    """

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
