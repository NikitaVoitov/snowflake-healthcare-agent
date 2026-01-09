"""Healthcare agent middleware for LangChain v1 create_agent.

This module implements middleware for the healthcare agent using LangChain v1's
@dynamic_prompt decorator. It provides dynamic system prompt generation based on:
- Member ID context (current or from conversation history)
- Conversation history (for follow-up questions)
- Runtime context (tenant_id, execution_id, etc.)

The middleware replaces the manual system message building from the old
workflow-based approach with proper LangChain v1 middleware patterns.
"""

from typing import Any

from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_core.messages import SystemMessage

from src.graphs.react_prompts import (
    HEALTHCARE_SYSTEM_PROMPT,
    extract_member_id_from_history,
    format_conversation_history,
)
from src.graphs.react_state import ConversationTurn


@dynamic_prompt
def healthcare_prompt(request: ModelRequest) -> str:
    """Generate dynamic healthcare system prompt based on agent state and context.

    This middleware runs before each model invocation to build a contextual
    system prompt that includes:
    1. Base healthcare assistant instructions (tool usage, answer style)
    2. Current member context (if provided or from history)
    3. Recent conversation history (last 5 turns for context)

    The prompt enables the LLM to:
    - Use the correct tool for each query type
    - Resolve pronouns ("they", "the member") from history
    - Maintain conversation continuity across turns

    Args:
        request: ModelRequest containing:
            - state: HealthcareAgentState with messages, member_id, history
            - runtime: Runtime context with custom context dict

    Returns:
        Complete system prompt string for ChatSnowflake.

    Example:
        ```python
        from langchain.agents import create_agent

        agent = create_agent(
            model=llm,
            tools=tools,
            middleware=[healthcare_prompt],
        )
        ```
    """
    state = request.state or {}
    # Handle None runtime or None context
    runtime_context = (getattr(request.runtime, 'context', None) or {}) if request.runtime else {}

    # Extract healthcare-specific fields from state
    member_id: str | None = state.get("member_id")
    conversation_history: list[ConversationTurn] = state.get("conversation_history", [])

    # Build the system prompt
    parts = [HEALTHCARE_SYSTEM_PROMPT]

    # Add member context
    if member_id:
        parts.append(f"\n## Current Member Context\nMember ID: {member_id}")
    elif conversation_history:
        # Try to get member_id from history for follow-up questions
        historical_member_id = extract_member_id_from_history(conversation_history)
        if historical_member_id:
            parts.append(
                f"\n## Current Member Context\n"
                f"Member ID: {historical_member_id} (from previous conversation)"
            )

    # Add conversation history
    history_text = format_conversation_history(conversation_history)
    if history_text:
        parts.append(history_text)

    # Add runtime context if available (tenant_id, execution_id for tracing)
    tenant_id = runtime_context.get("tenant_id") or state.get("tenant_id")
    execution_id = runtime_context.get("execution_id") or state.get("execution_id")

    if tenant_id or execution_id:
        context_parts = []
        if tenant_id:
            context_parts.append(f"Tenant: {tenant_id}")
        if execution_id:
            context_parts.append(f"Execution: {execution_id}")
        # This is internal context for tracing, not shown to user
        # Could be used for logging/debugging if needed

    return "\n".join(parts)


def create_healthcare_middleware() -> list[Any]:
    """Create the list of middleware for the healthcare agent.

    Returns a list containing:
    1. healthcare_prompt - Dynamic system prompt generation

    Additional middleware can be added here in the future:
    - SummarizationMiddleware for long conversations
    - RateLimitingMiddleware for API protection
    - CachingMiddleware for response caching

    Returns:
        List of AgentMiddleware instances for create_agent.

    Example:
        ```python
        from langchain.agents import create_agent

        agent = create_agent(
            model=llm,
            tools=tools,
            middleware=create_healthcare_middleware(),
        )
        ```
    """
    return [healthcare_prompt]


# Export for easy import
__all__ = ["healthcare_prompt", "create_healthcare_middleware"]

