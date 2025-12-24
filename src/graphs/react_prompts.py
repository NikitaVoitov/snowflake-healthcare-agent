"""System prompts for Healthcare Agent with ChatSnowflake.

Contains system message templates for the healthcare agent. With ChatSnowflake
and bind_tools(), tool schemas are automatically included - we focus on:
- Domain context (healthcare contact center)
- Tool selection guidance (when to use each tool)
- Conversation history handling
"""

from src.graphs.react_state import ConversationTurn

# System message for ChatSnowflake (tool schemas added automatically via bind_tools)
HEALTHCARE_SYSTEM_PROMPT = """You are a healthcare contact center assistant helping members with their insurance questions.

## Your Capabilities

You have access to tools for:
1. **Member Database** (query_member_data) - Query member information, claims, coverage, grievances
2. **Knowledge Base** (search_knowledge) - Search FAQs, policies, and call transcripts

## Tool Selection Rules

**Use query_member_data for:**
- Member personal info: name, DOB, gender, address, phone
- Plan/coverage details: plan name, plan type, premium, PCP
- Claims history: amounts, status, providers, dates
- Grievances: type, status, dates
- Aggregate queries: "How many members?", "Top 3 highest premiums"

**Use search_knowledge for:**
- Call history/transcripts: "What did they call about?", "Previous calls"
- Policy questions: "What is the deductible policy?"
- Process questions: "How do I submit a claim?"
- FAQ lookups: "What does copay mean?"

**IMPORTANT:**
- Call history is in search_knowledge, NOT query_member_data
- Pass the EXACT user question to query_member_data - don't paraphrase
- Stop searching when you find relevant results
- Use conversation history to resolve pronouns ("they", "their", "the member")

## Answer Style
- Be DIRECT - present data you have
- Include specific numbers, dates, amounts
- Format clearly with sections when appropriate
- If you have enough information, provide the final answer"""


def format_conversation_history(history: list[ConversationTurn]) -> str:
    """Format conversation history for inclusion in the system message.

    Provides context from previous turns so the agent can understand
    follow-up questions like "What plan are they on?" after discussing a member.

    Args:
        history: List of previous conversation turns.

    Returns:
        Formatted string showing conversation context.
    """
    if not history:
        return ""

    lines = ["\n## Recent Conversation"]
    for i, turn in enumerate(history[-5:], 1):  # Show last 5 turns for context
        member_info = f" (Member: {turn['member_id']})" if turn.get("member_id") else ""
        lines.append(f"\n**Turn {i}**{member_info}")
        lines.append(f"- User: {turn['user_query']}")
        # Truncate long answers for prompt efficiency
        answer = turn["final_answer"]
        if len(answer) > 200:
            answer = answer[:200] + "..."
        lines.append(f"- Assistant: {answer}")

    return "\n".join(lines)


def extract_member_id_from_history(history: list[ConversationTurn]) -> str | None:
    """Extract the most recent member_id from conversation history.

    Useful for follow-up questions where user refers to "they" or "the member".

    Args:
        history: Conversation history.

    Returns:
        Most recent member_id or None.
    """
    if not history:
        return None

    # Search from most recent to oldest
    for turn in reversed(history):
        if turn.get("member_id"):
            return turn["member_id"]
    return None


def build_system_message(
    member_id: str | None = None,
    conversation_history: list[ConversationTurn] | None = None,
) -> str:
    """Build the system message for ChatSnowflake.

    Combines:
    - Base healthcare assistant prompt
    - Current member context (if any)
    - Conversation history (if any)

    Args:
        member_id: Optional member ID for current context.
        conversation_history: Previous conversation turns for context.

    Returns:
        Complete system message string.
    """
    parts = [HEALTHCARE_SYSTEM_PROMPT]

    # Add member context
    if member_id:
        parts.append(f"\n## Current Member Context\nMember ID: {member_id}")
    elif conversation_history:
        # Try to get member_id from history for follow-up questions
        historical_member_id = extract_member_id_from_history(conversation_history)
        if historical_member_id:
            parts.append(f"\n## Current Member Context\nMember ID: {historical_member_id} (from previous conversation)")

    # Add conversation history
    history_text = format_conversation_history(conversation_history or [])
    if history_text:
        parts.append(history_text)

    return "\n".join(parts)
