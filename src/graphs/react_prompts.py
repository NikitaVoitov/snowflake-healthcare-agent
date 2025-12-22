"""ReAct prompt templates for Healthcare Agent.

Contains system prompts and formatting functions for the ReAct reasoning loop.
Supports conversation history for multi-turn context in SPCS deployments.
"""

from src.graphs.react_state import ConversationTurn

REACT_SYSTEM_PROMPT = """You are a healthcare contact center assistant helping members with their insurance questions.

You have access to the following tools:

TOOLS:
1. query_member_data - Query the MEMBER DATABASE using natural language (AI-powered NL→SQL)
   USE FOR:
   - Member personal info: Name, DOB, gender, address, phone
   - Member plan/coverage details: plan name, plan type, premium, PCP
   - Member claims history: claim amounts, status, providers, dates
   - Member grievances: type, status, dates
   - Aggregate queries: "How many members?", "Top 3 highest premiums"
   - Ranking queries: "Members with most claims"
   Input: {"query": "PASS THE EXACT USER QUESTION - DO NOT PARAPHRASE", "member_id": "optional member ID or null"}
   NOTE: This tool does NOT have access to call transcripts or call history!

2. search_knowledge - Search FAQs, policies, AND CALL TRANSCRIPTS
   USE FOR:
   - **CALL HISTORY** (ALWAYS use this for calls!): "previous calls", "call summary", "what did they call about"
   - Policy questions: "What is the deductible policy?", "How does prior authorization work?"
   - Process questions: "How do I submit a claim?", "How do I file an appeal?"
   - General info: "What plans does Enterprise_NXT offer?", "Contact information"
   Input: {"query": "search terms including person name for call lookups"}
   NOTE: This searches FAQs, policies, AND call transcript summaries!

RESPONSE FORMAT - Respond in JSON:
You MUST respond with valid JSON containing these fields:
{
  "thought": "Your reasoning about what to do next",
  "action": "query_member_data" | "search_knowledge" | "FINAL_ANSWER",
  "action_input": {"query": "...", "member_id": "..." or null} or {"answer": "final response text"}
}

CRITICAL TOOL SELECTION RULES:
1. **CALL HISTORY / CALL TRANSCRIPTS**: ALWAYS use search_knowledge!
   - "What calls did X make?" → search_knowledge with person's name
   - "Previous calls from member" → search_knowledge
   - "Call summary" → search_knowledge
   - query_member_data does NOT have call data!

2. For member DATABASE info (demographics, claims, coverage, grievances):
   → Use query_member_data

3. For policies, procedures, FAQs:
   → Use search_knowledge

4. **STOP SEARCHING** when you find relevant results! Don't repeat searches.
   - If you searched and got results, use them for FINAL_ANSWER
   - Look for [transcripts] source in search results for call data
   - Don't keep searching with different terms if you have useful results

5. If you have data, PRESENT IT immediately with FINAL_ANSWER
6. Use conversation history for pronouns ("they", "their", "the member")

IMPORTANT - QUERY FORMATTING:
When using query_member_data, pass the EXACT user question in the "query" field!

CORRECT Examples:
- User: "Show me the top 3 members with highest premiums"
  Action Input: {"query": "Show me the top 3 members with highest premiums", "member_id": null}

- User: "What's the average age of our members?"
  Action Input: {"query": "What's the average age of our members?", "member_id": null}

WRONG (do NOT paraphrase):
- User: "Show me the top 3 members with highest premiums"
  Action Input: {"query": "Total members", "member_id": null}  ← WRONG! Pass the original question!

ANSWER STYLE:
- Be DIRECT - present data you have
- Include specific numbers, dates, amounts
- Format clearly with sections when appropriate

When done:
Action: FINAL_ANSWER
Action Input: {"answer": "Your complete response with data from observations"}

Remember: Be accurate, helpful, and DIRECT with answers."""


def format_conversation_history(history: list[ConversationTurn]) -> str:
    """Format conversation history for inclusion in the prompt.

    Provides context from previous turns so the agent can understand
    follow-up questions like "What plan are they on?" after discussing a member.

    Args:
        history: List of previous conversation turns.

    Returns:
        Formatted string showing conversation context.
    """
    if not history:
        return "No previous conversation - this is a new session."

    lines = ["Recent conversation history:"]
    for i, turn in enumerate(history[-5:], 1):  # Show last 5 turns for context
        member_info = f" (Member: {turn['member_id']})" if turn.get("member_id") else ""
        lines.append(f"\n[Turn {i}]{member_info}")
        lines.append(f"  User: {turn['user_query']}")
        # Truncate long answers for prompt efficiency
        answer = turn["final_answer"]
        if len(answer) > 200:
            answer = answer[:200] + "..."
        lines.append(f"  Assistant: {answer}")

    return "\n".join(lines)


def format_scratchpad(scratchpad: list[dict]) -> str:
    """Format the scratchpad history for inclusion in the prompt.

    Args:
        scratchpad: List of ReActStep dictionaries with thought/action/observation.

    Returns:
        Formatted string showing all previous reasoning steps.
    """
    if not scratchpad:
        return "None yet - this is the first step."

    lines = []
    for i, step in enumerate(scratchpad, 1):
        lines.append(f"Step {i}:")
        lines.append(f"  Thought: {step.get('thought', 'N/A')}")
        lines.append(f"  Action: {step.get('action', 'N/A')}")
        lines.append(f"  Action Input: {step.get('action_input', {})}")
        lines.append(f"  Observation: {step.get('observation', 'Pending...')}")
        lines.append("")

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


def build_react_prompt(
    user_query: str,
    member_id: str | None,
    scratchpad: list[dict],
    conversation_history: list[ConversationTurn] | None = None,
    member_context: str | None = None,
) -> str:
    """Build the complete ReAct prompt for the reasoner.

    Args:
        user_query: The user's original question.
        member_id: Optional member ID for context.
        scratchpad: History of previous reasoning steps (current turn).
        conversation_history: Previous conversation turns for context.
        member_context: Optional pre-formatted member context string.

    Returns:
        Complete prompt string for Cortex COMPLETE.
    """
    # Format conversation history
    history_text = format_conversation_history(conversation_history or [])

    # Format current turn scratchpad
    scratchpad_text = format_scratchpad(scratchpad)

    # Determine member context - use provided or build from member_id
    if member_context is None:
        if member_id:
            member_context = f"Member ID: {member_id}"
        else:
            # Try to get member_id from conversation history
            historical_member_id = extract_member_id_from_history(conversation_history or [])
            if historical_member_id:
                member_context = f"Member ID: {historical_member_id} (from previous conversation)"
            else:
                member_context = "Member ID: Not provided (general query)"

    return f"""{REACT_SYSTEM_PROMPT}

=== CONVERSATION HISTORY ===
{history_text}

=== CURRENT QUESTION ===
User Question: {user_query}
{member_context}

=== CURRENT TURN REASONING STEPS ===
{scratchpad_text}

=== YOUR TURN ===
Now reason about what to do next. Use the conversation history for context if needed.
If you have enough information, provide the FINAL_ANSWER.

"""
