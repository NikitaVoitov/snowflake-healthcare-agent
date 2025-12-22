"""ReAct output parser for structured JSON responses from Cortex COMPLETE.

Parses the Thought/Action/Action Input JSON format from Cortex COMPLETE
with response_format structured output.
"""

from __future__ import annotations

import json
import logging
from typing import Any

# orjson for fast serialization (optional, falls back to stdlib)
try:
    import orjson

    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False

logger = logging.getLogger(__name__)

# Valid actions the agent can take
VALID_ACTIONS = {"query_member_data", "search_knowledge", "FINAL_ANSWER"}


# =============================================================================
# JSON Utilities
# =============================================================================


def json_dumps(obj: Any, default: Any = str) -> str:
    """Fast JSON serialization using orjson if available.

    Args:
        obj: Object to serialize.
        default: Default function for non-serializable types.

    Returns:
        JSON string.
    """
    if HAS_ORJSON:
        # orjson returns bytes, decode to str
        return orjson.dumps(obj, default=default).decode("utf-8")
    return json.dumps(obj, default=default)


def parse_react_response(response: str) -> dict:
    """Parse structured JSON response from Cortex COMPLETE.

    Expects direct JSON from Cortex COMPLETE with response_format:
        {"thought": "...", "action": "...", "action_input": {...}}

    Args:
        response: JSON string from Cortex COMPLETE structured output.

    Returns:
        Dictionary with 'thought', 'action', 'action_input' keys.
        Returns defaults if parsing fails.
    """
    # Default values for fallback
    fallback_answer = "I apologize, but I encountered an issue processing your request."
    result = {
        "thought": "",
        "action": "FINAL_ANSWER",
        "action_input": {"answer": fallback_answer},
    }

    if not response or not response.strip():
        logger.warning("Empty response from LLM")
        return result

    try:
        # Parse the JSON directly - structured output guarantees valid JSON
        parsed = json.loads(response) if isinstance(response, str) else response

        # Validate required fields
        thought = parsed.get("thought", "")
        action = parsed.get("action", "FINAL_ANSWER")
        action_input = parsed.get("action_input", {})

        # Validate action is in allowed set
        if action not in VALID_ACTIONS:
            logger.warning(f"Invalid action '{action}', using FINAL_ANSWER")
            action = "FINAL_ANSWER"
            if not action_input.get("answer"):
                action_input["answer"] = thought if thought else fallback_answer

        result["thought"] = thought
        result["action"] = action
        result["action_input"] = action_input

        logger.debug(f"Parsed ReAct: action={action}, thought_len={len(thought)}")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse structured JSON response: {e}")
        return result
    except Exception as e:
        logger.error(f"Unexpected error parsing response: {e}")
        return result


def format_observation(tool_name: str, result: dict | str | None) -> str:
    """Format tool result as a human-readable observation.

    Args:
        tool_name: Name of the tool that was executed.
        result: Result from the tool execution.

    Returns:
        Formatted observation string for the scratchpad.
    """
    if result is None:
        return f"[{tool_name}] No results found."

    if isinstance(result, str):
        return f"[{tool_name}] {result}"

    # Handle structured results from different tools
    if tool_name == "query_member_data":
        return _format_analyst_observation(result)
    elif tool_name == "search_knowledge":
        return _format_search_observation(result)
    else:
        return f"[{tool_name}] {json_dumps(result)[:500]}"


def _format_analyst_observation(result: dict) -> str:
    """Format Cortex Analyst result as observation.

    Args:
        result: Result dictionary from AsyncCortexAnalystTool.

    Returns:
        Formatted observation string.
    """
    parts = ["[Database Query Results]"]

    # Handle aggregate results
    if result.get("aggregate_result"):
        agg = result["aggregate_result"]
        query_type = agg.get("query_type", "unknown")

        if query_type == "member_count":
            parts.append(f"Total members: {agg.get('total_members', 'N/A')}")
            parts.append(f"Total plans: {agg.get('total_plans', 'N/A')}")
            parts.append(f"Plan types: {agg.get('plan_types', 'N/A')}")
        elif query_type == "member_names":
            names = agg.get("names", [])
            parts.append(f"Found {len(names)} members: {', '.join(names[:10])}")
            if len(names) > 10:
                parts.append(f"... and {len(names) - 10} more")
        elif query_type == "claims_count":
            parts.append(f"Total claims: {agg.get('total_claims', 'N/A')}")
            parts.append(f"Total billed: ${agg.get('total_billed', 'N/A')}")
        else:
            parts.append(f"Aggregate data: {json_dumps(agg)}")

        return "\n".join(parts)

    # Handle member-specific results
    member_id = result.get("member_id")
    if member_id:
        parts.append(f"Member ID: {member_id}")

    # Personal info - IMPORTANT: Include all member details
    member_info = result.get("member_info", {})
    if member_info:
        name = member_info.get("NAME") or member_info.get("name")
        if name:
            parts.append(f"Name: {name}")
        dob = member_info.get("DOB") or member_info.get("dob")
        if dob:
            parts.append(f"Date of Birth: {dob}")
        gender = member_info.get("GENDER") or member_info.get("gender")
        if gender:
            parts.append(f"Gender: {gender}")
        address = member_info.get("ADDRESS") or member_info.get("address")
        if address:
            parts.append(f"Address: {address}")
        phone = member_info.get("PHONE") or member_info.get("phone")
        phone = phone or member_info.get("MEMBER_PHONE")
        if phone:
            parts.append(f"Phone: {phone}")
        pcp = member_info.get("PCP") or member_info.get("pcp")
        if pcp:
            parts.append(f"Primary Care Physician: {pcp}")
        pcp_phone = member_info.get("PCP_PHONE") or member_info.get("pcp_phone")
        if pcp_phone:
            parts.append(f"PCP Phone: {pcp_phone}")
        smoker = member_info.get("SMOKER") or member_info.get("smoker")
        smoker = smoker or member_info.get("SMOKER_IND")
        if smoker is not None:
            parts.append(f"Smoker: {'Yes' if smoker else 'No'}")
        lifestyle = member_info.get("LIFESTYLE") or member_info.get("lifestyle")
        lifestyle = lifestyle or member_info.get("LIFESTYLE_INFO")
        if lifestyle:
            parts.append(f"Lifestyle: {lifestyle}")
        chronic = member_info.get("CHRONIC_CONDITION") or member_info.get("chronic_condition")
        if chronic:
            parts.append(f"Chronic Condition: {chronic}")

    # Coverage info
    coverage = result.get("coverage", {})
    if coverage:
        plan_name = coverage.get("PLAN_NAME") or coverage.get("plan_name")
        if plan_name:
            parts.append(f"Plan: {plan_name}")
        plan_type = coverage.get("PLAN_TYPE") or coverage.get("plan_type")
        if plan_type:
            parts.append(f"Plan Type: {plan_type}")
        premium = coverage.get("PREMIUM") or coverage.get("premium")
        if premium:
            parts.append(f"Premium: ${premium}")
        cvg_start = coverage.get("CVG_START_DATE") or coverage.get("cvg_start_date")
        if cvg_start:
            parts.append(f"Coverage Start: {cvg_start}")

    # Grievance info
    grievance = result.get("grievance", {})
    if grievance and grievance.get("GRIEVANCE_ID"):
        parts.append("--- Grievance ---")
        parts.append(f"Grievance ID: {grievance.get('GRIEVANCE_ID')}")
        parts.append(f"Type: {grievance.get('GRIEVANCE_TYPE', 'N/A')}")
        parts.append(f"Status: {grievance.get('GRIEVANCE_STATUS', 'N/A')}")
        parts.append(f"Date: {grievance.get('GRIEVANCE_DATE', 'N/A')}")

    # Claims summary
    claims = result.get("claims", [])
    if claims:
        parts.append(f"--- Claims ({len(claims)} found) ---")
        for claim in claims[:5]:  # Show first 5 claims
            service = claim.get("CLAIM_SERVICE") or claim.get("service") or "Unknown"
            amount = claim.get("CLAIM_BILL_AMT") or claim.get("amount") or "N/A"
            paid = claim.get("CLAIM_PAID_AMT") or claim.get("paid_amt") or "N/A"
            status = claim.get("CLAIM_STATUS") or claim.get("status") or "N/A"
            provider = claim.get("CLAIM_PROVIDER") or claim.get("provider") or ""
            date = claim.get("CLAIM_SERVICE_FROM_DATE") or claim.get("service_date") or ""
            parts.append(f"  - {service}: Billed ${amount}, Paid ${paid} ({status})")
            if provider:
                parts.append(f"    Provider: {provider}, Date: {date}")
        if len(claims) > 5:
            parts.append(f"  ... and {len(claims) - 5} more claims")

    # Raw response if available (for additional context)
    raw = result.get("raw_response")
    if raw and not member_info and not coverage and not claims:
        parts.append(f"Raw data: {str(raw)[:500]}")

    return "\n".join(parts) if len(parts) > 1 else "[Database Query] No specific data found."


def _format_search_observation(result: list | dict) -> str:
    """Format Cortex Search result as observation.

    Prioritizes transcript results for call-related queries and sorts by score.

    Args:
        result: Result from AsyncCortexSearchTool (list of search results).

    Returns:
        Formatted observation string.
    """
    if isinstance(result, dict):
        result = [result]

    if not result:
        return "[Knowledge Base Search] No relevant documents found."

    # Sort results by score (highest first) and prioritize transcripts
    def sort_key(item: dict) -> tuple:
        source = item.get("source", "")
        score = item.get("score", 0)
        # Prioritize transcripts (source priority: transcripts > policies > faqs)
        source_priority = 0 if source == "transcripts" else 1 if source == "policies" else 2
        return (source_priority, -score)  # Lower priority number = higher priority

    sorted_results = sorted(result, key=sort_key)

    parts = [f"[Knowledge Base Search] Found {len(result)} relevant result(s):"]

    # Show top 5 results to include results from multiple sources
    for i, item in enumerate(sorted_results[:5], 1):
        source = item.get("source", "unknown")
        text = item.get("text", "")
        # Truncate long text
        if len(text) > 300:
            text = text[:300] + "..."
        parts.append(f"\n{i}. [{source}]: {text}")

    if len(result) > 5:
        parts.append(f"\n... and {len(result) - 5} more results")

    return "\n".join(parts)
