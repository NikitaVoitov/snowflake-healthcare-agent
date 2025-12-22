"""ReAct output parser for LLM responses.

Parses the Thought/Action/Action Input format from Cortex COMPLETE responses.
Uses orjson for fast serialization, stdlib json for lenient LLM parsing.
"""

from __future__ import annotations

import json
import logging
import re
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
# JSON Utilities (orjson for speed, stdlib for flexibility)
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


def _parse_json_lenient(json_str: str) -> dict | None:
    """Parse JSON with multiple fallback strategies for malformed LLM output.

    Args:
        json_str: Potentially malformed JSON string.

    Returns:
        Parsed dictionary or None if all strategies fail.
    """
    # Strategy 1: Try standard parsing after cleaning
    cleaned = _clean_json_string(json_str)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Try to extract just the JSON object with regex
    json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    matches = re.findall(json_pattern, json_str, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    # Strategy 3: Try to fix common issues
    fixed = _fix_common_json_issues(cleaned)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Strategy 4: Extract key-value pairs manually for simple structures
    kv_result = _extract_key_values(json_str)
    if kv_result:
        return kv_result

    return None


def _fix_common_json_issues(json_str: str) -> str:
    """Fix common JSON formatting issues from LLM output.

    Args:
        json_str: JSON string with potential issues.

    Returns:
        Fixed JSON string.
    """
    # Remove trailing commas before } or ]
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

    # Fix unquoted null/true/false
    json_str = re.sub(r":\s*None\b", ": null", json_str)
    json_str = re.sub(r":\s*True\b", ": true", json_str)
    json_str = re.sub(r":\s*False\b", ": false", json_str)

    # Fix missing quotes around string values (simple cases)
    # e.g., {query: hello} -> {"query": "hello"}
    json_str = re.sub(r"\{(\s*)(\w+)(\s*):", r'{"\2":', json_str)

    return json_str


def _extract_key_values(text: str) -> dict | None:
    """Extract key-value pairs from text as last resort.

    Handles cases like: query: "some text", member_id: null

    Args:
        text: Text containing key-value pairs.

    Returns:
        Dictionary of extracted values or None.
    """
    result = {}

    # Pattern for: key: "value" or key: value
    patterns = [
        r'"?(\w+)"?\s*:\s*"([^"]*)"',  # key: "value"
        r'"?(\w+)"?\s*:\s*(\d+)',  # key: 123
        r'"?(\w+)"?\s*:\s*(null|true|false)',  # key: null/true/false
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            key = match.group(1).lower()
            value = match.group(2)
            if value.lower() == "null":
                value = None
            elif value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            result[key] = value

    return result if result else None


def parse_react_response(response: str) -> dict:
    """Parse ReAct-formatted LLM response into structured components.

    Expected format:
        Thought: [reasoning text]
        Action: [tool_name or FINAL_ANSWER]
        Action Input: {"key": "value"}

    Args:
        response: Raw text response from Cortex COMPLETE.

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
        # Extract Thought (everything between "Thought:" and "Action:")
        thought_pattern = r"Thought:\s*(.+?)(?=Action:|$)"
        thought_match = re.search(thought_pattern, response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()

        # Extract Action (single word after "Action:")
        action_pattern = r"Action:\s*(\w+)"
        action_match = re.search(action_pattern, response, re.IGNORECASE)
        if action_match:
            action = action_match.group(1).strip()
            # Validate action
            if action in VALID_ACTIONS:
                result["action"] = action
            else:
                logger.warning(f"Unknown action '{action}', defaulting to FINAL_ANSWER")
                result["action"] = "FINAL_ANSWER"

        # Extract Action Input (JSON object after "Action Input:")
        # More flexible pattern that handles multiline JSON
        action_input_pattern = r"Action Input:\s*(\{.+)"
        action_input_match = re.search(action_input_pattern, response, re.DOTALL | re.IGNORECASE)

        if action_input_match:
            action_input_str = action_input_match.group(1).strip()

            # Use lenient parser with multiple fallback strategies
            parsed = _parse_json_lenient(action_input_str)

            if parsed:
                result["action_input"] = parsed
            else:
                logger.warning("Failed to parse Action Input - using fallback")
                # Try to extract answer from malformed JSON for FINAL_ANSWER
                if result["action"] == "FINAL_ANSWER":
                    result["action_input"] = _extract_answer_fallback(response)
                else:
                    # For tool calls, try to extract query/member_id
                    result["action_input"] = _extract_action_input_fallback(
                        response, result["action"]
                    )
        else:
            # No Action Input found - try alternative patterns
            result["action_input"] = _extract_action_input_fallback(response, result["action"])

    except Exception as e:
        logger.error(f"Error parsing ReAct response: {e}")

    logger.debug(f"Parsed ReAct: action={result['action']}, thought_len={len(result['thought'])}")
    return result


def _clean_json_string(json_str: str) -> str:
    """Clean common JSON formatting issues from LLM output.

    Args:
        json_str: Raw JSON string that may have formatting issues.

    Returns:
        Cleaned JSON string.
    """
    # Remove any trailing content after the JSON object
    # Find the matching closing brace
    brace_count = 0
    end_idx = 0
    for i, char in enumerate(json_str):
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break

    if end_idx > 0:
        json_str = json_str[:end_idx]

    # Replace single quotes with double quotes (common LLM mistake)
    # But be careful not to replace quotes inside strings
    # Simple approach: only if it looks like a simple key-value structure
    if "'" in json_str and '"' not in json_str:
        json_str = json_str.replace("'", '"')

    # Escape unescaped control characters inside JSON strings (common LLM issue)
    # Newlines, tabs, etc. inside strings must be escaped
    # This is a targeted fix for the most common issue: literal newlines in "answer" values
    json_str = _escape_control_chars_in_strings(json_str)

    return json_str


def _escape_control_chars_in_strings(json_str: str) -> str:
    """Escape control characters inside JSON string values.

    LLMs often produce JSON with literal newlines in strings like:
    {"answer": "Line 1
    Line 2"}

    This should be:
    {"answer": "Line 1\\nLine 2"}

    Args:
        json_str: JSON string that may have unescaped control chars.

    Returns:
        JSON string with properly escaped control characters.
    """
    result = []
    in_string = False
    i = 0

    while i < len(json_str):
        char = json_str[i]

        # Track when we're inside a JSON string
        if char == '"' and (i == 0 or json_str[i - 1] != "\\"):
            in_string = not in_string
            result.append(char)
        elif in_string:
            # Inside a string - escape control characters
            if char == "\n":
                result.append("\\n")
            elif char == "\r":
                result.append("\\r")
            elif char == "\t":
                result.append("\\t")
            else:
                result.append(char)
        else:
            result.append(char)

        i += 1

    return "".join(result)


def _extract_answer_fallback(response: str) -> dict:
    """Extract answer from response when JSON parsing fails.

    Args:
        response: Full LLM response text.

    Returns:
        Dictionary with 'answer' key.
    """
    # Try to find answer content after "answer":
    answer_pattern = r'"?answer"?\s*:\s*"([^"]+)"'
    match = re.search(answer_pattern, response, re.IGNORECASE)
    if match:
        return {"answer": match.group(1)}

    # Try multiline answer pattern (answer spans multiple lines)
    multiline_pattern = r'"?answer"?\s*:\s*"([\s\S]+?)"(?:\s*}|\s*$)'
    multiline_match = re.search(multiline_pattern, response, re.IGNORECASE)
    if multiline_match:
        # Clean up the extracted answer
        answer = multiline_match.group(1).replace("\n", " ").strip()
        if answer:
            return {"answer": answer}

    # IMPORTANT: Do NOT fall back to "Thought" - that's internal reasoning
    # Instead, provide a generic response that signals the need to retry or escalate
    logger.warning("Could not extract answer from LLM response, using generic fallback")
    fallback_msg = (
        "I was unable to complete the analysis. The data I found may not fully "
        "answer your question. Please try rephrasing or contact support."
    )
    return {"answer": fallback_msg}


def _extract_action_input_fallback(response: str, action: str) -> dict:
    """Extract action input using fallback patterns.

    Args:
        response: Full LLM response text.
        action: The action being taken.

    Returns:
        Dictionary with action parameters.
    """
    if action == "FINAL_ANSWER":
        return _extract_answer_fallback(response)

    # For tool actions, try to find query parameter
    query_pattern = r'"?query"?\s*:\s*"([^"]+)"'
    query_match = re.search(query_pattern, response, re.IGNORECASE)
    if query_match:
        result = {"query": query_match.group(1)}

        # Also try to find member_id
        member_pattern = r'"?member_id"?\s*:\s*"?([^",\s}]+)"?'
        member_match = re.search(member_pattern, response, re.IGNORECASE)
        if member_match:
            member_id = member_match.group(1)
            if member_id.lower() not in ("null", "none"):
                result["member_id"] = member_id

        return result

    return {}


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
