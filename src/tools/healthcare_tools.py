"""Healthcare tools using LangChain @tool decorator.

These tools are designed for use with LangGraph's ToolNode for automatic
tool execution. Each tool is an async function that returns a string result.

Tools:
- query_member_data: Query Snowflake database via Cortex Analyst
- search_knowledge: Search FAQs, policies, and call transcripts via Cortex Search
"""

from __future__ import annotations

import logging
from typing import Annotated

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Module-level session for SPCS OAuth token auth
_snowpark_session = None


def set_snowpark_session(session) -> None:
    """Set the Snowpark session for tool execution.

    Called during app startup with the OAuth-authenticated session.
    """
    global _snowpark_session
    _snowpark_session = session
    logger.info("Snowpark session set for healthcare tools")


def get_snowpark_session():
    """Get the current Snowpark session."""
    return _snowpark_session


# =============================================================================
# Tool Definitions
# =============================================================================


@tool
async def query_member_data(
    query: Annotated[str, "Natural language query about member data, claims, or coverage"],
    member_id: Annotated[str | None, "Optional member ID for member-specific queries"] = None,
) -> str:
    """Query the healthcare database for member information, claims, and coverage.

    Use this tool to:
    - Look up member personal information (name, address, phone)
    - Get member's insurance plan and coverage details
    - Retrieve claims history (services, amounts, status)
    - Answer aggregate questions (how many members, total claims, etc.)

    Args:
        query: Natural language query describing what data you need.
        member_id: Optional member ID for member-specific queries. Pass None for aggregate queries.

    Returns:
        Formatted string with query results or error message.
    """
    from src.services.cortex_tools import AsyncCortexAnalystTool

    session = get_snowpark_session()
    if not session:
        logger.warning("No Snowpark session - returning mock data")
        return "[Database Query] No database connection available. Unable to retrieve member data."

    try:
        analyst_tool = AsyncCortexAnalystTool(session)
        result = await analyst_tool.execute(query=query, member_id=member_id)

        # Format the result as a readable string
        return _format_analyst_result(result)

    except Exception as e:
        logger.error(f"query_member_data failed: {e}")
        return f"[Database Query Error] Unable to retrieve data: {str(e)}"


@tool
async def search_knowledge(
    query: Annotated[str, "Search query for FAQs, policies, or call transcripts"],
) -> str:
    """Search the healthcare knowledge base for FAQs, policies, and call transcripts.

    Use this tool to:
    - Find answers to common questions (FAQs)
    - Look up policy details and coverage rules
    - Search call transcripts for previous interactions
    - Answer questions about procedures and requirements

    Args:
        query: Natural language search query.

    Returns:
        Formatted string with relevant search results or message if none found.
    """
    from src.services.cortex_tools import AsyncCortexSearchTool

    session = get_snowpark_session()
    if not session:
        logger.warning("No Snowpark session - returning mock data")
        return "[Knowledge Search] No database connection available. Unable to search knowledge base."

    try:
        search_tool = AsyncCortexSearchTool(session)
        results = await search_tool.execute(query=query)

        # Format the results as a readable string
        return _format_search_results(results)

    except Exception as e:
        logger.error(f"search_knowledge failed: {e}")
        return f"[Knowledge Search Error] Unable to search: {str(e)}"


# =============================================================================
# Result Formatting Helpers
# =============================================================================


def _format_analyst_result(result: dict) -> str:
    """Format Cortex Analyst result as a readable string."""
    parts = ["[Database Query Results]"]

    # Handle aggregate results
    if result.get("aggregate_result"):
        agg = result["aggregate_result"]
        query_type = agg.get("query_type", "unknown")

        if query_type == "member_count":
            parts.append(f"Total members: {agg.get('total_members', 'N/A')}")
            if agg.get("total_plans"):
                parts.append(f"Total plans: {agg.get('total_plans')}")
        elif query_type == "member_names":
            names = agg.get("names", [])
            parts.append(f"Found {len(names)} members: {', '.join(names[:10])}")
            if len(names) > 10:
                parts.append(f"... and {len(names) - 10} more")
        else:
            parts.append(f"Query result: {agg}")

        return "\n".join(parts)

    # Handle member-specific results
    member_id = result.get("member_id")
    if member_id:
        parts.append(f"Member ID: {member_id}")

    # Personal info
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
        phone = member_info.get("PHONE") or member_info.get("phone") or member_info.get("MEMBER_PHONE")
        if phone:
            parts.append(f"Phone: {phone}")
        address = member_info.get("ADDRESS") or member_info.get("address")
        if address:
            parts.append(f"Address: {address}")
        # Additional personal details
        smoker = member_info.get("SMOKER")
        if smoker is not None:
            parts.append(f"Smoker: {'Yes' if smoker else 'No'}")
        lifestyle = member_info.get("LIFESTYLE")
        if lifestyle:
            parts.append(f"Lifestyle: {lifestyle}")
        chronic = member_info.get("CHRONIC_CONDITION")
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

    # Claims
    claims = result.get("claims", [])
    if claims:
        parts.append(f"--- Claims ({len(claims)} found) ---")
        for claim in claims[:5]:
            service = claim.get("CLAIM_SERVICE") or claim.get("service") or "Unknown"
            amount = claim.get("CLAIM_BILL_AMT") or claim.get("amount") or "N/A"
            status = claim.get("CLAIM_STATUS") or claim.get("status") or "N/A"
            parts.append(f"  - {service}: ${amount} ({status})")
        if len(claims) > 5:
            parts.append(f"  ... and {len(claims) - 5} more claims")

    if len(parts) == 1:
        parts.append("No specific data found for this query.")

    return "\n".join(parts)


def _format_search_results(results: list) -> str:
    """Format Cortex Search results as a readable string."""
    if not results:
        return "[Knowledge Search] No relevant documents found."

    # Sort by source priority (transcripts > policies > faqs)
    def sort_key(item: dict) -> tuple:
        source = item.get("source", "")
        score = item.get("score", 0)
        source_priority = 0 if source == "transcripts" else 1 if source == "policies" else 2
        return (source_priority, -score)

    sorted_results = sorted(results, key=sort_key)

    parts = [f"[Knowledge Search] Found {len(results)} relevant result(s):"]

    for i, item in enumerate(sorted_results[:5], 1):
        source = item.get("source", "unknown")
        text = item.get("text", "")
        if len(text) > 300:
            text = text[:300] + "..."
        parts.append(f"\n{i}. [{source}]: {text}")

    if len(results) > 5:
        parts.append(f"\n... and {len(results) - 5} more results")

    return "\n".join(parts)


# =============================================================================
# Tool Collection for ToolNode
# =============================================================================


def get_healthcare_tools() -> list:
    """Get the list of healthcare tools for ToolNode.

    Returns:
        List of @tool decorated functions.
    """
    return [query_member_data, search_knowledge]
