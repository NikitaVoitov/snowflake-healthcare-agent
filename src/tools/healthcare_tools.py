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
        response = await search_tool.execute(query=query)

        # Format the results as a readable string (response is now a dict with results and metadata)
        return _format_search_results(response)

    except Exception as e:
        logger.error(f"search_knowledge failed: {e}")
        return f"[Knowledge Search Error] Unable to search: {str(e)}"


# =============================================================================
# Result Formatting Helpers
# =============================================================================


def _format_analyst_result(result: dict) -> str:
    """Format Cortex Analyst result as a readable string with observability metadata.

    The formatted string includes structured metadata markers that can be parsed
    by instrumentation to extract Cortex Analyst metrics and trace correlation.
    """
    parts = ["[Database Query Results]"]

    # Extract Cortex Analyst observability metadata
    cortex_metadata = result.get("cortex_analyst_metadata", {})

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

        # Add metadata for aggregate results too
        _append_cortex_analyst_metadata(parts, cortex_metadata, result)
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
            service = claim.get("CLAIM_SERVICE") or claim.get("claim_service") or claim.get("service") or "Unknown"
            billed = claim.get("CLAIM_BILL_AMT") or claim.get("claim_bill_amt") or claim.get("claim_billed_amount") or claim.get("amount")
            paid = claim.get("CLAIM_PAID_AMT") or claim.get("claim_paid_amt") or claim.get("claim_paid_amount")
            status = claim.get("CLAIM_STATUS") or claim.get("claim_status") or claim.get("status") or "N/A"
            provider = claim.get("CLAIM_PROVIDER") or claim.get("claim_provider") or claim.get("provider")
            claim_id = claim.get("CLAIM_ID") or claim.get("claim_id")

            # Build claim line with available info
            line = f"  - {service}"
            if claim_id:
                line = f"  - [{claim_id}] {service}"
            if billed:
                line += f": Billed ${billed}"
            if paid:
                line += f" / Paid ${paid}"
            if provider:
                line += f" ({provider})"
            line += f" - {status}"
            parts.append(line)
        if len(claims) > 5:
            parts.append(f"  ... and {len(claims) - 5} more claims")

    if len(parts) == 1:
        parts.append("No specific data found for this query.")

    # Add metadata for member-specific results
    _append_cortex_analyst_metadata(parts, cortex_metadata, result)
    return "\n".join(parts)


def _append_cortex_analyst_metadata(parts: list, cortex_metadata: dict, result: dict) -> None:
    """Append Cortex Analyst observability metadata markers to the result.

    These markers follow the same pattern as Cortex Search (@sources, @request_ids, etc.)
    and can be parsed by instrumentation to extract Cortex Analyst metrics and trace correlation.
    """
    if not cortex_metadata:
        return

    parts.append("\n--- Cortex Analyst Metadata ---")

    # Snowflake context
    sf_ctx = cortex_metadata.get("snowflake", {})
    if sf_ctx.get("database"):
        parts.append(f"@snowflake.database: {sf_ctx['database']}")
    if sf_ctx.get("schema"):
        parts.append(f"@snowflake.schema: {sf_ctx['schema']}")
    if sf_ctx.get("warehouse"):
        parts.append(f"@snowflake.warehouse: {sf_ctx['warehouse']}")

    # Semantic model info
    sem_model = cortex_metadata.get("semantic_model", {})
    if sem_model.get("name"):
        parts.append(f"@cortex_analyst.semantic_model.name: {sem_model['name']}")
    if sem_model.get("type"):
        parts.append(f"@cortex_analyst.semantic_model.type: {sem_model['type']}")

    # Request ID for correlation with Snowflake logs
    if cortex_metadata.get("request_id"):
        parts.append(f"@cortex_analyst.request_id: {cortex_metadata['request_id']}")

    # Generated SQL (may be truncated for readability)
    sql = result.get("sql")
    if sql:
        # Truncate very long SQL for the formatted output
        sql_display = sql[:500] + "..." if len(sql) > 500 else sql
        parts.append(f"@cortex_analyst.sql: {sql_display}")

    # LLM models used
    model_names = cortex_metadata.get("model_names", [])
    if model_names:
        parts.append(f"@cortex_analyst.model_names: {model_names}")

    # Question category
    if cortex_metadata.get("question_category"):
        parts.append(f"@cortex_analyst.question_category: {cortex_metadata['question_category']}")

    # Verified Query Repository (VQR) info
    vqr = cortex_metadata.get("verified_query_used")
    if vqr:
        if vqr.get("name"):
            parts.append(f"@cortex_analyst.verified_query.name: {vqr['name']}")
        if vqr.get("question"):
            parts.append(f"@cortex_analyst.verified_query.question: {vqr['question']}")
        if vqr.get("sql"):
            vqr_sql = vqr["sql"][:300] + "..." if len(vqr["sql"]) > 300 else vqr["sql"]
            parts.append(f"@cortex_analyst.verified_query.sql: {vqr_sql}")
        if vqr.get("verified_at"):
            parts.append(f"@cortex_analyst.verified_query.verified_at: {vqr['verified_at']}")
        if vqr.get("verified_by"):
            parts.append(f"@cortex_analyst.verified_query.verified_by: {vqr['verified_by']}")

    # Warnings count
    warnings = result.get("warnings", [])
    parts.append(f"@cortex_analyst.warnings_count: {len(warnings)}")


def _format_search_results(response: dict) -> str:
    """Format Cortex Search results as a readable string with preserved metadata.

    The formatted string includes structured metadata markers that can be parsed
    by instrumentation to extract Cortex Search metrics like scores and request IDs.

    Args:
        response: Dictionary with 'results' list and 'cortex_search_metadata' dict.
    """
    # Handle both old format (list) and new format (dict with results key)
    if isinstance(response, list):
        results = response
        cortex_metadata = {}
    else:
        results = response.get("results", [])
        cortex_metadata = response.get("cortex_search_metadata", {})

    if not results:
        return "[Knowledge Search] No relevant documents found."

    # Sort by source priority (transcripts > policies > faqs)
    def sort_key(item: dict) -> tuple:
        source = item.get("source", "")
        # Use cosine_similarity as primary score if available
        metadata = item.get("metadata", {})
        scores = metadata.get("@scores", {})
        score = scores.get("cosine_similarity", item.get("score", 0))
        source_priority = 0 if source == "transcripts" else 1 if source == "policies" else 2
        return (source_priority, -score if isinstance(score, (int, float)) else 0)

    sorted_results = sorted(results, key=sort_key)

    parts = [f"[Knowledge Search] Found {len(results)} relevant result(s):"]

    # Collect metrics for instrumentation visibility
    all_scores = []
    request_ids = set()
    sources_searched = set()

    for i, item in enumerate(sorted_results[:5], 1):
        source = item.get("source", "unknown")
        sources_searched.add(source)
        text = item.get("text", "")
        if len(text) > 300:
            text = text[:300] + "..."

        # Extract scores from metadata for display
        metadata = item.get("metadata", {})
        scores = metadata.get("@scores", {})
        cosine_sim = scores.get("cosine_similarity")
        text_match = scores.get("text_match")

        # Collect request_id if available
        req_id = metadata.get("_snowflake_request_id")
        if req_id:
            request_ids.add(req_id)

        # Build result line with score info
        score_info = ""
        if cosine_sim is not None or text_match is not None:
            score_parts = []
            if cosine_sim is not None:
                score_parts.append(f"cosine={cosine_sim:.3f}")
                all_scores.append({"cosine_similarity": cosine_sim, "text_match": text_match})
            if text_match is not None:
                score_parts.append(f"text={text_match:.3f}")
            score_info = f" @scores: {{{', '.join(score_parts)}}}"

        parts.append(f"\n{i}. [{source}]: {text}{score_info}")

    if len(results) > 5:
        parts.append(f"\n... and {len(results) - 5} more results")

    # Add metadata summary at end for instrumentation parsing
    if request_ids:
        parts.append(f"\n@request_ids: {list(request_ids)}")
    if sources_searched:
        parts.append(f"\n@sources: {list(sources_searched)}")

    # Add top score summary
    if all_scores:
        top_cosine = max((s.get("cosine_similarity", 0) or 0) for s in all_scores)
        parts.append(f"\n@top_cosine_similarity: {top_cosine:.3f}")

    # Add Snowflake context metadata (same as Cortex Analyst)
    _append_cortex_search_metadata(parts, cortex_metadata)

    return "\n".join(parts)


def _append_cortex_search_metadata(parts: list, cortex_metadata: dict) -> None:
    """Append Cortex Search observability metadata markers (Snowflake context).

    These markers follow the same pattern as Cortex Analyst (@snowflake.database, etc.)
    to provide consistent observability attributes for both services.
    """
    if not cortex_metadata:
        return

    parts.append("\n--- Cortex Search Metadata ---")

    # Snowflake context (same attributes as Cortex Analyst)
    sf_ctx = cortex_metadata.get("snowflake", {})
    if sf_ctx.get("database"):
        parts.append(f"@snowflake.database: {sf_ctx['database']}")
    if sf_ctx.get("schema"):
        parts.append(f"@snowflake.schema: {sf_ctx['schema']}")
    if sf_ctx.get("warehouse"):
        parts.append(f"@snowflake.warehouse: {sf_ctx['warehouse']}")


# =============================================================================
# Tool Collection for ToolNode
# =============================================================================


def get_healthcare_tools() -> list:
    """Get the list of healthcare tools for ToolNode.

    Returns:
        List of @tool decorated functions.
    """
    return [query_member_data, search_knowledge]
