"""Async wrappers for Snowflake Cortex services."""

import asyncio
import json
import logging
from typing import Any

from snowflake.snowpark import Session
from snowflake.snowpark.row import Row

from src.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions (extracted for testability)
# =============================================================================


def _row_to_dict(row: Row) -> dict[str, Any]:
    """Convert Snowpark Row to dictionary.

    Args:
        row: Snowpark Row object with _fields attribute.

    Returns:
        Dictionary with field names as keys.
    """
    return dict(zip(row._fields, row, strict=True))


def _build_member_query(member_id: str) -> str:
    """Build SQL query for member data.

    SQL Safety: member_id is validated by QueryRequest.validate_member_id()
    to be exactly 9 numeric digits, preventing SQL injection.

    Args:
        member_id: Validated 9-digit member ID.

    Returns:
        SQL query string for member data.
    """
    return f"""
    SELECT DISTINCT
        member_id, name, dob, gender, address, member_phone,
        plan_id, plan_name, plan_type, premium,
        pcp, pcp_phone
    FROM HEALTHCARE_DB.MEMBER_SCHEMA.CALL_CENTER_MEMBER_DENORMALIZED
    WHERE member_id = '{member_id}'
    LIMIT 1
    """


def _build_claims_query(member_id: str) -> str:
    """Build SQL query for member claims.

    SQL Safety: member_id is validated by QueryRequest.validate_member_id()
    to be exactly 9 numeric digits, preventing SQL injection.

    Args:
        member_id: Validated 9-digit member ID.

    Returns:
        SQL query string for claims data.
    """
    return f"""
    SELECT DISTINCT
        claim_id, claim_service_from_date, claim_service,
        claim_bill_amt, claim_paid_amt, claim_status
    FROM HEALTHCARE_DB.MEMBER_SCHEMA.CALL_CENTER_MEMBER_DENORMALIZED
    WHERE member_id = '{member_id}' AND claim_id IS NOT NULL
    LIMIT 5
    """


def _build_coverage_query(member_id: str) -> str:
    """Build SQL query for member coverage.

    SQL Safety: member_id is validated by QueryRequest.validate_member_id()
    to be exactly 9 numeric digits, preventing SQL injection.

    Args:
        member_id: Validated 9-digit member ID.

    Returns:
        SQL query string for coverage data.
    """
    return f"""
    SELECT DISTINCT
        plan_id, plan_name, plan_type, premium
    FROM HEALTHCARE_DB.MEMBER_SCHEMA.CALL_CENTER_MEMBER_DENORMALIZED
    WHERE member_id = '{member_id}'
    LIMIT 1
    """


def _get_search_columns(source: str) -> str:
    """Get column specification for search source.

    Args:
        source: Search service name.

    Returns:
        JSON array string of columns to return.
    """
    column_map = {
        "FAQS_SEARCH": '["faq_id", "question", "answer"]',
        "POLICIES_SEARCH": '["policy_id", "policy_name", "content"]',
        "TRANSCRIPTS_SEARCH": '["transcript_id", "member_id", "summary"]',
    }
    return column_map.get(source, "[]")


def _parse_search_result(result_data: str | dict, source: str, limit: int) -> list[dict]:
    """Parse Cortex Search result into structured format.

    Args:
        result_data: Raw search result (JSON string or dict).
        source: Search service name for metadata.
        limit: Maximum results to return.

    Returns:
        List of parsed search result dictionaries.
    """
    parsed_results: list[dict] = []

    try:
        if isinstance(result_data, str):
            json_data = json.loads(result_data)
        else:
            json_data = result_data

        search_results = json_data.get("results", [])
        for item in search_results[:limit]:
            # Combine relevant fields into text, truncated to 5000 chars
            text_parts = []
            for key, val in item.items():
                if key not in ["@scores", "@search_score"] and val:
                    text_parts.append(f"{key}: {val}")
            text = " | ".join(text_parts)[:5000]

            parsed_results.append({
                "text": text,
                "score": item.get("@search_score", 0.9),
                "source": source.lower().replace("_search", ""),
                "metadata": item.get("@scores", {}),
            })

    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse search result: {e}")
        # Fallback: truncate raw result
        parsed_results.append({
            "text": str(result_data)[:5000],
            "score": 0.9,
            "source": source.lower().replace("_search", ""),
            "metadata": {},
        })

    return parsed_results


# =============================================================================
# Cortex Analyst Tool
# =============================================================================


class AsyncCortexAnalystTool:
    """Async wrapper for Cortex Analyst queries against member data.

    Provides async interface to query member data from the denormalized
    CALL_CENTER_MEMBER_DENORMALIZED table using Snowpark.
    """

    def __init__(self, session: Session) -> None:
        """Initialize with Snowpark session.

        Args:
            session: Active Snowpark Session for Snowflake queries.
        """
        self.session = session

    def _execute_member_queries(self, member_id: str) -> dict[str, list]:
        """Execute member-specific queries synchronously.

        Args:
            member_id: Validated 9-digit member ID.

        Returns:
            Dictionary with member, claims, and coverage data lists.
        """
        # Ensure warehouse is set
        self.session.sql(f"USE WAREHOUSE {settings.snowflake_warehouse}").collect()

        # Execute queries
        member_sql = _build_member_query(member_id)
        claims_sql = _build_claims_query(member_id)
        coverage_sql = _build_coverage_query(member_id)

        logger.debug("Executing member query: %s", member_sql)
        member_rows = self.session.sql(member_sql).collect()

        logger.debug("Executing claims query: %s", claims_sql)
        claims_rows = self.session.sql(claims_sql).collect()

        logger.debug("Executing coverage query: %s", coverage_sql)
        coverage_rows = self.session.sql(coverage_sql).collect()

        return {
            "member": [_row_to_dict(r) for r in member_rows] if member_rows else [],
            "claims": [_row_to_dict(r) for r in claims_rows] if claims_rows else [],
            "coverage": [_row_to_dict(r) for r in coverage_rows] if coverage_rows else [],
        }

    def _execute_semantic_query(self, query: str) -> dict[str, str | None]:
        """Execute semantic query using Cortex COMPLETE.

        Args:
            query: Natural language query.

        Returns:
            Dictionary with response field.
        """
        # Ensure warehouse is set
        self.session.sql(f"USE WAREHOUSE {settings.snowflake_warehouse}").collect()

        escaped_query = query.replace("'", "''")
        sql = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            'llama3.1-70b',
            'Based on healthcare member data, answer: {escaped_query}'
        ) AS response
        """
        result = self.session.sql(sql).collect()
        return {"response": str(result[0][0]) if result else None}

    def _execute_aggregate_query(self, query: str) -> dict[str, Any]:
        """Execute aggregate query over member database.

        Handles queries like "how many members", "total claims", etc.

        Args:
            query: Natural language query about database aggregates.

        Returns:
            Dictionary with aggregate results.
        """
        # Ensure warehouse is set
        self.session.sql(f"USE WAREHOUSE {settings.snowflake_warehouse}").collect()

        query_lower = query.lower()

        # Determine what aggregate to run based on query
        if "member" in query_lower and any(kw in query_lower for kw in ["how many", "count", "total"]):
            sql = """
            SELECT 
                COUNT(DISTINCT member_id) as total_members,
                COUNT(DISTINCT plan_id) as total_plans,
                COUNT(DISTINCT plan_type) as plan_types
            FROM HEALTHCARE_DB.MEMBER_SCHEMA.CALL_CENTER_MEMBER_DENORMALIZED
            """
            result = self.session.sql(sql).collect()
            if result:
                row = result[0]
                return {
                    "query_type": "member_count",
                    "total_members": row[0],
                    "total_plans": row[1],
                    "plan_types": row[2],
                }

        elif "claim" in query_lower and any(kw in query_lower for kw in ["how many", "count", "total"]):
            sql = """
            SELECT 
                COUNT(DISTINCT claim_id) as total_claims,
                SUM(claim_bill_amt) as total_billed,
                SUM(claim_paid_amt) as total_paid
            FROM HEALTHCARE_DB.MEMBER_SCHEMA.CALL_CENTER_MEMBER_DENORMALIZED
            WHERE claim_id IS NOT NULL
            """
            result = self.session.sql(sql).collect()
            if result:
                row = result[0]
                return {
                    "query_type": "claims_count",
                    "total_claims": row[0],
                    "total_billed": float(row[1]) if row[1] else 0,
                    "total_paid": float(row[2]) if row[2] else 0,
                }

        # Default: get general database stats
        sql = """
        SELECT 
            COUNT(DISTINCT member_id) as total_members,
            COUNT(DISTINCT claim_id) as total_claims,
            COUNT(DISTINCT plan_id) as total_plans
        FROM HEALTHCARE_DB.MEMBER_SCHEMA.CALL_CENTER_MEMBER_DENORMALIZED
        """
        result = self.session.sql(sql).collect()
        if result:
            row = result[0]
            return {
                "query_type": "database_stats",
                "total_members": row[0],
                "total_claims": row[1],
                "total_plans": row[2],
            }

        return {"query_type": "unknown", "error": "Could not determine aggregate query"}

    def _is_aggregate_query(self, query: str) -> bool:
        """Check if query is an aggregate query (counts, totals, etc.).

        Args:
            query: Natural language query.

        Returns:
            True if query is asking for aggregate data.
        """
        query_lower = query.lower()
        aggregate_keywords = [
            "how many",
            "total",
            "count",
            "all members",
            "all claims",
            "statistics",
            "summary",
            "average",
            "database",
        ]
        return any(kw in query_lower for kw in aggregate_keywords)

    async def execute(self, query: str, member_id: str | None = None) -> dict[str, Any]:
        """Execute Cortex Analyst query asynchronously.

        Args:
            query: Natural language query about member data.
            member_id: Optional member ID for filtered queries (validated 9 digits).

        Returns:
            Dictionary with member_id, claims, coverage, raw_response, and/or aggregate_result.

        Raises:
            Exception: Re-raised from underlying Snowflake query failures.
        """
        try:
            if member_id:
                # Member-specific query
                result = await asyncio.to_thread(
                    self._execute_member_queries, member_id
                )
                return {
                    "member_id": member_id,
                    "claims": result.get("claims", []),
                    "coverage": result.get("coverage", [{}])[0] if result.get("coverage") else {},
                    "raw_response": str(result.get("member", [])),
                    "aggregate_result": None,
                }
            elif self._is_aggregate_query(query):
                # Database aggregate query (counts, totals, etc.)
                aggregate_result = await asyncio.to_thread(
                    self._execute_aggregate_query, query
                )
                return {
                    "member_id": None,
                    "claims": [],
                    "coverage": {},
                    "raw_response": str(aggregate_result),
                    "aggregate_result": aggregate_result,
                }
            else:
                # General semantic query
                result = await asyncio.to_thread(self._execute_semantic_query, query)
                return {
                    "member_id": None,
                    "claims": [],
                    "coverage": {},
                    "raw_response": result.get("response"),
                    "aggregate_result": None,
                }

        except Exception as exc:
            logger.error("Cortex Analyst error: %s", exc)
            logger.error("Query was for member_id: %s", member_id)
            logger.error("Database context: %s", settings.snowflake_database)
            raise


# =============================================================================
# Cortex Search Tool
# =============================================================================


class AsyncCortexSearchTool:
    """Async wrapper for Cortex Search queries across knowledge base.

    Searches across FAQs, policies, and transcripts using Snowflake
    Cortex Search with parallel execution.
    """

    def __init__(self, session: Session) -> None:
        """Initialize with Snowpark session.

        Args:
            session: Active Snowpark Session for Snowflake queries.
        """
        self.session = session

    def _execute_search(self, source: str, query: str, limit: int) -> list[dict]:
        """Execute single source search synchronously.

        Args:
            source: Search service name (e.g., FAQS_SEARCH).
            query: Search query text.
            limit: Maximum results to return.

        Returns:
            List of parsed search results.
        """
        # Ensure warehouse is set
        self.session.sql(f"USE WAREHOUSE {settings.snowflake_warehouse}").collect()

        escaped_query = query.replace("'", "''").replace('"', '\\"')
        columns = _get_search_columns(source)

        sql = f"""
        SELECT SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
            'HEALTHCARE_DB.KNOWLEDGE_SCHEMA.{source}',
            '{{"query": "{escaped_query}", "columns": {columns}, "limit": {limit}}}'
        ) AS results
        """

        try:
            rows = self.session.sql(sql).collect()
        except Exception as exc:
            logger.warning("Search source %s failed: %s", source, exc)
            return []

        parsed_results: list[dict] = []
        for row in rows:
            row_dict = row.as_dict() if hasattr(row, "as_dict") else row
            result_data = row_dict.get("RESULTS", "{}")
            parsed_results.extend(_parse_search_result(result_data, source, limit))

        return parsed_results

    async def _search_source_async(self, source: str, query: str, limit: int) -> list[dict]:
        """Execute search for a single source asynchronously.

        Args:
            source: Search service name.
            query: Search query text.
            limit: Maximum results per source.

        Returns:
            List of search results for this source.
        """
        return await asyncio.to_thread(self._execute_search, source, query, limit)

    async def execute(
        self,
        query: str,
        sources: list[str] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Execute Cortex Search query asynchronously.

        Searches multiple sources in parallel using asyncio.TaskGroup.

        Args:
            query: Natural language search query.
            sources: List of search services to query (default: all three).
            limit: Maximum results per source.

        Returns:
            List of search results with text, score, source, metadata.
        """
        sources = sources or ["FAQS_SEARCH", "POLICIES_SEARCH", "TRANSCRIPTS_SEARCH"]

        all_results: list[dict] = []

        try:
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(self._search_source_async(source, query, limit))
                    for source in sources
                ]

            for task in tasks:
                all_results.extend(task.result())

        except* Exception as eg:
            for exc in eg.exceptions:
                logger.warning("Search task failed: %s", exc)

        return all_results
