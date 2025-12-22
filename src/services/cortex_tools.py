"""Async wrappers for Snowflake Cortex services.

This module provides async tools for:
- Cortex Analyst: NL→SQL conversion using semantic models (via REST API)
- Cortex Search: Semantic search across knowledge base documents

The Cortex Analyst tool uses the Cortex Analyst REST API with a semantic
model for accurate natural language to SQL conversion. Falls back to
direct Snowpark SQL queries in local development environments.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from snowflake.snowpark import Session
from snowflake.snowpark.row import Row

from src.config import settings

if TYPE_CHECKING:
    from src.services.cortex_analyst_client import AsyncCortexAnalystClient

logger = logging.getLogger(__name__)

# Check if running in SPCS (where OAuth token is available)
SPCS_TOKEN_FILE = Path("/snowflake/session/token")
IS_SPCS = SPCS_TOKEN_FILE.exists() or os.environ.get("SNOWFLAKE_HOST") is not None


# =============================================================================
# Helper Functions
# =============================================================================


def _row_to_dict(row: Row) -> dict[str, Any]:
    """Convert Snowpark Row to dictionary.

    Args:
        row: Snowpark Row object with _fields attribute.

    Returns:
        Dictionary with field names as keys.
    """
    return dict(zip(row._fields, row, strict=True))


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
        json_data = json.loads(result_data) if isinstance(result_data, str) else result_data

        search_results = json_data.get("results", [])
        for item in search_results[:limit]:
            # Combine relevant fields into text, truncated to 5000 chars
            text_parts = []
            for key, val in item.items():
                if key not in ["@scores", "@search_score"] and val:
                    text_parts.append(f"{key}: {val}")
            text = " | ".join(text_parts)[:5000]

            parsed_results.append(
                {
                    "text": text,
                    "score": item.get("@search_score", 0.9),
                    "source": source.lower().replace("_search", ""),
                    "metadata": item.get("@scores", {}),
                }
            )

    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse search result: {e}")
        parsed_results.append(
            {
                "text": str(result_data)[:5000],
                "score": 0.9,
                "source": source.lower().replace("_search", ""),
                "metadata": {},
            }
        )

    return parsed_results


# =============================================================================
# Cortex Analyst Tool
# =============================================================================


class AsyncCortexAnalystTool:
    """Async wrapper for Cortex Analyst queries using semantic model.

    Uses Snowflake Cortex Analyst REST API with a YAML semantic model
    for accurate natural language to SQL conversion.

    In local development (without OAuth), falls back to direct Snowpark
    SQL queries against the denormalized member table.
    """

    def __init__(self, session: Session) -> None:
        """Initialize with Snowpark session.

        Args:
            session: Active Snowpark Session for Snowflake queries.
        """
        self.session = session
        self._analyst_client: AsyncCortexAnalystClient | None = None

    def _get_analyst_client(self) -> AsyncCortexAnalystClient:
        """Get or create Cortex Analyst client.

        Returns:
            AsyncCortexAnalystClient instance.
        """
        if self._analyst_client is None:
            from src.services.cortex_analyst_client import AsyncCortexAnalystClient

            self._analyst_client = AsyncCortexAnalystClient(
                session=self.session,
                database=settings.snowflake_database,
                schema="STAGING",
            )
        return self._analyst_client

    async def execute(self, query: str, member_id: str | None = None) -> dict[str, Any]:
        """Execute natural language query using Cortex Analyst API.

        Uses Cortex Analyst REST API with semantic model for NL→SQL conversion.
        Supports both SPCS (OAuth) and local (JWT key-pair) authentication.

        Args:
            query: Natural language query about healthcare data.
            member_id: Optional member ID for filtered queries.

        Returns:
            Dictionary with member_id, claims, coverage, raw_response, etc.

        Raises:
            Exception: Re-raised from API or SQL execution failures.
        """
        try:
            return await self._execute_via_cortex_analyst_api(query, member_id)
        except ValueError as exc:
            # If authentication fails (no token available), fall back to Snowpark
            if "Cannot retrieve authentication token" in str(exc):
                logger.warning("Cortex Analyst auth unavailable, using Snowpark fallback")
                return await self._execute_via_snowpark(query, member_id)
            raise

    async def _execute_via_cortex_analyst_api(self, query: str, member_id: str | None = None) -> dict[str, Any]:
        """Execute query using Cortex Analyst REST API with semantic model.

        Args:
            query: Natural language query.
            member_id: Optional member ID for context.

        Returns:
            Structured query results.
        """
        client = self._get_analyst_client()

        # Enhance query with member context if provided
        enhanced_query = query
        if member_id:
            enhanced_query = f"For member ID {member_id}: {query}"

        logger.info(f"Cortex Analyst API query: {enhanced_query[:100]}...")

        # Call Cortex Analyst API
        response = await client.execute_query(enhanced_query)

        # Build result structure
        result: dict[str, Any] = {
            "member_id": member_id,
            "member_info": {},
            "claims": [],
            "coverage": {},
            "grievance": {},
            "raw_response": None,
            "aggregate_result": None,
            "sql": response.get("sql"),
            "interpretation": response.get("interpretation"),
            "suggestions": response.get("suggestions", []),
            "request_id": response.get("request_id"),
        }

        # Handle errors or suggestions
        if response.get("error"):
            result["error"] = response["error"]
            logger.warning(f"Cortex Analyst error: {response['error']}")
            return result

        if response.get("suggestions"):
            logger.info(f"Cortex Analyst returned suggestions: {response['suggestions']}")
            return result

        # Process query results
        rows = response.get("results", [])
        if rows:
            result["raw_response"] = str(rows)
            result_type = self._detect_result_type(query, rows)

            if result_type == "member_info":
                self._extract_member_info(rows, result, member_id)
            elif result_type == "claims":
                result["claims"] = self._format_claims(rows)
            elif result_type == "aggregate":
                result["aggregate_result"] = self._format_aggregate(rows)
            else:
                result["aggregate_result"] = {"query_type": "general", "data": rows}

        return result

    async def _execute_via_snowpark(self, query: str, member_id: str | None = None) -> dict[str, Any]:
        """Execute query using direct Snowpark SQL (local fallback).

        Uses the denormalized member table for comprehensive data access.

        Args:
            query: Natural language query.
            member_id: Optional member ID for filtered queries.

        Returns:
            Structured query results.
        """
        result: dict[str, Any] = {
            "member_id": member_id,
            "member_info": {},
            "claims": [],
            "coverage": {},
            "grievance": {},
            "raw_response": None,
            "aggregate_result": None,
            "sql": None,
        }

        try:
            if member_id:
                # Member-specific query
                rows = await asyncio.to_thread(self._query_member_data, member_id)
                result["raw_response"] = str(rows.get("member", []))
                result["claims"] = rows.get("claims", [])
                if rows.get("member"):
                    member = rows["member"][0] if rows["member"] else {}
                    result["member_info"] = member
                if rows.get("coverage"):
                    result["coverage"] = rows["coverage"][0] if rows["coverage"] else {}
            else:
                # Aggregate query
                agg_result = await asyncio.to_thread(self._query_aggregate, query)
                result["aggregate_result"] = agg_result
                result["raw_response"] = str(agg_result)

        except Exception as exc:
            logger.error(f"Snowpark query error: {exc}", exc_info=True)
            raise

        return result

    def _query_member_data(self, member_id: str) -> dict[str, list]:
        """Query member-specific data via Snowpark.

        Args:
            member_id: Validated member ID (9 digits).

        Returns:
            Dictionary with member, claims, and coverage data.
        """
        self.session.sql(f"USE WAREHOUSE {settings.snowflake_warehouse}").collect()

        # Member info query
        member_sql = f"""
        SELECT DISTINCT
            MEMBER_ID, NAME, DOB, GENDER, ADDRESS, MEMBER_PHONE,
            PLAN_ID, PLAN_NAME, PLAN_TYPE, PREMIUM, PCP, PCP_PHONE
        FROM HEALTHCARE_DB.MEMBER_SCHEMA.CALL_CENTER_MEMBER_DENORMALIZED
        WHERE MEMBER_ID = '{member_id}'
        LIMIT 1
        """

        # Claims query
        claims_sql = f"""
        SELECT DISTINCT
            CLAIM_ID, CLAIM_SERVICE_FROM_DATE, CLAIM_SERVICE,
            CLAIM_BILL_AMT, CLAIM_PAID_AMT, CLAIM_STATUS
        FROM HEALTHCARE_DB.MEMBER_SCHEMA.CALL_CENTER_MEMBER_DENORMALIZED
        WHERE MEMBER_ID = '{member_id}' AND CLAIM_ID IS NOT NULL
        ORDER BY CLAIM_SERVICE_FROM_DATE DESC
        LIMIT 10
        """

        # Coverage query
        coverage_sql = f"""
        SELECT DISTINCT
            PLAN_ID, PLAN_NAME, PLAN_TYPE, PREMIUM
        FROM HEALTHCARE_DB.MEMBER_SCHEMA.CALL_CENTER_MEMBER_DENORMALIZED
        WHERE MEMBER_ID = '{member_id}'
        LIMIT 1
        """

        member_rows = self.session.sql(member_sql).collect()
        claims_rows = self.session.sql(claims_sql).collect()
        coverage_rows = self.session.sql(coverage_sql).collect()

        return {
            "member": [_row_to_dict(r) for r in member_rows] if member_rows else [],
            "claims": [_row_to_dict(r) for r in claims_rows] if claims_rows else [],
            "coverage": [_row_to_dict(r) for r in coverage_rows] if coverage_rows else [],
        }

    def _query_aggregate(self, query: str) -> dict[str, Any]:
        """Query aggregate data via Snowpark.

        Args:
            query: Natural language query.

        Returns:
            Aggregate results dictionary.
        """
        self.session.sql(f"USE WAREHOUSE {settings.snowflake_warehouse}").collect()
        query_lower = query.lower()

        # Member count
        count_keywords = ["how many", "count", "total"]
        if "member" in query_lower and any(kw in query_lower for kw in count_keywords):
            sql = """
            SELECT
                COUNT(DISTINCT MEMBER_ID) as total_members,
                COUNT(DISTINCT PLAN_ID) as total_plans,
                COUNT(DISTINCT PLAN_TYPE) as plan_types
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

        # Top N members by premium (highest/lowest)
        top_match = re.search(r"top\s+(\d+)", query_lower)
        if top_match and "premium" in query_lower:
            limit = int(top_match.group(1))
            order = "DESC" if "highest" in query_lower or "top" in query_lower else "ASC"
            sql = f"""
            SELECT DISTINCT MEMBER_ID, NAME, PREMIUM
            FROM HEALTHCARE_DB.MEMBER_SCHEMA.CALL_CENTER_MEMBER_DENORMALIZED
            WHERE PREMIUM IS NOT NULL
            ORDER BY PREMIUM {order}
            LIMIT {limit}
            """
            result = self.session.sql(sql).collect()
            if result:
                members = [{"MEMBER_ID": row[0], "MEMBER_NAME": row[1], "PREMIUM": str(row[2])} for row in result]
                return {"query_type": "general", "data": members}

        # Top N members by claims (highest/most)
        if top_match and any(kw in query_lower for kw in ["claim", "paid", "billed"]):
            limit = int(top_match.group(1))
            sql = f"""
            SELECT MEMBER_ID, NAME, SUM(CLAIM_PAID_AMT) as TOTAL_PAID
            FROM HEALTHCARE_DB.MEMBER_SCHEMA.CALL_CENTER_MEMBER_DENORMALIZED
            WHERE CLAIM_ID IS NOT NULL
            GROUP BY MEMBER_ID, NAME
            ORDER BY TOTAL_PAID DESC
            LIMIT {limit}
            """
            result = self.session.sql(sql).collect()
            if result:
                members = [{"MEMBER_ID": row[0], "MEMBER_NAME": row[1], "TOTAL_PAID": str(row[2])} for row in result]
                return {"query_type": "general", "data": members}

        # Member names list
        if any(kw in query_lower for kw in ["names", "list members", "member names"]):
            sql = """
            SELECT DISTINCT MEMBER_ID, NAME
            FROM HEALTHCARE_DB.MEMBER_SCHEMA.CALL_CENTER_MEMBER_DENORMALIZED
            ORDER BY NAME
            LIMIT 50
            """
            result = self.session.sql(sql).collect()
            if result:
                members = [{"member_id": row[0], "name": row[1]} for row in result]
                return {"query_type": "member_list", "members": members, "count": len(members)}

        # Claims count
        if "claim" in query_lower and any(kw in query_lower for kw in count_keywords):
            sql = """
            SELECT
                COUNT(DISTINCT CLAIM_ID) as total_claims,
                SUM(CLAIM_BILL_AMT) as total_billed,
                SUM(CLAIM_PAID_AMT) as total_paid
            FROM HEALTHCARE_DB.MEMBER_SCHEMA.CALL_CENTER_MEMBER_DENORMALIZED
            WHERE CLAIM_ID IS NOT NULL
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

        # Default: general database stats
        sql = """
        SELECT
            COUNT(DISTINCT MEMBER_ID) as total_members,
            COUNT(DISTINCT CLAIM_ID) as total_claims,
            COUNT(DISTINCT PLAN_ID) as total_plans
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

        return {"query_type": "unknown", "error": "Could not determine query type"}

    def _detect_result_type(self, query: str, rows: list[dict]) -> str:
        """Detect result type based on query and columns.

        Args:
            query: Original natural language query.
            rows: Query result rows.

        Returns:
            Result type: 'member_info', 'claims', 'aggregate', or 'general'.
        """
        if not rows:
            return "general"

        columns = {k.upper() for k in rows[0]}

        # Member info detection
        if {"NAME", "DOB", "ADDRESS"}.intersection(columns) and "CLAIM_ID" not in columns:
            return "member_info"

        # Claims detection
        if {"CLAIM_ID", "CLAIM_STATUS"}.intersection(columns):
            return "claims"

        # Aggregate detection
        query_lower = query.lower()
        if any(kw in query_lower for kw in ["count", "total", "sum", "avg", "how many"]):
            return "aggregate"

        return "general"

    def _extract_member_info(self, rows: list[dict], result: dict, member_id: str | None) -> None:
        """Extract member personal info and coverage from results.

        Args:
            rows: Query result rows.
            result: Result dictionary to update.
            member_id: Optional member ID override.
        """
        if not rows:
            return

        first_row = rows[0]
        result["member_id"] = first_row.get("MEMBER_ID", first_row.get("member_id", member_id))

        # Extract personal information - check both UPPER and lower case keys
        # Helper to get value with fallbacks
        def get_val(*keys: str, default: Any = None) -> Any:
            for k in keys:
                if first_row.get(k) is not None:
                    return first_row[k]
            return default

        # Log available columns for debugging
        logger.info(f"Available columns in result: {list(first_row.keys())}")

        # Extract with all possible column name variations (raw DB, semantic model names)
        result["member_info"] = {
            "NAME": get_val("NAME", "name", "member_name", "MEMBER_NAME"),
            "DOB": str(get_val("DOB", "dob", "date_of_birth", "DATE_OF_BIRTH", default="")),
            "GENDER": get_val("GENDER", "gender"),
            "ADDRESS": get_val("ADDRESS", "address"),
            "PHONE": get_val("MEMBER_PHONE", "member_phone", "phone", "PHONE"),
            "SMOKER": get_val("SMOKER_IND", "smoker_indicator", "SMOKER_INDICATOR"),
            "LIFESTYLE": get_val("LIFESTYLE_INFO", "lifestyle_info", "LIFESTYLE_INFO"),
            "CHRONIC_CONDITION": get_val("CHRONIC_CONDITION", "chronic_condition"),
            "PCP": get_val("PCP", "pcp"),
            "PCP_PHONE": get_val("PCP_PHONE", "pcp_phone"),
        }

        # Extract coverage/plan information
        result["coverage"] = {
            "PLAN_ID": get_val("PLAN_ID", "plan_id"),
            "PLAN_NAME": get_val("PLAN_NAME", "plan_name"),
            "PLAN_TYPE": get_val("PLAN_TYPE", "plan_type"),
            "PREMIUM": get_val("PREMIUM", "premium"),
            "CVG_START_DATE": str(get_val("CVG_START_DATE", "cvg_start_date", default="")),
            "CVG_END_DATE": str(get_val("CVG_END_DATE", "cvg_end_date", default="")),
            "DEDUCTIBLE": get_val("DEDUCTIBLE", "deductible"),
            "COPAY_OFFICE": get_val("COPAY_OFFICE", "copay_office"),
            "COPAY_ER": get_val("COPAY_ER", "copay_er"),
        }

        # Extract grievance information if present
        grievance_id = get_val("GRIEVANCE_ID", "grievance_id")
        if grievance_id:
            grv_date = str(get_val("GRIEVANCE_DATE", "grievance_date", default=""))
            result["grievance"] = {
                "GRIEVANCE_ID": grievance_id,
                "GRIEVANCE_TYPE": get_val("GRIEVANCE_TYPE", "grievance_type"),
                "GRIEVANCE_STATUS": get_val("GRIEVANCE_STATUS", "grievance_status"),
                "GRIEVANCE_DATE": grv_date,
            }

    def _format_claims(self, rows: list[dict]) -> list[dict]:
        """Format claims results.

        Args:
            rows: Query result rows containing claims.

        Returns:
            List of formatted claim dictionaries.
        """
        claims = []
        for row in rows:
            service = row.get("CLAIM_SERVICE") or row.get("service_type") or row.get("SERVICE_TYPE")
            bill_amt = row.get("AMOUNT") or row.get("amount") or row.get("CLAIM_BILL_AMT")
            status = row.get("STATUS") or row.get("status") or row.get("CLAIM_STATUS")
            claims.append(
                {
                    "CLAIM_ID": row.get("CLAIM_ID", row.get("claim_id")),
                    "CLAIM_SERVICE": service,
                    "CLAIM_SERVICE_FROM_DATE": str(row.get("CLAIM_DATE", row.get("claim_date", ""))),
                    "CLAIM_BILL_AMT": bill_amt,
                    "CLAIM_STATUS": status,
                }
            )
        return claims

    def _format_aggregate(self, rows: list[dict]) -> dict:
        """Format aggregate query results.

        Args:
            rows: Query result rows.

        Returns:
            Formatted aggregate result dictionary.
        """
        if not rows:
            return {"query_type": "aggregate", "data": None}

        if len(rows) == 1:
            return {"query_type": "aggregate", **rows[0]}

        return {
            "query_type": "aggregate_grouped",
            "data": rows,
            "row_count": len(rows),
        }


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
            logger.info("Searching %s for: %s", source, query[:50])
            rows = self.session.sql(sql).collect()
        except Exception as exc:
            logger.warning("Search source %s failed: %s", source, exc)
            return []

        parsed_results: list[dict] = []
        for row in rows:
            row_dict = row.as_dict() if hasattr(row, "as_dict") else row
            result_data = row_dict.get("RESULTS", "{}")
            parsed_results.extend(_parse_search_result(result_data, source, limit))

        logger.info("Search %s returned %d results", source, len(parsed_results))
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
                tasks = [tg.create_task(self._search_source_async(source, query, limit)) for source in sources]

            for task in tasks:
                all_results.extend(task.result())

        except* Exception as eg:
            for exc in eg.exceptions:
                logger.warning("Search task failed: %s", exc)

        return all_results
