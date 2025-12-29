"""Async wrappers for Snowflake Cortex services.

This module provides async tools for:
- Cortex Analyst: NL→SQL conversion using langchain-snowflake's SnowflakeCortexAnalyst
- Cortex Search: Semantic search across knowledge base documents

The Cortex Analyst tool uses langchain-snowflake's SnowflakeCortexAnalyst for
the REST API call, then executes the generated SQL via Snowpark and post-processes
the results into structured healthcare data.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from snowflake.snowpark import Session
from snowflake.snowpark.row import Row

from src.config import settings
from src.services.analyst_service import get_snowflake_analyst

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


# =============================================================================
# Cortex Analyst Tool (using langchain-snowflake)
# =============================================================================


class AsyncCortexAnalystTool:
    """Async wrapper for Cortex Analyst queries using langchain-snowflake.

    Uses SnowflakeCortexAnalyst from langchain-snowflake for NL→SQL conversion,
    then executes the generated SQL via Snowpark and post-processes results
    into structured healthcare data.

    In local development (without OAuth), falls back to direct Snowpark
    SQL queries against the denormalized member table.
    """

    def __init__(self, session: Session) -> None:
        """Initialize with Snowpark session.

        Args:
            session: Active Snowpark Session for Snowflake queries.
        """
        self.session = session

    def _extract_sql_from_response(self, parsed: dict) -> tuple[str | None, str | None, list[str], dict[str, Any] | None]:
        """Extract SQL, text explanation, suggestions, and verified query info from response.

        Args:
            parsed: Parsed JSON response from SnowflakeCortexAnalyst._arun()

        Returns:
            Tuple of (sql_statement, text_explanation, suggestions, verified_query_used)
        """
        sql = None
        text = None
        suggestions: list[str] = []
        verified_query_used: dict[str, Any] | None = None

        for item in parsed.get("content", []):
            item_type = item.get("type")
            if item_type == "sql":
                sql = item.get("statement")
                # Extract verified_query_used from confidence object if present
                confidence = item.get("confidence", {})
                if confidence and "verified_query_used" in confidence:
                    verified_query_used = confidence["verified_query_used"]
            elif item_type == "text":
                text = item.get("text")
            elif item_type == "suggestions":
                suggestions = item.get("suggestions", [])

        return sql, text, suggestions, verified_query_used

    async def execute(self, query: str, member_id: str | None = None) -> dict[str, Any]:
        """Execute natural language query using Cortex Analyst.

        Uses SnowflakeCortexAnalyst from langchain-snowflake for NL→SQL conversion,
        then executes the SQL and post-processes results.

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
        except Exception as exc:
            # If langchain-snowflake fails, fall back to direct Snowpark queries
            error_msg = str(exc).lower()
            if any(kw in error_msg for kw in ["authentication", "token", "unauthorized", "401"]):
                logger.warning("Cortex Analyst auth failed, using Snowpark fallback: %s", exc)
                return await self._execute_via_snowpark(query, member_id)
            raise

    async def _execute_via_cortex_analyst_api(self, query: str, member_id: str | None = None) -> dict[str, Any]:
        """Execute query using SnowflakeCortexAnalyst from langchain-snowflake.

        Args:
            query: Natural language query.
            member_id: Optional member ID for context.

        Returns:
            Structured query results.
        """
        # Enhance query with member context if provided
        enhanced_query = query
        if member_id:
            enhanced_query = f"For member ID {member_id}: {query}"

        logger.info(f"Cortex Analyst query: {enhanced_query[:100]}...")

        # Get SnowflakeCortexAnalyst instance from factory
        analyst = get_snowflake_analyst(self.session)

        # Call SnowflakeCortexAnalyst._arun() - returns JSON string
        json_result = await analyst._arun(enhanced_query)

        # Parse JSON response
        try:
            parsed = json.loads(json_result)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Cortex Analyst response: {e}")
            return {
                "member_id": member_id,
                "error": f"Invalid response format: {e}",
                "raw_response": json_result[:500] if json_result else None,
            }

        # Extract SQL, text, suggestions, and verified query info from response
        sql, interpretation, suggestions, verified_query_used = self._extract_sql_from_response(parsed)

        # Extract metadata for observability
        metadata = parsed.get("metadata", {})
        model_names = metadata.get("model_names", [])
        question_category = metadata.get("question_category")

        # Determine semantic model type based on configuration
        semantic_model = settings.cortex_analyst_semantic_model
        semantic_model_type = "FILE_ON_STAGE" if semantic_model.startswith("@") else "SEMANTIC_VIEW"

        # Build result structure with full observability metadata
        result: dict[str, Any] = {
            "member_id": member_id,
            "member_info": {},
            "claims": [],
            "coverage": {},
            "grievance": {},
            "raw_response": None,
            "aggregate_result": None,
            "sql": sql,
            "interpretation": interpretation,
            "suggestions": suggestions,
            "request_id": parsed.get("request_id"),
            "warnings": parsed.get("warnings", []),
            # Cortex Analyst observability metadata
            "cortex_analyst_metadata": {
                "request_id": parsed.get("request_id"),
                "model_names": model_names,
                "question_category": question_category,
                "verified_query_used": verified_query_used,
                "semantic_model": {
                    "name": semantic_model,
                    "type": semantic_model_type,
                },
                "snowflake": {
                    "database": settings.snowflake_database,
                    "schema": settings.snowflake_schema,
                    "warehouse": settings.snowflake_warehouse,
                },
            },
        }

        # Handle suggestions (ambiguous query)
        if suggestions and not sql:
            result["error"] = "Question was ambiguous. See suggestions for alternatives."
            logger.info(f"Cortex Analyst returned suggestions: {suggestions}")
            return result

        # If no SQL was generated, return error
        if not sql:
            result["error"] = "Could not generate SQL for this question."
            logger.warning("No SQL generated by Cortex Analyst")
            return result

        # Execute the generated SQL and process results
        return await self._execute_sql_and_process(sql, query, member_id, result)

    async def _execute_sql_and_process(
        self,
        sql: str,
        original_query: str,
        member_id: str | None,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute generated SQL via Snowpark and process results.

        Args:
            sql: SQL statement to execute.
            original_query: Original natural language query (for result type detection).
            member_id: Optional member ID.
            result: Result dictionary to update.

        Returns:
            Updated result dictionary with query results.
        """
        try:

            def _execute_sql() -> list[dict]:
                """Execute SQL synchronously (for thread pool)."""
                # Ensure warehouse is set
                self.session.sql(f"USE WAREHOUSE {settings.snowflake_warehouse}").collect()
                rows = self.session.sql(sql).collect()
                # Convert rows to dictionaries
                results = []
                for row in rows:
                    if hasattr(row, "as_dict"):
                        results.append(row.as_dict())
                    else:
                        # Fallback for older Snowpark versions
                        results.append(dict(zip(row._fields, row, strict=True)))
                return results

            rows = await asyncio.to_thread(_execute_sql)
            logger.info(f"Executed SQL successfully, returned {len(rows)} rows")

            # Process query results
            if rows:
                result["raw_response"] = str(rows)
                result_type = self._detect_result_type(original_query, rows)

                if result_type == "member_info":
                    self._extract_member_info(rows, result, member_id)
                elif result_type == "claims":
                    result["claims"] = self._format_claims(rows)
                elif result_type == "aggregate":
                    result["aggregate_result"] = self._format_aggregate(rows)
                else:
                    result["aggregate_result"] = {"query_type": "general", "data": rows}

        except Exception as exc:
            logger.error(f"SQL execution failed: {exc}")
            result["error"] = f"SQL execution error: {exc!s}"

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
# Cortex Search Tool (using langchain-snowflake)
# =============================================================================


class AsyncCortexSearchTool:
    """Parallel search using langchain-snowflake SnowflakeCortexSearchRetriever.

    Searches across FAQs, policies, and transcripts using Snowflake
    Cortex Search REST API with parallel execution via asyncio.TaskGroup.

    Uses REST API instead of SEARCH_PREVIEW SQL function for:
    - Production-ready performance (no 300KB limit)
    - Native async support (aiohttp)
    - Parameterized queries (no SQL injection risk)
    """

    def __init__(self, session: Session) -> None:
        """Initialize with Snowpark session.

        Args:
            session: Active Snowpark Session for Snowflake queries.
        """
        self.session = session
        self._retrievers: dict | None = None

    @property
    def retrievers(self) -> dict:
        """Lazy-load retrievers on first access."""
        if self._retrievers is None:
            from src.services.search_service import get_cortex_search_retrievers

            self._retrievers = get_cortex_search_retrievers(self.session)
        return self._retrievers

    def _source_to_key(self, source: str) -> str:
        """Convert source name to retriever key.

        Args:
            source: Service name like "FAQS_SEARCH".

        Returns:
            Lowercase key like "faqs".
        """
        return source.replace("_SEARCH", "").lower()

    def _convert_documents_to_results(self, docs: list, source: str) -> list[dict]:
        """Convert LangChain Documents to our result dict format.

        Args:
            docs: List of Document objects from retriever.
            source: Original source name for labeling.

        Returns:
            List of dicts with text, score, source, metadata.
            Metadata includes @scores (cosine_similarity, text_match) and _snowflake_request_id.
        """
        results = []
        # Keys to exclude from text representation (metadata-only fields)
        exclude_keys = {
            "@scores",  # Cortex Search scoring metadata
            "_formatted_for_rag",
            "_original_page_content",
            "_content_field_used",
            "_snowflake_request_id",  # Snowflake request tracking ID
        }

        for doc in docs:
            # Combine metadata fields into text (same format as before)
            text_parts = []
            for key, val in doc.metadata.items():
                if key not in exclude_keys and val:
                    text_parts.append(f"{key}: {val}")
            text = " | ".join(text_parts)[:5000]

            # Extract @scores properly (Snowflake returns @scores dict, not @search_score)
            scores = doc.metadata.get("@scores", {})
            cosine_similarity = scores.get("cosine_similarity") if isinstance(scores, dict) else None
            text_match = scores.get("text_match") if isinstance(scores, dict) else None

            # Use cosine_similarity as primary score, fallback to 0.9
            primary_score = cosine_similarity if cosine_similarity is not None else 0.9

            results.append(
                {
                    "text": text,
                    "score": primary_score,
                    "cosine_similarity": cosine_similarity,
                    "text_match": text_match,
                    "source": self._source_to_key(source),  # "faqs", "policies", "transcripts"
                    "metadata": doc.metadata,
                }
            )
        return results

    async def _search_source_async(self, source: str, query: str, limit: int) -> list[dict]:
        """Search single source using SnowflakeCortexSearchRetriever.

        Args:
            source: Search service name (e.g., FAQS_SEARCH).
            query: Search query text.
            limit: Maximum results per source.

        Returns:
            List of search results for this source.
        """
        key = self._source_to_key(source)
        retriever = self.retrievers.get(key)

        if not retriever:
            logger.warning("Unknown search source: %s", source)
            return []

        # Update k if different limit requested
        if retriever.k != limit:
            retriever.k = limit

        try:
            logger.info("Searching %s (REST API) for: %s", source, query[:50])
            docs = await retriever.ainvoke(query)
            results = self._convert_documents_to_results(docs, source)
            logger.info("Search %s returned %d results", source, len(results))
            return results
        except Exception as e:
            logger.warning("Search %s failed: %s", source, e)
            return []

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
