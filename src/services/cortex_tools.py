"""Async wrappers for Snowflake Cortex services."""

import asyncio
import logging
from typing import Any

from snowflake.snowpark import Session

from src.config import settings

logger = logging.getLogger(__name__)


class AsyncCortexAnalystTool:
    """Async wrapper for Cortex Analyst queries against member data."""

    def __init__(self, session: Session) -> None:
        """Initialize with Snowpark session."""
        self.session = session

    async def execute(self, query: str, member_id: str | None = None) -> dict[str, Any]:
        """Execute Cortex Analyst query asynchronously.

        Args:
            query: Natural language query about member data
            member_id: Optional member ID for filtered queries

        Returns:
            Dictionary with member_id, claims, coverage, and raw_response
        """

        def _query() -> dict:
            # Ensure warehouse is set
            self.session.sql(f"USE WAREHOUSE {settings.snowflake_warehouse}").collect()

            if member_id:
                # Query the DENORMALIZED table which has ALL member data in one place
                # Including: MEMBER_ID, NAME, DOB, GENDER, ADDRESS, MEMBER_PHONE, PLAN_ID, PLAN_NAME,
                # PCP, PCP_PHONE, CLAIM_ID, CLAIM_SERVICE, CLAIM_STATUS, CLAIM_BILL_AMT, etc.
                member_sql = f"""
                SELECT DISTINCT
                    member_id, name, dob, gender, address, member_phone,
                    plan_id, plan_name, plan_type, premium,
                    pcp, pcp_phone
                FROM HEALTHCARE_DB.MEMBER_SCHEMA.CALL_CENTER_MEMBER_DENORMALIZED
                WHERE member_id = '{member_id}'
                LIMIT 1
                """
                # Get claims from same denormalized table
                claims_sql = f"""
                SELECT DISTINCT
                    claim_id, claim_service_from_date, claim_service, 
                    claim_bill_amt, claim_paid_amt, claim_status
                FROM HEALTHCARE_DB.MEMBER_SCHEMA.CALL_CENTER_MEMBER_DENORMALIZED
                WHERE member_id = '{member_id}' AND claim_id IS NOT NULL
                LIMIT 5
                """
                # Coverage info from same table
                coverage_sql = f"""
                SELECT DISTINCT
                    plan_id, plan_name, plan_type, premium
                FROM HEALTHCARE_DB.MEMBER_SCHEMA.CALL_CENTER_MEMBER_DENORMALIZED
                WHERE member_id = '{member_id}'
                LIMIT 1
                """
                logger.debug(f"Executing member query: {member_sql}")
                member_rows = self.session.sql(member_sql).collect()
                logger.debug(f"Executing claims query: {claims_sql}")
                claims_rows = self.session.sql(claims_sql).collect()
                logger.debug(f"Executing coverage query: {coverage_sql}")
                coverage_rows = self.session.sql(coverage_sql).collect()

                return {
                    "member": [dict(zip(r._fields, r)) for r in member_rows] if member_rows else [],
                    "claims": [dict(zip(r._fields, r)) for r in claims_rows] if claims_rows else [],
                    "coverage": [dict(zip(r._fields, r)) for r in coverage_rows]
                    if coverage_rows
                    else [],
                }
            else:
                # Use Cortex COMPLETE for semantic query
                escaped_query = query.replace("'", "''")
                sql = f"""
                SELECT SNOWFLAKE.CORTEX.COMPLETE(
                    'llama3.1-70b',
                    'Based on healthcare member data, answer: {escaped_query}'
                ) AS response
                """
                result = self.session.sql(sql).collect()
                return {"response": str(result[0][0]) if result else None}

        try:
            result = await asyncio.to_thread(_query)
            if member_id:
                return {
                    "member_id": member_id,
                    "claims": result.get("claims", []),
                    "coverage": result.get("coverage", [{}])[0] if result.get("coverage") else {},
                    "raw_response": str(result.get("member", [])),
                }
            else:
                return {
                    "member_id": None,
                    "claims": [],
                    "coverage": {},
                    "raw_response": result.get("response"),
                }
        except Exception as exc:
            # Log the exact SQL that failed for debugging
            logger.error(f"Cortex Analyst error: {exc}")
            logger.error(f"Query was for member_id: {member_id}")
            logger.error(f"Database context: {settings.snowflake_database}")
            raise


class AsyncCortexSearchTool:
    """Async wrapper for Cortex Search queries across knowledge base."""

    def __init__(self, session: Session) -> None:
        """Initialize with Snowpark session."""
        self.session = session

    async def execute(
        self,
        query: str,
        sources: list[str] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Execute Cortex Search query asynchronously.

        Args:
            query: Natural language search query
            sources: List of search services to query (default: all)
            limit: Maximum results per source

        Returns:
            List of search results with text, score, source, metadata
        """
        sources = sources or ["FAQS_SEARCH", "POLICIES_SEARCH", "TRANSCRIPTS_SEARCH"]

        async def search_source(source: str) -> list[dict]:
            def _search() -> list:
                # Ensure warehouse is set
                self.session.sql(f"USE WAREHOUSE {settings.snowflake_warehouse}").collect()

                escaped_query = query.replace("'", "''").replace('"', '\\"')

                # Map source to appropriate return columns based on service
                if source == "FAQS_SEARCH":
                    columns = '["faq_id", "question", "answer"]'
                elif source == "POLICIES_SEARCH":
                    columns = '["policy_id", "policy_name", "content"]'
                elif source == "TRANSCRIPTS_SEARCH":
                    columns = '["transcript_id", "member_id", "summary"]'
                else:
                    columns = "[]"

                sql = f"""
                SELECT SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
                    'HEALTHCARE_DB.KNOWLEDGE_SCHEMA.{source}',
                    '{{"query": "{escaped_query}", "columns": {columns}, "limit": {limit}}}'
                ) AS results
                """
                try:
                    return self.session.sql(sql).collect()
                except Exception as exc:
                    logger.warning(f"Search source {source} failed: {exc}")
                    return []

            results = await asyncio.to_thread(_search)
            parsed_results = []
            for r in results:
                # Extract the JSON result from the row
                row_dict = r.as_dict() if hasattr(r, "as_dict") else r
                result_data = row_dict.get("RESULTS", "{}")

                # Parse JSON and extract individual results
                import json

                try:
                    if isinstance(result_data, str):
                        json_data = json.loads(result_data)
                    else:
                        json_data = result_data

                    # Extract results array from Cortex Search response
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
                    # Fallback: truncate raw result
                    parsed_results.append(
                        {
                            "text": str(result_data)[:5000],
                            "score": 0.9,
                            "source": source.lower().replace("_search", ""),
                            "metadata": {},
                        }
                    )
            return parsed_results

        # Search all sources in parallel using TaskGroup
        all_results: list[dict] = []
        try:
            async with asyncio.TaskGroup() as tg:
                tasks = [tg.create_task(search_source(s)) for s in sources]
            for task in tasks:
                all_results.extend(task.result())
        except* Exception as eg:
            for exc in eg.exceptions:
                logger.warning(f"Search task failed: {exc}")

        return all_results
