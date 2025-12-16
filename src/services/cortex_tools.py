"""Async wrappers for Snowflake Cortex services."""

import asyncio
import logging
from typing import Any

from snowflake.snowpark import Session

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

        def _query() -> list:
            if member_id:
                # Direct member lookup with joins
                sql = f"""
                SELECT m.*, c.*, cov.*
                FROM HEALTHCARE_DB.MEMBER_SCHEMA.MEMBERS m
                LEFT JOIN HEALTHCARE_DB.MEMBER_SCHEMA.CLAIMS c ON m.member_id = c.member_id
                LEFT JOIN HEALTHCARE_DB.MEMBER_SCHEMA.COVERAGE cov ON m.plan_id = cov.plan_id
                WHERE m.member_id = '{member_id}'
                """
            else:
                # Use Cortex COMPLETE for semantic query
                escaped_query = query.replace("'", "''")
                sql = f"""
                SELECT SNOWFLAKE.CORTEX.COMPLETE(
                    'llama3.1-70b',
                    'Based on healthcare member data, answer: {escaped_query}'
                ) AS response
                """
            return self.session.sql(sql).collect()

        try:
            result = await asyncio.to_thread(_query)
            return {
                "member_id": member_id,
                "claims": [dict(r.as_dict()) for r in result] if result else [],
                "coverage": {},
                "raw_response": str(result[0]) if result else None,
            }
        except Exception as exc:
            logger.error(f"Cortex Analyst error: {exc}", exc_info=True)
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
                escaped_query = query.replace("'", "''").replace('"', '\\"')
                sql = f"""
                SELECT *
                FROM TABLE(
                    SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
                        'HEALTHCARE_DB.KNOWLEDGE_SCHEMA.{source}',
                        '{{"query": "{escaped_query}", "columns": ["*"], "limit": {limit}}}'
                    )
                )
                """
                try:
                    return self.session.sql(sql).collect()
                except Exception as exc:
                    logger.warning(f"Search source {source} failed: {exc}")
                    return []

            results = await asyncio.to_thread(_search)
            return [
                {
                    "text": str(r.as_dict() if hasattr(r, "as_dict") else r),
                    "score": 0.9,  # Cortex Search doesn't return explicit scores
                    "source": source.lower().replace("_search", ""),
                    "metadata": {},
                }
                for r in results
            ]

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

