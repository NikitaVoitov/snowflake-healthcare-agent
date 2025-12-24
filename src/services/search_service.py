"""Factory for SnowflakeCortexSearchRetriever instances.

Provides configured retrievers for each Cortex Search service:
- FAQs: Healthcare FAQ documents
- Policies: Policy documents
- Transcripts: Call transcripts with summaries

Uses langchain-snowflake's REST API-based retriever instead of
SEARCH_PREVIEW SQL function for production-ready performance.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_snowflake import SnowflakeCortexSearchRetriever

from src.config import settings

if TYPE_CHECKING:
    from snowflake.snowpark import Session

logger = logging.getLogger(__name__)


def get_cortex_search_retrievers(
    session: Session,
) -> dict[str, SnowflakeCortexSearchRetriever]:
    """Create retrievers for each Cortex Search service.

    Args:
        session: Active Snowpark session for Snowflake queries.

    Returns:
        Dict keyed by lowercase service type: {"faqs": ..., "policies": ..., "transcripts": ...}
    """
    database = settings.snowflake_database
    schema = "KNOWLEDGE_SCHEMA"
    k = settings.cortex_search_limit

    retrievers = {
        "faqs": SnowflakeCortexSearchRetriever(
            service_name=f"{database}.{schema}.FAQS_SEARCH",
            session=session,
            database=database,
            schema=schema,  # Required by SnowflakeConnectionMixin
            k=k,
            search_columns=["faq_id", "question", "answer", "category"],
            auto_format_for_rag=False,  # We format ourselves
        ),
        "policies": SnowflakeCortexSearchRetriever(
            service_name=f"{database}.{schema}.POLICIES_SEARCH",
            session=session,
            database=database,
            schema=schema,  # Required by SnowflakeConnectionMixin
            k=k,
            search_columns=["policy_id", "policy_name", "content"],
            auto_format_for_rag=False,
        ),
        "transcripts": SnowflakeCortexSearchRetriever(
            service_name=f"{database}.{schema}.TRANSCRIPTS_SEARCH",
            session=session,
            database=database,
            schema=schema,  # Required by SnowflakeConnectionMixin
            k=k,
            search_columns=["transcript_id", "member_id", "summary", "transcript_text"],
            auto_format_for_rag=False,
        ),
    }

    logger.info("Created %d Cortex Search retrievers (REST API mode)", len(retrievers))
    return retrievers
