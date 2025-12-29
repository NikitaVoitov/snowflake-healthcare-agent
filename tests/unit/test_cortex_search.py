"""Unit tests for Cortex Search with SnowflakeCortexSearchRetriever."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document


class TestSearchServiceFactory:
    """Tests for get_cortex_search_retrievers factory."""

    @patch("src.services.search_service.SnowflakeCortexSearchRetriever")
    @patch("src.services.search_service.settings")
    def test_creates_three_retrievers(self, mock_settings: MagicMock, mock_retriever_class: MagicMock) -> None:
        """Should create retrievers for faqs, policies, and transcripts."""
        from src.services.search_service import get_cortex_search_retrievers

        mock_settings.snowflake_database = "HEALTHCARE_DB"
        mock_settings.cortex_search_limit = 5
        mock_session = MagicMock()

        retrievers = get_cortex_search_retrievers(mock_session)

        assert "faqs" in retrievers
        assert "policies" in retrievers
        assert "transcripts" in retrievers
        assert mock_retriever_class.call_count == 3

    @patch("src.services.search_service.SnowflakeCortexSearchRetriever")
    @patch("src.services.search_service.settings")
    def test_retriever_service_names(self, mock_settings: MagicMock, mock_retriever_class: MagicMock) -> None:
        """Should configure correct service names."""
        from src.services.search_service import get_cortex_search_retrievers

        mock_settings.snowflake_database = "HEALTHCARE_DB"
        mock_settings.cortex_search_limit = 5
        mock_session = MagicMock()

        get_cortex_search_retrievers(mock_session)

        # Check the calls for service names
        calls = mock_retriever_class.call_args_list
        service_names = [call.kwargs.get("service_name") for call in calls]

        assert "HEALTHCARE_DB.KNOWLEDGE_SCHEMA.FAQS_SEARCH" in service_names
        assert "HEALTHCARE_DB.KNOWLEDGE_SCHEMA.POLICIES_SEARCH" in service_names
        assert "HEALTHCARE_DB.KNOWLEDGE_SCHEMA.TRANSCRIPTS_SEARCH" in service_names

    @patch("src.services.search_service.SnowflakeCortexSearchRetriever")
    @patch("src.services.search_service.settings")
    def test_auto_format_disabled(self, mock_settings: MagicMock, mock_retriever_class: MagicMock) -> None:
        """Should disable auto_format_for_rag."""
        from src.services.search_service import get_cortex_search_retrievers

        mock_settings.snowflake_database = "HEALTHCARE_DB"
        mock_settings.cortex_search_limit = 5
        mock_session = MagicMock()

        get_cortex_search_retrievers(mock_session)

        # All retrievers should have auto_format_for_rag=False
        for call in mock_retriever_class.call_args_list:
            assert call.kwargs.get("auto_format_for_rag") is False


class TestAsyncCortexSearchTool:
    """Tests for refactored AsyncCortexSearchTool with retriever pattern."""

    @pytest.fixture
    def mock_session(self) -> MagicMock:
        """Create mock Snowpark session."""
        return MagicMock()

    @pytest.fixture
    def mock_retriever(self) -> AsyncMock:
        """Create mock SnowflakeCortexSearchRetriever."""
        retriever = AsyncMock()
        retriever.k = 5
        retriever.ainvoke.return_value = [
            Document(
                page_content="",
                metadata={
                    "faq_id": "FAQ001",
                    "question": "What is a copay?",
                    "answer": "A fixed amount you pay",
                    "@search_score": 0.95,
                },
            ),
            Document(
                page_content="",
                metadata={
                    "faq_id": "FAQ002",
                    "question": "How do I file a claim?",
                    "answer": "Submit through portal",
                    "@search_score": 0.87,
                },
            ),
        ]
        return retriever

    def test_source_to_key_conversion(self, mock_session: MagicMock) -> None:
        """Should convert source name to retriever key."""
        from src.services.cortex_tools import AsyncCortexSearchTool

        tool = AsyncCortexSearchTool(mock_session)

        assert tool._source_to_key("FAQS_SEARCH") == "faqs"
        assert tool._source_to_key("POLICIES_SEARCH") == "policies"
        assert tool._source_to_key("TRANSCRIPTS_SEARCH") == "transcripts"

    def test_convert_documents_to_results(self, mock_session: MagicMock) -> None:
        """Should convert Documents to expected dict format."""
        from src.services.cortex_tools import AsyncCortexSearchTool

        tool = AsyncCortexSearchTool(mock_session)
        docs = [
            Document(
                page_content="",
                metadata={
                    "faq_id": "FAQ001",
                    "question": "What is a copay?",
                    "answer": "A fixed amount",
                    "@search_score": 0.95,
                },
            )
        ]

        results = tool._convert_documents_to_results(docs, "FAQS_SEARCH")

        assert len(results) == 1
        assert results[0]["source"] == "faqs"
        # Score from _convert_documents_to_results uses default 0.9 when cosine_similarity is None
        # The @search_score is stored in metadata, not used directly for score field
        assert results[0]["score"] == 0.9  # Default score when cosine_similarity is None
        assert "faq_id: FAQ001" in results[0]["text"]
        # @search_score is now included in text for OTel observability parsing
        assert "@search_score: 0.95" in results[0]["text"]

    def test_convert_documents_default_score(self, mock_session: MagicMock) -> None:
        """Should use default score when @search_score missing."""
        from src.services.cortex_tools import AsyncCortexSearchTool

        tool = AsyncCortexSearchTool(mock_session)
        docs = [
            Document(
                page_content="",
                metadata={"faq_id": "FAQ001", "answer": "Test"},
            )
        ]

        results = tool._convert_documents_to_results(docs, "FAQS_SEARCH")

        assert results[0]["score"] == 0.9  # Default score

    @pytest.mark.asyncio
    @patch("src.services.cortex_tools.AsyncCortexSearchTool.retrievers", new_callable=lambda: property(lambda self: {}))
    async def test_search_unknown_source_returns_empty(self, mock_session: MagicMock) -> None:
        """Should return empty list for unknown source."""
        from src.services.cortex_tools import AsyncCortexSearchTool

        tool = AsyncCortexSearchTool(mock_session)
        tool._retrievers = {}  # Empty retrievers

        results = await tool._search_source_async("UNKNOWN_SEARCH", "test query", 5)

        assert results == []

    @pytest.mark.asyncio
    async def test_search_source_async_calls_retriever(self, mock_session: MagicMock, mock_retriever: AsyncMock) -> None:
        """Should call retriever.ainvoke with query."""
        from src.services.cortex_tools import AsyncCortexSearchTool

        tool = AsyncCortexSearchTool(mock_session)
        tool._retrievers = {"faqs": mock_retriever}

        results = await tool._search_source_async("FAQS_SEARCH", "test query", 5)

        mock_retriever.ainvoke.assert_called_once_with("test query")
        assert len(results) == 2
        assert results[0]["source"] == "faqs"

    @pytest.mark.asyncio
    async def test_search_source_updates_limit(self, mock_session: MagicMock, mock_retriever: AsyncMock) -> None:
        """Should update retriever.k when limit differs."""
        from src.services.cortex_tools import AsyncCortexSearchTool

        tool = AsyncCortexSearchTool(mock_session)
        mock_retriever.k = 5
        tool._retrievers = {"faqs": mock_retriever}

        await tool._search_source_async("FAQS_SEARCH", "test query", 10)

        assert mock_retriever.k == 10

    @pytest.mark.asyncio
    async def test_execute_parallel_search(self, mock_session: MagicMock) -> None:
        """Should execute parallel search across multiple sources."""
        from src.services.cortex_tools import AsyncCortexSearchTool

        # Create mock retrievers for each source
        faqs_retriever = AsyncMock()
        faqs_retriever.k = 5
        faqs_retriever.ainvoke.return_value = [Document(page_content="", metadata={"faq_id": "FAQ1", "@search_score": 0.9})]

        policies_retriever = AsyncMock()
        policies_retriever.k = 5
        policies_retriever.ainvoke.return_value = [Document(page_content="", metadata={"policy_id": "POL1", "@search_score": 0.85})]

        transcripts_retriever = AsyncMock()
        transcripts_retriever.k = 5
        transcripts_retriever.ainvoke.return_value = [Document(page_content="", metadata={"transcript_id": "TR1", "@search_score": 0.8})]

        tool = AsyncCortexSearchTool(mock_session)
        tool._retrievers = {
            "faqs": faqs_retriever,
            "policies": policies_retriever,
            "transcripts": transcripts_retriever,
        }

        result = await tool.execute("test query")

        # Result is now a dict with 'results' and 'cortex_search_metadata'
        assert "results" in result
        assert "cortex_search_metadata" in result

        results = result["results"]
        # Should have results from all sources
        assert len(results) == 3
        sources = {r["source"] for r in results}
        assert sources == {"faqs", "policies", "transcripts"}

        # All retrievers should be called
        faqs_retriever.ainvoke.assert_called_once()
        policies_retriever.ainvoke.assert_called_once()
        transcripts_retriever.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_handles_retriever_error(self, mock_session: MagicMock) -> None:
        """Should handle errors from individual retrievers gracefully."""
        from src.services.cortex_tools import AsyncCortexSearchTool

        # One retriever fails
        faqs_retriever = AsyncMock()
        faqs_retriever.k = 5
        faqs_retriever.ainvoke.side_effect = Exception("API Error")

        # Other retriever succeeds
        policies_retriever = AsyncMock()
        policies_retriever.k = 5
        policies_retriever.ainvoke.return_value = [Document(page_content="", metadata={"policy_id": "POL1", "@search_score": 0.85})]

        tool = AsyncCortexSearchTool(mock_session)
        tool._retrievers = {
            "faqs": faqs_retriever,
            "policies": policies_retriever,
        }

        # Should not raise, should return partial results
        result = await tool.execute("test query", sources=["FAQS_SEARCH", "POLICIES_SEARCH"])

        # Result is now a dict with 'results' and 'cortex_search_metadata'
        results = result["results"]

        # Should have result from successful retriever only
        assert len(results) == 1
        assert results[0]["source"] == "policies"

    @pytest.mark.asyncio
    async def test_execute_default_sources(self, mock_session: MagicMock) -> None:
        """Should use all three sources by default."""
        from src.services.cortex_tools import AsyncCortexSearchTool

        tool = AsyncCortexSearchTool(mock_session)

        # Create simple mock retrievers
        for key in ["faqs", "policies", "transcripts"]:
            retriever = AsyncMock()
            retriever.k = 5
            retriever.ainvoke.return_value = []
            if tool._retrievers is None:
                tool._retrievers = {}
            tool._retrievers[key] = retriever

        await tool.execute("test query")

        # All three should be called
        for key in ["faqs", "policies", "transcripts"]:
            tool._retrievers[key].ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_custom_sources(self, mock_session: MagicMock) -> None:
        """Should only search specified sources."""
        from src.services.cortex_tools import AsyncCortexSearchTool

        faqs_retriever = AsyncMock()
        faqs_retriever.k = 5
        faqs_retriever.ainvoke.return_value = []

        transcripts_retriever = AsyncMock()
        transcripts_retriever.k = 5
        transcripts_retriever.ainvoke.return_value = []

        tool = AsyncCortexSearchTool(mock_session)
        tool._retrievers = {
            "faqs": faqs_retriever,
            "transcripts": transcripts_retriever,
        }

        await tool.execute("test query", sources=["FAQS_SEARCH"])

        # Only FAQs should be searched
        faqs_retriever.ainvoke.assert_called_once()
        transcripts_retriever.ainvoke.assert_not_called()


class TestCortexSearchIntegration:
    """Integration-style tests for full search flow."""

    @pytest.mark.asyncio
    @patch("src.services.search_service.SnowflakeCortexSearchRetriever")
    @patch("src.services.search_service.settings")
    async def test_full_search_flow(self, mock_settings: MagicMock, mock_retriever_class: MagicMock) -> None:
        """Test complete search flow from tool to results."""
        from src.services.cortex_tools import AsyncCortexSearchTool

        # Setup
        mock_settings.snowflake_database = "HEALTHCARE_DB"
        mock_settings.cortex_search_limit = 5

        # Mock retriever instance
        mock_retriever_instance = AsyncMock()
        mock_retriever_instance.k = 5
        mock_retriever_instance.ainvoke.return_value = [
            Document(
                page_content="",
                metadata={
                    "faq_id": "FAQ001",
                    "question": "How to file appeal?",
                    "answer": "Submit form to appeals dept",
                    "category": "appeals",
                    "@search_score": 0.92,
                },
            )
        ]
        mock_retriever_class.return_value = mock_retriever_instance

        # Execute
        mock_session = MagicMock()
        tool = AsyncCortexSearchTool(mock_session)

        # Force retriever initialization by accessing property
        _ = tool.retrievers

        execute_result = await tool.execute("How do I file an appeal?", sources=["FAQS_SEARCH"])

        # Result is now a dict with 'results' and 'cortex_search_metadata'
        assert "results" in execute_result
        assert "cortex_search_metadata" in execute_result

        results = execute_result["results"]

        # Verify
        assert len(results) == 1
        result = results[0]
        assert result["source"] == "faqs"
        # Score uses default 0.9 when cosine_similarity is None
        assert result["score"] == 0.9
        assert "How to file appeal" in result["text"]
        assert "Submit form" in result["text"]
