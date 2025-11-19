"""Unit tests for wizard orchestration functions."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from logsqueak.wizard.wizard import offer_graph_indexing, index_graph_after_setup


class TestOfferGraphIndexing:
    """Tests for offer_graph_indexing function."""

    @pytest.mark.asyncio
    @patch("logseq_outline.graph.GraphPaths")
    @patch("logsqueak.services.page_indexer.PageIndexer")
    @patch("logsqueak.services.rag_search.RAGSearch")
    @patch("logsqueak.wizard.wizard.prompt_index_graph")
    @patch("logsqueak.wizard.wizard.index_graph_after_setup")
    async def test_calls_index_when_user_accepts(self, mock_index, mock_prompt, mock_rag, mock_indexer, mock_graph_paths):
        """Test that indexing is triggered when user accepts."""
        # Setup: Index doesn't exist
        mock_rag_instance = MagicMock()
        mock_rag_instance.has_indexed_data.return_value = False
        mock_rag.return_value = mock_rag_instance

        mock_prompt.return_value = True
        mock_index.return_value = True

        await offer_graph_indexing("/path/to/graph")

        mock_prompt.assert_called_once()
        mock_index.assert_called_once_with("/path/to/graph")

    @pytest.mark.asyncio
    @patch("logseq_outline.graph.GraphPaths")
    @patch("logsqueak.services.page_indexer.PageIndexer")
    @patch("logsqueak.services.rag_search.RAGSearch")
    @patch("logsqueak.wizard.wizard.prompt_index_graph")
    @patch("logsqueak.wizard.wizard.index_graph_after_setup")
    async def test_skips_index_when_user_declines(self, mock_index, mock_prompt, mock_rag, mock_indexer, mock_graph_paths):
        """Test that indexing is skipped when user declines."""
        # Setup: Index doesn't exist
        mock_rag_instance = MagicMock()
        mock_rag_instance.has_indexed_data.return_value = False
        mock_rag.return_value = mock_rag_instance

        mock_prompt.return_value = False

        await offer_graph_indexing("/path/to/graph")

        mock_prompt.assert_called_once()
        mock_index.assert_not_called()

    @pytest.mark.asyncio
    @patch("logseq_outline.graph.GraphPaths")
    @patch("logsqueak.services.page_indexer.PageIndexer")
    @patch("logsqueak.services.rag_search.RAGSearch")
    @patch("logsqueak.wizard.wizard.prompt_index_graph")
    @patch("logsqueak.wizard.wizard.index_graph_after_setup")
    @patch("logsqueak.wizard.wizard.rprint")
    async def test_shows_skip_message_when_declined(self, mock_rprint, mock_index, mock_prompt, mock_rag, mock_indexer, mock_graph_paths):
        """Test that helpful message is shown when user declines."""
        # Setup: Index doesn't exist
        mock_rag_instance = MagicMock()
        mock_rag_instance.has_indexed_data.return_value = False
        mock_rag.return_value = mock_rag_instance

        mock_prompt.return_value = False

        await offer_graph_indexing("/path/to/graph")

        # Verify informative message was shown
        assert mock_rprint.call_count >= 2
        calls_text = " ".join(str(call) for call in mock_rprint.call_args_list)
        assert "later" in calls_text.lower()
        assert "search" in calls_text.lower()

    @pytest.mark.asyncio
    @patch("logseq_outline.graph.GraphPaths")
    @patch("logsqueak.services.page_indexer.PageIndexer")
    @patch("logsqueak.services.rag_search.RAGSearch")
    @patch("logsqueak.wizard.wizard.prompt_index_graph")
    @patch("logsqueak.wizard.wizard.index_graph_after_setup")
    async def test_skips_prompt_when_index_exists(self, mock_index, mock_prompt, mock_rag, mock_indexer, mock_graph_paths):
        """Test that prompt is skipped when index already exists."""
        # Setup: Index already exists
        mock_rag_instance = MagicMock()
        mock_rag_instance.has_indexed_data.return_value = True
        mock_rag.return_value = mock_rag_instance

        await offer_graph_indexing("/path/to/graph")

        # Should not prompt or index
        mock_prompt.assert_not_called()
        mock_index.assert_not_called()

    @pytest.mark.asyncio
    @patch("logseq_outline.graph.GraphPaths", side_effect=Exception("Graph error"))
    @patch("logsqueak.wizard.wizard.prompt_index_graph")
    @patch("logsqueak.wizard.wizard.index_graph_after_setup")
    async def test_skips_prompt_on_check_error(self, mock_index, mock_prompt, mock_graph_paths):
        """Test that prompt is skipped if index check fails."""
        await offer_graph_indexing("/path/to/graph")

        # Should not prompt or index (safe default)
        mock_prompt.assert_not_called()
        mock_index.assert_not_called()


class TestIndexGraphAfterSetup:
    """Tests for index_graph_after_setup function."""

    @pytest.mark.asyncio
    @patch("logseq_outline.graph.GraphPaths")
    @patch("logsqueak.services.page_indexer.PageIndexer")
    @patch("logsqueak.services.rag_search.RAGSearch")
    @patch("logsqueak.utils.index_progress.create_index_progress_callback")
    async def test_successful_initial_indexing(self, mock_progress, mock_rag, mock_indexer_class, mock_graph_paths):
        """Test successful initial graph indexing (no existing data)."""
        # Setup mocks
        mock_indexer = MagicMock()
        mock_indexer.build_index = AsyncMock(return_value=10)  # 10 pages indexed
        mock_indexer_class.return_value = mock_indexer

        mock_rag_instance = MagicMock()
        mock_rag_instance.has_indexed_data.return_value = False  # No existing data
        mock_rag.return_value = mock_rag_instance

        mock_callback = MagicMock()
        mock_cleanup = MagicMock()
        mock_progress.return_value = (mock_callback, mock_cleanup)

        # Execute
        result = await index_graph_after_setup("/path/to/graph")

        # Verify
        assert result is True
        mock_indexer.build_index.assert_called_once_with(progress_callback=mock_callback)
        mock_cleanup.assert_called_once()
        mock_rag_instance.has_indexed_data.assert_called_once()

    @pytest.mark.asyncio
    @patch("logseq_outline.graph.GraphPaths")
    @patch("logsqueak.services.page_indexer.PageIndexer")
    @patch("logsqueak.services.rag_search.RAGSearch")
    @patch("logsqueak.utils.index_progress.create_index_progress_callback")
    async def test_successful_update_indexing(self, mock_progress, mock_rag, mock_indexer_class, mock_graph_paths):
        """Test successful index update (existing data)."""
        # Setup mocks
        mock_indexer = MagicMock()
        mock_indexer.build_index = AsyncMock(return_value=3)  # 3 pages updated
        mock_indexer_class.return_value = mock_indexer

        mock_rag_instance = MagicMock()
        mock_rag_instance.has_indexed_data.return_value = True  # Existing data
        mock_rag.return_value = mock_rag_instance

        mock_callback = MagicMock()
        mock_cleanup = MagicMock()
        mock_progress.return_value = (mock_callback, mock_cleanup)

        # Execute
        result = await index_graph_after_setup("/path/to/graph")

        # Verify
        assert result is True
        mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    @patch("logseq_outline.graph.GraphPaths")
    @patch("logsqueak.services.page_indexer.PageIndexer")
    @patch("logsqueak.services.rag_search.RAGSearch")
    @patch("logsqueak.utils.index_progress.create_index_progress_callback")
    async def test_no_pages_need_indexing(self, mock_progress, mock_rag, mock_indexer_class, mock_graph_paths):
        """Test when all pages are already up-to-date."""
        # Setup mocks
        mock_indexer = MagicMock()
        mock_indexer.build_index = AsyncMock(return_value=0)  # 0 pages indexed
        mock_indexer_class.return_value = mock_indexer

        mock_rag_instance = MagicMock()
        mock_rag_instance.has_indexed_data.return_value = True
        mock_rag.return_value = mock_rag_instance

        mock_callback = MagicMock()
        mock_cleanup = MagicMock()
        mock_progress.return_value = (mock_callback, mock_cleanup)

        # Execute
        result = await index_graph_after_setup("/path/to/graph")

        # Verify - should still return True (success)
        assert result is True
        mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    @patch("logseq_outline.graph.GraphPaths")
    @patch("logsqueak.services.page_indexer.PageIndexer")
    @patch("logsqueak.services.rag_search.RAGSearch")
    @patch("logsqueak.utils.index_progress.create_index_progress_callback")
    async def test_handles_indexing_error(self, mock_progress, mock_rag, mock_indexer_class, mock_graph_paths):
        """Test error handling during indexing."""
        # Setup mocks
        mock_indexer = MagicMock()
        mock_indexer.build_index = AsyncMock(side_effect=Exception("Index error"))
        mock_indexer_class.return_value = mock_indexer

        mock_rag_instance = MagicMock()
        mock_rag_instance.has_indexed_data.return_value = False
        mock_rag.return_value = mock_rag_instance

        mock_callback = MagicMock()
        mock_cleanup = MagicMock()
        mock_progress.return_value = (mock_callback, mock_cleanup)

        # Execute
        with patch("logsqueak.wizard.wizard.rprint") as mock_rprint:
            result = await index_graph_after_setup("/path/to/graph")

        # Verify
        assert result is False
        mock_cleanup.assert_called_once()  # Cleanup should still be called
        # Verify error message was displayed
        calls_text = " ".join(str(call) for call in mock_rprint.call_args_list)
        assert "failed" in calls_text.lower() or "error" in calls_text.lower()

    @pytest.mark.asyncio
    @patch("logseq_outline.graph.GraphPaths", side_effect=Exception("Invalid graph path"))
    async def test_handles_initialization_error(self, mock_graph_paths):
        """Test error handling during service initialization."""
        # Execute
        with patch("logsqueak.wizard.wizard.rprint") as mock_rprint:
            result = await index_graph_after_setup("/invalid/path")

        # Verify
        assert result is False
        # Verify error message was displayed
        calls_text = " ".join(str(call) for call in mock_rprint.call_args_list)
        assert "failed" in calls_text.lower() or "error" in calls_text.lower()

    @pytest.mark.asyncio
    @patch("logseq_outline.graph.GraphPaths")
    @patch("logsqueak.services.page_indexer.PageIndexer")
    @patch("logsqueak.services.rag_search.RAGSearch")
    @patch("logsqueak.utils.index_progress.create_index_progress_callback")
    async def test_uses_correct_progress_callback_settings(self, mock_progress, mock_rag, mock_indexer_class, mock_graph_paths):
        """Test that progress callback is created with correct settings."""
        # Setup mocks
        mock_indexer = MagicMock()
        mock_indexer.build_index = AsyncMock(return_value=5)
        mock_indexer_class.return_value = mock_indexer

        mock_rag_instance = MagicMock()
        mock_rag_instance.has_indexed_data.return_value = False
        mock_rag.return_value = mock_rag_instance

        mock_callback = MagicMock()
        mock_cleanup = MagicMock()
        mock_progress.return_value = (mock_callback, mock_cleanup)

        # Execute
        await index_graph_after_setup("/path/to/graph")

        # Verify progress callback was created with wizard-appropriate settings
        mock_progress.assert_called_once()
        call_kwargs = mock_progress.call_args[1]
        assert call_kwargs["reindex"] is False  # Wizard never forces reindex
        assert call_kwargs["use_echo"] is False  # Wizard uses rprint style
        assert "has_data" in call_kwargs
