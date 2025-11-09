"""Integration tests for LLM NDJSON streaming."""

import pytest
import json
from unittest.mock import AsyncMock, Mock, patch

from logsqueak.services.llm_client import LLMClient
from logsqueak.models.config import LLMConfig
from logsqueak.models.llm_chunks import (
    KnowledgeClassificationChunk,
    ContentRewordingChunk,
    IntegrationDecisionChunk
)


def create_mock_streaming_client(mock_lines):
    """Helper to create properly mocked httpx.AsyncClient for streaming."""
    # Mock httpx response
    mock_response = AsyncMock()
    mock_response.raise_for_status = Mock()

    async def mock_aiter_lines():
        for line in mock_lines:
            yield line

    mock_response.aiter_lines = mock_aiter_lines
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    # Mock httpx client
    mock_client = AsyncMock()
    mock_client.stream = Mock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    return mock_client


class TestLLMStreamingIntegration:
    """Integration tests for LLM NDJSON streaming with various scenarios."""

    @pytest.fixture
    def llm_config(self):
        """Create test LLM configuration."""
        return LLMConfig(
            endpoint="https://api.test.com/v1",
            api_key="test-key",
            model="test-model"
        )

    @pytest.fixture
    def llm_client(self, llm_config):
        """Create LLM client."""
        return LLMClient(llm_config)

    @pytest.mark.asyncio
    async def test_classification_streaming_workflow(self, llm_client):
        """Test complete classification streaming workflow."""
        # Simulate realistic NDJSON response (LLM only returns knowledge blocks)
        mock_lines = [
            '{"type": "classification", "block_id": "block-1", "confidence": 0.92, "reason": "Contains reusable programming insight"}',
            '{"type": "classification", "block_id": "block-3", "confidence": 0.85, "reason": "Explains important concept"}',
        ]

        mock_client = create_mock_streaming_client(mock_lines)

        with patch('httpx.AsyncClient', return_value=mock_client):
            results = []
            async for chunk in llm_client.stream_ndjson(
                prompt="Classify these blocks...",
                system_prompt="You are a knowledge classifier",
                chunk_model=KnowledgeClassificationChunk
            ):
                results.append(chunk)

            # Verify all chunks received (only knowledge blocks)
            assert len(results) == 2

            # Verify knowledge blocks
            assert results[0].block_id == "block-1"
            assert results[0].confidence == 0.92
            assert results[1].block_id == "block-3"
            assert results[1].confidence == 0.85

    @pytest.mark.asyncio
    async def test_rewording_streaming_workflow(self, llm_client):
        """Test content rewording streaming workflow."""
        mock_lines = [
            '{"type": "rewording", "block_id": "block-1", "reworded_content": "asyncio enables concurrent programming"}',
            '{"type": "rewording", "block_id": "block-2", "reworded_content": "Python type hints improve code clarity"}',
        ]

        mock_client = create_mock_streaming_client(mock_lines)

        with patch('httpx.AsyncClient', return_value=mock_client):
            results = []
            async for chunk in llm_client.stream_ndjson(
                prompt="Reword these blocks...",
                system_prompt="Remove temporal context",
                chunk_model=ContentRewordingChunk
            ):
                results.append(chunk)

            assert len(results) == 2
            assert all(chunk.reworded_content for chunk in results)
            assert results[0].block_id == "block-1"

    @pytest.mark.asyncio
    async def test_integration_decision_streaming_workflow(self, llm_client):
        """Test integration decision streaming workflow."""
        mock_lines = [
            '{"type": "decision", "knowledge_block_id": "kb-1", "target_page": "Programming/Python", "action": "add_under", "target_block_id": "async-section", "target_block_title": "Async Patterns", "confidence": 0.87, "reasoning": "Fits under async section"}',
            '{"type": "decision", "knowledge_block_id": "kb-1", "target_page": "Concurrency", "action": "add_section", "target_block_id": null, "target_block_title": null, "confidence": 0.72, "reasoning": "Alternative location"}',
        ]

        mock_client = create_mock_streaming_client(mock_lines)

        with patch('httpx.AsyncClient', return_value=mock_client):
            results = []
            async for chunk in llm_client.stream_ndjson(
                prompt="Plan integrations...",
                system_prompt="Suggest where to integrate",
                chunk_model=IntegrationDecisionChunk
            ):
                results.append(chunk)

            # Same knowledge block, two different target pages
            assert len(results) == 2
            assert all(r.knowledge_block_id == "kb-1" for r in results)
            assert results[0].target_page == "Programming/Python"
            assert results[1].target_page == "Concurrency"

    @pytest.mark.asyncio
    async def test_malformed_json_recovery(self, llm_client):
        """Test streaming recovers from malformed JSON lines."""
        mock_lines = [
            '{"type": "classification", "block_id": "block-1", "confidence": 0.92}',
            '{malformed json here}',
            'not json at all',
            '{"type": "classification", "block_id": "block-2", "confidence": 0.15}',
            '',  # Empty line
            '{"type": "classification", "block_id": "block-3", "confidence": 0.88}',
        ]

        mock_client = create_mock_streaming_client(mock_lines)

        with patch('httpx.AsyncClient', return_value=mock_client):
            results = []
            async for chunk in llm_client.stream_ndjson(
                prompt="Test",
                system_prompt="Test",
                chunk_model=KnowledgeClassificationChunk
            ):
                results.append(chunk)

            # Should get 3 valid chunks, skipping malformed/empty lines
            assert len(results) == 3
            assert results[0].block_id == "block-1"
            assert results[1].block_id == "block-2"
            assert results[2].block_id == "block-3"

    @pytest.mark.asyncio
    async def test_network_timeout_retry_success(self, llm_client):
        """Test successful retry after network timeout."""
        import httpx

        mock_lines = [
            '{"type": "classification", "block_id": "block-1", "confidence": 0.92}',
        ]

        attempts = [0]

        def mock_stream_factory(*args, **kwargs):
            attempts[0] += 1
            if attempts[0] == 1:
                # First attempt: timeout
                raise httpx.ReadTimeout("Connection timeout")
            else:
                # Second attempt: success
                mock_response = AsyncMock()
                mock_response.raise_for_status = Mock()

                async def mock_aiter_lines():
                    for line in mock_lines:
                        yield line

                mock_response.aiter_lines = mock_aiter_lines
                mock_response.__aenter__ = AsyncMock(return_value=mock_response)
                mock_response.__aexit__ = AsyncMock(return_value=None)
                return mock_response

        mock_client = AsyncMock()
        mock_client.stream = mock_stream_factory
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch('httpx.AsyncClient', return_value=mock_client):
            with patch('asyncio.sleep', new_callable=AsyncMock):
                results = []
                async for chunk in llm_client.stream_ndjson(
                    prompt="Test",
                    system_prompt="Test",
                    chunk_model=KnowledgeClassificationChunk,
                    max_retries=1,
                    retry_delay=2.0
                ):
                    results.append(chunk)

                # Should succeed after 1 retry
                assert len(results) == 1
                assert attempts[0] == 2

    @pytest.mark.asyncio
    async def test_empty_stream(self, llm_client):
        """Test handling of empty stream (no results)."""
        mock_lines = []  # Empty response

        mock_client = create_mock_streaming_client(mock_lines)

        with patch('httpx.AsyncClient', return_value=mock_client):
            results = []
            async for chunk in llm_client.stream_ndjson(
                prompt="Test",
                system_prompt="Test",
                chunk_model=KnowledgeClassificationChunk
            ):
                results.append(chunk)

            # Should handle empty stream gracefully
            assert len(results) == 0
