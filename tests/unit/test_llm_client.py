"""Unit tests for LLMClient."""

import pytest
import httpx
from unittest.mock import AsyncMock, Mock, patch
import json

from logsqueak.services.llm_client import LLMClient
from logsqueak.models.config import LLMConfig
from logsqueak.models.llm_chunks import KnowledgeClassificationChunk


class TestLLMClient:
    """Test LLMClient class."""

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
        """Create LLM client with test config."""
        return LLMClient(llm_config)

    def test_client_initialization(self, llm_client, llm_config):
        """Test LLM client initializes correctly."""
        assert llm_client.config == llm_config
        assert llm_client.timeout.read == 60.0
        assert llm_client.timeout.connect == 10.0

    @pytest.mark.asyncio
    async def test_stream_ndjson_success(self, llm_client):
        """Test successful NDJSON streaming."""
        # Mock response data (LLM only returns knowledge blocks)
        mock_lines = [
            '{"type": "classification", "block_id": "abc123", "confidence": 0.92, "reason": "Test"}',
            '{"type": "classification", "block_id": "def456", "confidence": 0.15}',
        ]

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

        with patch('httpx.AsyncClient', return_value=mock_client):
            chunks = []
            async for chunk in llm_client.stream_ndjson(
                prompt="Test prompt",
                system_prompt="Test system",
                chunk_model=KnowledgeClassificationChunk
            ):
                chunks.append(chunk)

            assert len(chunks) == 2
            assert chunks[0].block_id == "abc123"
            assert chunks[1].block_id == "def456"

    @pytest.mark.asyncio
    async def test_stream_ndjson_skips_malformed_json(self, llm_client):
        """Test streaming skips malformed JSON lines."""
        mock_lines = [
            '{"type": "classification", "block_id": "abc123", "confidence": 0.92}',
            '{invalid json}',  # Malformed
            '{"type": "classification", "block_id": "def456", "confidence": 0.15}',
        ]

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

        with patch('httpx.AsyncClient', return_value=mock_client):
            chunks = []
            async for chunk in llm_client.stream_ndjson(
                prompt="Test",
                system_prompt="Test",
                chunk_model=KnowledgeClassificationChunk
            ):
                chunks.append(chunk)

            # Should get 2 valid chunks, skipping malformed line
            assert len(chunks) == 2
            assert chunks[0].block_id == "abc123"
            assert chunks[1].block_id == "def456"

    @pytest.mark.asyncio
    async def test_stream_ndjson_skips_empty_lines(self, llm_client):
        """Test streaming skips empty lines."""
        mock_lines = [
            '{"type": "classification", "block_id": "abc123", "confidence": 0.92}',
            '',  # Empty line
            '   ',  # Whitespace only
            '{"type": "classification", "block_id": "def456", "confidence": 0.15}',
        ]

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

        with patch('httpx.AsyncClient', return_value=mock_client):
            chunks = []
            async for chunk in llm_client.stream_ndjson(
                prompt="Test",
                system_prompt="Test",
                chunk_model=KnowledgeClassificationChunk
            ):
                chunks.append(chunk)

            # Should get 2 chunks, skipping empty lines
            assert len(chunks) == 2

    @pytest.mark.asyncio
    async def test_stream_ndjson_retry_on_timeout(self, llm_client):
        """Test automatic retry on timeout errors."""
        # First attempt: timeout, second attempt: success
        mock_lines = [
            '{"type": "classification", "block_id": "abc123", "confidence": 0.92}',
        ]

        attempts = [0]

        def mock_stream_factory(*args, **kwargs):
            attempts[0] += 1
            if attempts[0] == 1:
                raise httpx.ReadTimeout("Timeout")
            else:
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
            with patch('asyncio.sleep', new_callable=AsyncMock):  # Skip actual sleep
                chunks = []
                async for chunk in llm_client.stream_ndjson(
                    prompt="Test",
                    system_prompt="Test",
                    chunk_model=KnowledgeClassificationChunk,
                    max_retries=1,
                    retry_delay=0.1
                ):
                    chunks.append(chunk)

                # Should succeed after retry
                assert len(chunks) == 1
                assert attempts[0] == 2

    @pytest.mark.asyncio
    async def test_stream_ndjson_fails_after_max_retries(self, llm_client):
        """Test that streaming fails after max retries exhausted."""
        def mock_stream_factory(*args, **kwargs):
            raise httpx.ReadTimeout("Timeout")

        mock_client = AsyncMock()
        mock_client.stream = mock_stream_factory
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch('httpx.AsyncClient', return_value=mock_client):
            with patch('asyncio.sleep', new_callable=AsyncMock):
                with pytest.raises(httpx.ReadTimeout):
                    async for chunk in llm_client.stream_ndjson(
                        prompt="Test",
                        system_prompt="Test",
                        chunk_model=KnowledgeClassificationChunk,
                        max_retries=1
                    ):
                        pass

    @pytest.mark.asyncio
    async def test_stream_ndjson_no_retry_on_4xx_errors(self, llm_client):
        """Test that 4xx errors don't trigger retry."""
        def mock_stream_factory(*args, **kwargs):
            mock_response = Mock()
            mock_response.status_code = 401
            raise httpx.HTTPStatusError("Unauthorized", request=Mock(), response=mock_response)

        mock_client = AsyncMock()
        mock_client.stream = mock_stream_factory
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch('httpx.AsyncClient', return_value=mock_client):
            with pytest.raises(httpx.HTTPStatusError):
                async for chunk in llm_client.stream_ndjson(
                    prompt="Test",
                    system_prompt="Test",
                    chunk_model=KnowledgeClassificationChunk,
                    max_retries=1
                ):
                    pass
