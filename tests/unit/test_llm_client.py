"""Unit tests for LLMClient."""

import pytest
import httpx
from unittest.mock import AsyncMock, Mock, MagicMock, patch
import json

from logsqueak.services.llm_client import LLMClient
from logsqueak.models.config import LLMConfig
from logsqueak.models.llm_chunks import KnowledgeClassificationChunk


def create_mock_response(lines, status_code=200, headers=None):
    """Create a properly configured mock HTTP response.

    Uses MagicMock instead of AsyncMock to avoid triggering AsyncMockMixin
    warnings when accessing attributes like headers.
    """
    # Use a regular object with __aenter__ and __aexit__ instead of MagicMock
    # to avoid any async mock issues
    class MockResponse:
        def __init__(self):
            self.status_code = status_code
            self.headers = headers or {}
            self.raise_for_status = Mock()

        async def aiter_lines(self):
            for line in lines:
                yield line

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

    return MockResponse()


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
        """Test successful NDJSON streaming with new insight-based format."""
        # Mock response data (insights with reworded content, 1:1 block mapping)
        mock_lines = [
            '{"block_id": "abc123", "insight": "Test insight 1", "confidence": 0.92}',
            '{"block_id": "def456", "insight": "Test insight 2", "confidence": 0.85}',
        ]

        # Mock httpx response
        mock_response = create_mock_response(mock_lines)

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
            assert chunks[0].insight == "Test insight 1"
            assert chunks[1].block_id == "def456"
            assert chunks[1].insight == "Test insight 2"

    @pytest.mark.asyncio
    async def test_stream_ndjson_skips_malformed_json(self, llm_client):
        """Test streaming skips malformed JSON lines."""
        mock_lines = [
            '{"block_id": "abc123", "insight": "Test", "confidence": 0.92}',
            '{invalid json}',  # Malformed
            '{"block_id": "def456", "insight": "Test2", "confidence": 0.85}',
        ]

        # Mock httpx response
        mock_response = create_mock_response(mock_lines)

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
            '{"block_id": "abc123", "insight": "Test insight", "confidence": 0.92}',
            '',  # Empty line
            '   ',  # Whitespace only
            '{"block_id": "def456", "insight": "Another insight", "confidence": 0.85}',
        ]

        # Mock httpx response


        mock_response = create_mock_response(mock_lines)

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
            '{"block_id": "abc123", "insight": "Test insight", "confidence": 0.92}',
        ]

        attempts = [0]

        def mock_stream_factory(*args, **kwargs):
            attempts[0] += 1
            if attempts[0] == 1:
                raise httpx.ReadTimeout("Timeout")
            else:
                return create_mock_response(mock_lines)

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
        """Test that 4xx errors (except 429) don't trigger retry."""
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

    @pytest.mark.asyncio
    async def test_stream_ndjson_retry_on_429_rate_limit(self, llm_client):
        """Test automatic retry on 429 rate limit errors with exponential backoff."""
        # First attempt: 429, second attempt: success
        mock_lines = [
            '{"block_id": "abc123", "insight": "Test insight", "confidence": 0.92}',
        ]

        attempts = [0]

        def mock_stream_factory(*args, **kwargs):
            attempts[0] += 1
            if attempts[0] == 1:
                mock_response = Mock()
                mock_response.status_code = 429
                mock_response.headers = {}  # No Retry-After header
                raise httpx.HTTPStatusError("Rate limited", request=Mock(), response=mock_response)
            else:
                return create_mock_response(mock_lines)

        mock_client = AsyncMock()
        mock_client.stream = mock_stream_factory
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch('httpx.AsyncClient', return_value=mock_client):
            with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                chunks = []
                async for chunk in llm_client.stream_ndjson(
                    prompt="Test",
                    system_prompt="Test",
                    chunk_model=KnowledgeClassificationChunk,
                    max_retries=1,
                    retry_delay=2.0
                ):
                    chunks.append(chunk)

                # Should succeed after retry
                assert len(chunks) == 1
                assert attempts[0] == 2
                # Should use exponential backoff: 2.0 * (2 ** 0) = 2.0 seconds
                mock_sleep.assert_called_once_with(2.0)

    @pytest.mark.asyncio
    async def test_stream_ndjson_429_respects_retry_after_header(self, llm_client):
        """Test that 429 retry respects Retry-After header from API."""
        mock_lines = [
            '{"block_id": "abc123", "insight": "Test insight", "confidence": 0.92}',
        ]

        attempts = [0]

        def mock_stream_factory(*args, **kwargs):
            attempts[0] += 1
            if attempts[0] == 1:
                mock_response = Mock()
                mock_response.status_code = 429
                mock_response.headers = {"Retry-After": "5"}  # API says wait 5 seconds
                raise httpx.HTTPStatusError("Rate limited", request=Mock(), response=mock_response)
            else:
                return create_mock_response(mock_lines)

        mock_client = AsyncMock()
        mock_client.stream = mock_stream_factory
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch('httpx.AsyncClient', return_value=mock_client):
            with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                chunks = []
                async for chunk in llm_client.stream_ndjson(
                    prompt="Test",
                    system_prompt="Test",
                    chunk_model=KnowledgeClassificationChunk,
                    max_retries=1
                ):
                    chunks.append(chunk)

                # Should succeed after retry
                assert len(chunks) == 1
                assert attempts[0] == 2
                # Should respect Retry-After header
                mock_sleep.assert_called_once_with(5)

    @pytest.mark.asyncio
    async def test_stream_ndjson_429_fails_after_max_retries(self, llm_client):
        """Test that 429 errors fail after max retries exhausted."""
        def mock_stream_factory(*args, **kwargs):
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.headers = {}
            raise httpx.HTTPStatusError("Rate limited", request=Mock(), response=mock_response)

        mock_client = AsyncMock()
        mock_client.stream = mock_stream_factory
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch('httpx.AsyncClient', return_value=mock_client):
            with patch('asyncio.sleep', new_callable=AsyncMock):
                with pytest.raises(httpx.HTTPStatusError) as exc_info:
                    async for chunk in llm_client.stream_ndjson(
                        prompt="Test",
                        system_prompt="Test",
                        chunk_model=KnowledgeClassificationChunk,
                        max_retries=1
                    ):
                        pass

                # Verify it's a 429 error
                assert exc_info.value.response.status_code == 429

    @pytest.mark.asyncio
    async def test_stream_sse_format(self, llm_client):
        """Test SSE format (Server-Sent Events) with 'data: ' prefix."""
        # Mock SSE response data (Ollama/OpenAI format with data: prefix)
        mock_lines = [
            'data: {"block_id": "abc123", "insight": "SSE test insight", "confidence": 0.92}',
            'data: {"block_id": "def456", "insight": "Another SSE insight", "confidence": 0.85}',
            'data: [DONE]',  # SSE termination message
        ]

        # Mock httpx response


        mock_response = create_mock_response(mock_lines)

        # Mock httpx client
        mock_client = AsyncMock()
        mock_client.stream = Mock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch('httpx.AsyncClient', return_value=mock_client):
            chunks = []
            async for chunk in llm_client.stream_ndjson(
                prompt="Test SSE",
                system_prompt="Test system",
                chunk_model=KnowledgeClassificationChunk
            ):
                chunks.append(chunk)

            # Should parse 2 chunks (skip [DONE] message)
            assert len(chunks) == 2
            assert chunks[0].block_id == "abc123"
            assert chunks[0].insight == "SSE test insight"
            assert chunks[0].confidence == 0.92
            assert chunks[1].block_id == "def456"
            assert chunks[1].insight == "Another SSE insight"
            assert chunks[1].confidence == 0.85

    @pytest.mark.asyncio
    async def test_stream_openai_incremental_format(self, llm_client):
        """Test OpenAI incremental streaming format where JSON is built token-by-token."""
        # Mock OpenAI-style streaming where content comes in fragments
        # The LLM streams the JSON response token-by-token
        mock_lines = [
            'data: {"choices":[{"delta":{"content":"{\\"block"}}]}',
            'data: {"choices":[{"delta":{"content":"_id\\": \\"abc123\\", \\"insight\\": \\"Test"}}]}',
            'data: {"choices":[{"delta":{"content":" insight\\", \\"confidence\\": 0.92}\\n"}}]}',
            'data: {"choices":[{"delta":{"content":"{\\"block_id\\": \\"def456\\", \\"insight\\": \\"Another insight\\", \\"confidence\\": 0.85}"}}]}',
            'data: [DONE]',
        ]

        # Mock httpx response with SSE headers
        mock_response = create_mock_response(
            mock_lines,
            headers={"content-type": "text/event-stream"}
        )

        # Mock httpx client
        mock_client = AsyncMock()
        mock_client.stream = Mock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch('httpx.AsyncClient', return_value=mock_client):
            chunks = []
            async for chunk in llm_client.stream_ndjson(
                prompt="Test OpenAI incremental",
                system_prompt="Test system",
                chunk_model=KnowledgeClassificationChunk
            ):
                chunks.append(chunk)

            # Should accumulate fragments and parse 2 complete JSON objects
            assert len(chunks) == 2
            assert chunks[0].block_id == "abc123"
            assert chunks[0].insight == "Test insight"
            assert chunks[0].confidence == 0.92
            assert chunks[1].block_id == "def456"
            assert chunks[1].insight == "Another insight"
            assert chunks[1].confidence == 0.85

    @pytest.mark.asyncio
    async def test_stream_ndjson_json_array_fallback(self, llm_client):
        """Test that JSON array fallback works when NDJSON parsing fails."""
        # Simulate LLM returning JSON array instead of NDJSON
        mock_lines = [
            'data: {"choices":[{"delta":{"content":"[{\\"block_id\\": \\"abc123\\", \\"insight\\": \\"Test\\", \\"confidence\\": 0.92}, {\\"block_id\\": \\"def456\\", \\"insight\\": \\"Test2\\", \\"confidence\\": 0.85}]"}}]}',
            'data: [DONE]',
        ]

        mock_response = create_mock_response(
            mock_lines,
            headers={"content-type": "text/event-stream"}
        )

        mock_client = AsyncMock()
        mock_client.stream = Mock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch('httpx.AsyncClient', return_value=mock_client):
            chunks = []
            async for chunk in llm_client.stream_ndjson(
                prompt="Test JSON array fallback",
                system_prompt="Test system",
                chunk_model=KnowledgeClassificationChunk
            ):
                chunks.append(chunk)

            # Should fallback to parsing as JSON array and yield 2 chunks
            assert len(chunks) == 2
            assert chunks[0].block_id == "abc123"
            assert chunks[0].insight == "Test"
            assert chunks[0].confidence == 0.92
            assert chunks[1].block_id == "def456"
            assert chunks[1].insight == "Test2"
            assert chunks[1].confidence == 0.85

    @pytest.mark.asyncio
    async def test_stream_ndjson_no_fallback_if_ndjson_succeeds(self, llm_client):
        """Test that fallback is NOT triggered when some NDJSON lines parse successfully."""
        # Mix of valid NDJSON and invalid lines
        mock_lines = [
            'data: {"choices":[{"delta":{"content":"{\\"block_id\\": \\"abc123\\", \\"insight\\": \\"Test insight\\", \\"confidence\\": 0.92}\\n"}}]}',
            'data: {"choices":[{"delta":{"content":"invalid line\\n"}}]}',
            'data: [DONE]',
        ]

        mock_response = create_mock_response(
            mock_lines,
            headers={"content-type": "text/event-stream"}
        )

        mock_client = AsyncMock()
        mock_client.stream = Mock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch('httpx.AsyncClient', return_value=mock_client):
            chunks = []
            async for chunk in llm_client.stream_ndjson(
                prompt="Test no fallback",
                system_prompt="Test system",
                chunk_model=KnowledgeClassificationChunk
            ):
                chunks.append(chunk)

            # Should parse 1 valid NDJSON line, skip invalid, NOT trigger fallback
            assert len(chunks) == 1
            assert chunks[0].block_id == "abc123"

    @pytest.mark.asyncio
    async def test_stream_ndjson_fallback_fails_on_invalid_json(self, llm_client):
        """Test that fallback gracefully fails on invalid JSON."""
        # Invalid JSON that can't be parsed as NDJSON or array
        mock_lines = [
            'data: {"choices":[{"delta":{"content":"not valid json at all"}}]}',
            'data: [DONE]',
        ]

        mock_response = create_mock_response(
            mock_lines,
            headers={"content-type": "text/event-stream"}
        )

        mock_client = AsyncMock()
        mock_client.stream = Mock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch('httpx.AsyncClient', return_value=mock_client):
            chunks = []
            async for chunk in llm_client.stream_ndjson(
                prompt="Test invalid JSON",
                system_prompt="Test system",
                chunk_model=KnowledgeClassificationChunk
            ):
                chunks.append(chunk)

            # Should return 0 chunks and log warnings
            assert len(chunks) == 0

    @pytest.mark.asyncio
    async def test_stream_ndjson_fallback_handles_wrapped_object(self, llm_client):
        """Test that fallback handles JSON object wrapped with extra text."""
        # JSON object with conversational wrapper (not valid NDJSON or array)
        mock_lines = [
            'data: {"choices":[{"delta":{"content":"Here is the result: {\\"block_id\\": \\"abc123\\", \\"confidence\\": 0.92}"}}]}',
            'data: [DONE]',
        ]

        mock_response = create_mock_response(
            mock_lines,
            headers={"content-type": "text/event-stream"}
        )

        mock_client = AsyncMock()
        mock_client.stream = Mock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch('httpx.AsyncClient', return_value=mock_client):
            chunks = []
            async for chunk in llm_client.stream_ndjson(
                prompt="Test wrapped JSON",
                system_prompt="Test system",
                chunk_model=KnowledgeClassificationChunk
            ):
                chunks.append(chunk)

            # Should return 0 chunks (wrapped text can't be parsed as NDJSON,
            # and fallback won't work because it's not valid JSON or array)
            assert len(chunks) == 0

    @pytest.mark.asyncio
    async def test_stream_ndjson_handles_multiline_json(self, llm_client):
        """Test that multi-line formatted JSON objects are reassembled and parsed."""
        # LLM returns pretty-printed JSON (multi-line) instead of single-line NDJSON
        mock_lines = [
            'data: {"choices":[{"delta":{"content":" {\\n"}}]}',
            'data: {"choices":[{"delta":{"content":"  \\"block_id\\": \\"abc123\\",\\n"}}]}',
            'data: {"choices":[{"delta":{"content":"  \\"insight\\": \\"Test insight\\",\\n"}}]}',
            'data: {"choices":[{"delta":{"content":"  \\"confidence\\": 0.92\\n"}}]}',
            'data: {"choices":[{"delta":{"content":"}"}}]}',
            'data: [DONE]',
        ]

        mock_response = create_mock_response(
            mock_lines,
            headers={"content-type": "text/event-stream"}
        )

        mock_client = AsyncMock()
        mock_client.stream = Mock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch('httpx.AsyncClient', return_value=mock_client):
            chunks = []
            async for chunk in llm_client.stream_ndjson(
                prompt="Test multiline JSON",
                system_prompt="Test system",
                chunk_model=KnowledgeClassificationChunk
            ):
                chunks.append(chunk)

            # Should reassemble multi-line JSON and parse 1 chunk
            assert len(chunks) == 1
            assert chunks[0].block_id == "abc123"
            assert chunks[0].confidence == 0.92

    @pytest.mark.asyncio
    async def test_stream_ndjson_handles_multiple_multiline_objects(self, llm_client):
        """Test that multiple multi-line JSON objects separated by blank lines are parsed."""
        # LLM returns multiple pretty-printed JSON objects separated by blank lines
        mock_lines = [
            'data: {"choices":[{"delta":{"content":" {\\n"}}]}',
            'data: {"choices":[{"delta":{"content":"\\"block_id\\": \\"abc123\\",\\n"}}]}',
            'data: {"choices":[{"delta":{"content":"\\"insight\\": \\"Test insight\\",\\n"}}]}',
            'data: {"choices":[{"delta":{"content":"\\"confidence\\": 0.92\\n"}}]}',
            'data: {"choices":[{"delta":{"content":"}\\n\\n"}}]}',
            'data: {"choices":[{"delta":{"content":"{\\n"}}]}',
            'data: {"choices":[{"delta":{"content":"\\"block_id\\": \\"def456\\",\\n"}}]}',
            'data: {"choices":[{"delta":{"content":"\\"insight\\": \\"Another insight\\",\\n"}}]}',
            'data: {"choices":[{"delta":{"content":"\\"confidence\\": 0.85\\n"}}]}',
            'data: {"choices":[{"delta":{"content":"}"}}]}',
            'data: [DONE]',
        ]

        mock_response = create_mock_response(
            mock_lines,
            headers={"content-type": "text/event-stream"}
        )

        mock_client = AsyncMock()
        mock_client.stream = Mock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch('httpx.AsyncClient', return_value=mock_client):
            chunks = []
            async for chunk in llm_client.stream_ndjson(
                prompt="Test multiple multiline",
                system_prompt="Test system",
                chunk_model=KnowledgeClassificationChunk
            ):
                chunks.append(chunk)

            # Should split on blank lines and parse 2 chunks
            assert len(chunks) == 2
            assert chunks[0].block_id == "abc123"
            assert chunks[0].confidence == 0.92
            assert chunks[1].block_id == "def456"
            assert chunks[1].confidence == 0.85
