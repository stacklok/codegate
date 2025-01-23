from typing import AsyncIterator

import pytest

from codegate.types.generators import (
    anthropic_stream_generator,
    sse_stream_generator,
)
from codegate.types.common import ModelResponse


@pytest.mark.asyncio
async def test_sse_stream_generator():
    # Mock stream data
    mock_chunks = [
        ModelResponse(id="1", choices=[{"text": "Hello"}]),
        ModelResponse(id="2", choices=[{"text": "World"}]),
    ]

    async def mock_stream():
        for chunk in mock_chunks:
            yield chunk

    # Collect generated SSE messages
    messages = []
    async for message in sse_stream_generator(mock_stream()):
        messages.append(message)

    # Verify format and content
    assert len(messages) == len(mock_chunks) + 1  # +1 for the [DONE] message
    assert all(msg.startswith("data:") for msg in messages)
    assert "Hello" in messages[0]
    assert "World" in messages[1]
    assert messages[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_anthropic_stream_generator():
    # Mock Anthropic-style chunks
    mock_chunks = [
        {"type": "message_start", "message": {"id": "1"}},
        {"type": "content_block_start", "content_block": {"text": "Hello"}},
        {"type": "content_block_stop", "content_block": {"text": "World"}},
    ]

    async def mock_stream():
        for chunk in mock_chunks:
            yield chunk

    # Collect generated SSE messages
    messages = []
    async for message in anthropic_stream_generator(mock_stream()):
        messages.append(message)

    # Verify format and content
    assert len(messages) == 3
    for msg, chunk in zip(messages, mock_chunks):
        assert msg.startswith(f"event: {chunk['type']}\ndata:")
    assert "Hello" in messages[1]  # content_block_start message
    assert "World" in messages[2]  # content_block_stop message


@pytest.mark.asyncio
async def test_generators_error_handling():
    async def error_stream() -> AsyncIterator[str]:
        raise Exception("Test error")
        yield  # This will never be reached, but is needed for AsyncIterator typing

    # Test SSE generator error handling
    messages = []
    async for message in sse_stream_generator(error_stream()):
        messages.append(message)
    assert len(messages) == 2
    assert "Test error" in messages[0]
    assert messages[1] == "data: [DONE]\n\n"

    # Test Anthropic generator error handling
    messages = []
    async for message in anthropic_stream_generator(error_stream()):
        messages.append(message)
    assert len(messages) == 1
    assert "Test error" in messages[0]
