from typing import Any, AsyncIterator, Dict
from unittest.mock import AsyncMock

import pytest
from fastapi.responses import StreamingResponse
from litellm import ChatCompletionRequest, ModelResponse

from codegate.providers.litellmshim import BaseAdapter, LiteLLmShim, sse_stream_generator


class MockAdapter(BaseAdapter):
    def __init__(self):
        self.stream_generator = AsyncMock()
        super().__init__(self.stream_generator)

    def translate_completion_input_params(self, kwargs: Dict) -> ChatCompletionRequest:
        # Validate required fields
        if "messages" not in kwargs or "model" not in kwargs:
            raise ValueError("Required fields 'messages' and 'model' must be present")

        modified_kwargs = kwargs.copy()
        modified_kwargs["mock_adapter_processed"] = True
        return ChatCompletionRequest(**modified_kwargs)

    def translate_completion_output_params(self, response: ModelResponse) -> Any:
        response.mock_adapter_processed = True
        return response

    def translate_completion_output_params_streaming(
        self,
        completion_stream: Any,
    ) -> Any:
        async def modified_stream():
            async for chunk in completion_stream:
                chunk.mock_adapter_processed = True
                yield chunk

        return modified_stream()


@pytest.mark.asyncio
async def test_complete_non_streaming():
    # Mock response
    mock_response = ModelResponse(id="123", choices=[{"text": "test response"}])
    mock_completion = AsyncMock(return_value=mock_response)

    # Create shim with mocked completion
    litellm_shim = LiteLLmShim(
        stream_generator=sse_stream_generator, completion_func=mock_completion
    )

    # Test data
    data = {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "gpt-3.5-turbo",
    }

    # Execute
    result = await litellm_shim.execute_completion(data, base_url=None, api_key=None)

    # Verify
    assert result == mock_response
    mock_completion.assert_called_once()
    called_args = mock_completion.call_args[1]
    assert called_args["messages"] == data["messages"]


@pytest.mark.asyncio
async def test_complete_streaming():
    # Mock streaming response with specific test content
    async def mock_stream() -> AsyncIterator[ModelResponse]:
        yield ModelResponse(id="123", choices=[{"text": "chunk1"}])
        yield ModelResponse(id="123", choices=[{"text": "chunk2"}])

    mock_completion = AsyncMock(return_value=mock_stream())
    litellm_shim = LiteLLmShim(
        stream_generator=sse_stream_generator, completion_func=mock_completion
    )

    # Test data
    data = {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "gpt-3.5-turbo",
        "stream": True,
    }

    # Execute
    result_stream = await litellm_shim.execute_completion(
        ChatCompletionRequest(**data), base_url=None, api_key=None
    )

    # Verify stream contents and adapter processing
    chunks = []
    async for chunk in result_stream:
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0].choices[0]["text"] == "chunk1"
    assert chunks[1].choices[0]["text"] == "chunk2"

    # Verify completion function was called with correct parameters
    mock_completion.assert_called_once()
    called_args = mock_completion.call_args[1]
    assert called_args["messages"] == data["messages"]
    assert called_args["model"] == data["model"]
    assert called_args["stream"] is True


@pytest.mark.asyncio
async def test_create_streaming_response():
    # Create a simple async generator that we know works
    async def mock_stream_gen():
        for msg in ["Hello", "World"]:
            yield msg.encode()  # FastAPI expects bytes

    # Create and verify the generator
    generator = mock_stream_gen()

    litellm_shim = LiteLLmShim(stream_generator=sse_stream_generator)
    response = litellm_shim._create_streaming_response(generator)

    # Verify response metadata
    assert isinstance(response, StreamingResponse)
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-cache"
    assert response.headers["Connection"] == "keep-alive"
    assert response.headers["Transfer-Encoding"] == "chunked"
