from typing import Any, AsyncIterator, Dict
from unittest.mock import AsyncMock

import pytest
from fastapi.responses import StreamingResponse
from litellm import ChatCompletionRequest, ModelResponse

from codegate.providers.litellmshim import BaseAdapter, LiteLLmShim


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
            self, completion_stream: Any,
    ) -> Any:
        async def modified_stream():
            async for chunk in completion_stream:
                chunk.mock_adapter_processed = True
                yield chunk
        return modified_stream()

@pytest.fixture
def mock_adapter():
    return MockAdapter()

@pytest.fixture
def litellm_shim(mock_adapter):
    return LiteLLmShim(mock_adapter)

@pytest.mark.asyncio
async def test_complete_non_streaming(litellm_shim, mock_adapter):
    # Mock response
    mock_response = ModelResponse(id="123", choices=[{"text": "test response"}])
    mock_completion = AsyncMock(return_value=mock_response)

    # Create shim with mocked completion
    litellm_shim = LiteLLmShim(mock_adapter, completion_func=mock_completion)

    # Test data
    data = {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "gpt-3.5-turbo"
    }
    api_key = "test-key"

    # Execute
    result = await litellm_shim.complete(data, api_key)

    # Verify
    assert result == mock_response
    mock_completion.assert_called_once()
    called_args = mock_completion.call_args[1]
    assert called_args["api_key"] == api_key
    assert called_args["messages"] == data["messages"]
    # Verify adapter processed the input
    assert called_args["mock_adapter_processed"] is True

@pytest.mark.asyncio
async def test_complete_streaming():
    # Mock streaming response with specific test content
    async def mock_stream() -> AsyncIterator[ModelResponse]:
        yield ModelResponse(id="123", choices=[{"text": "chunk1"}])
        yield ModelResponse(id="123", choices=[{"text": "chunk2"}])

    mock_completion = AsyncMock(return_value=mock_stream())
    mock_adapter = MockAdapter()
    litellm_shim = LiteLLmShim(mock_adapter, completion_func=mock_completion)

    # Test data
    data = {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "gpt-3.5-turbo",
        "stream": True
    }
    api_key = "test-key"

    # Execute
    result_stream = await litellm_shim.complete(data, api_key)

    # Verify stream contents and adapter processing
    chunks = []
    async for chunk in result_stream:
        chunks.append(chunk)
        # Verify each chunk was processed by the adapter
        assert hasattr(chunk, "mock_adapter_processed")
        assert chunk.mock_adapter_processed is True

    assert len(chunks) == 2
    assert chunks[0].choices[0]["text"] == "chunk1"
    assert chunks[1].choices[0]["text"] == "chunk2"

    # Verify completion function was called with correct parameters
    mock_completion.assert_called_once()
    called_args = mock_completion.call_args[1]
    assert called_args["mock_adapter_processed"] is True  # Verify input was processed
    assert called_args["messages"] == data["messages"]
    assert called_args["model"] == data["model"]
    assert called_args["stream"] is True
    assert called_args["api_key"] == api_key

@pytest.mark.asyncio
async def test_create_streaming_response(litellm_shim):
    # Create a simple async generator that we know works
    async def mock_stream_gen():
        for msg in ["Hello", "World"]:
            yield msg.encode()  # FastAPI expects bytes

    # Create and verify the generator
    generator = mock_stream_gen()

    response = litellm_shim.create_streaming_response(generator)

    # Verify response metadata
    assert isinstance(response, StreamingResponse)
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-cache"
    assert response.headers["Connection"] == "keep-alive"
    assert response.headers["Transfer-Encoding"] == "chunked"

@pytest.mark.asyncio
async def test_complete_invalid_params():
    mock_completion = AsyncMock()
    mock_adapter = MockAdapter()
    litellm_shim = LiteLLmShim(mock_adapter, completion_func=mock_completion)

    # Test data missing required fields
    data = {
        "invalid_field": "test"
        # missing 'messages' and 'model'
    }
    api_key = "test-key"

    # Execute and verify specific exception is raised
    with pytest.raises(
            ValueError,
            match="Required fields 'messages' and 'model' must be present",
    ):
        await litellm_shim.complete(data, api_key)

    # Verify the completion function was never called
    mock_completion.assert_not_called()
