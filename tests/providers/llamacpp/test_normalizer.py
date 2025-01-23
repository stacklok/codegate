import pytest
from llama_cpp.llama_types import CreateChatCompletionStreamResponse

from codegate.providers.llamacpp.normalizer import (
    LLamaCppOutputNormalizer,
)
from codegate.types.common import ModelResponse, Delta, StreamingChoices


class TestLLamaCppStreamNormalizer:
    @pytest.mark.asyncio
    async def test_normalize_streaming(self):
        """
        Test the normalize_streaming method
        Verify conversion from llama.cpp stream to ModelResponse stream
        """

        # Mock CreateChatCompletionStreamResponse stream
        async def mock_llamacpp_stream():
            responses = [
                CreateChatCompletionStreamResponse(
                    id="test_id1",
                    model="llama-model",
                    object="chat.completion.chunk",
                    created=1234567,
                    choices=[{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
                ),
                CreateChatCompletionStreamResponse(
                    id="test_id2",
                    model="llama-model",
                    object="chat.completion.chunk",
                    created=1234568,
                    choices=[{"index": 0, "delta": {"content": " World"}, "finish_reason": "stop"}],
                ),
            ]
            for resp in responses:
                yield resp

        # Create normalizer and normalize stream
        normalizer = LLamaCppOutputNormalizer()
        normalized_stream = normalizer.normalize_streaming(mock_llamacpp_stream())

        # Collect results
        results = []
        async for response in normalized_stream:
            results.append(response)

        # Assertions
        assert len(results) == 2
        assert all(isinstance(r, ModelResponse) for r in results)

        # Check first chunk
        assert results[0].choices[0].delta.content == "Hello"
        assert results[0].choices[0].finish_reason is None

        # Check second chunk
        assert results[1].choices[0].delta.content == " World"
        assert results[1].choices[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_denormalize_streaming(self):
        """
        Test the denormalize_streaming method
        Verify conversion from ModelResponse stream to llama.cpp stream
        """

        # Mock ModelResponse stream
        async def mock_model_response_stream():
            responses = [
                ModelResponse(
                    id="test_id1",
                    model="litellm-model",
                    object="chat.completion",
                    created=1234567,
                    choices=[
                        StreamingChoices(index=0, delta=Delta(content="Hello"), finish_reason=None)
                    ],
                ),
                ModelResponse(
                    id="test_id2",
                    model="litellm-model",
                    object="chat.completion",
                    created=1234568,
                    choices=[
                        StreamingChoices(
                            index=0, delta=Delta(content=" World"), finish_reason="stop"
                        )
                    ],
                ),
            ]
            for resp in responses:
                yield resp

        # Create normalizer and denormalize stream
        normalizer = LLamaCppOutputNormalizer()
        denormalized_stream = normalizer.denormalize_streaming(mock_model_response_stream())

        # Collect results
        results = []
        async for response in denormalized_stream:
            results.append(response)

        # Assertions
        assert len(results) == 2

        # Check first chunk
        assert results[0]["choices"][0]["delta"]["content"] == "Hello"
        assert results[0]["choices"][0]["finish_reason"] is None

        # Check second chunk
        assert results[1]["choices"][0]["delta"]["content"] == " World"
        assert results[1]["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_streaming_edge_cases(self):
        """
        Test edge cases and error scenarios in streaming
        """

        # Empty stream
        async def empty_stream():
            return
            yield

        normalizer = LLamaCppOutputNormalizer()

        # Test empty stream for normalize_streaming
        normalized_empty = normalizer.normalize_streaming(empty_stream())
        with pytest.raises(StopAsyncIteration):
            await normalized_empty.__anext__()

        # Test empty stream for denormalize_streaming
        async def empty_model_stream():
            return
            yield

        denormalized_empty = normalizer.denormalize_streaming(empty_model_stream())
        with pytest.raises(StopAsyncIteration):
            await denormalized_empty.__anext__()
