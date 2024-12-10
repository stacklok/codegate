from typing import Any, AsyncIterable, AsyncIterator, Dict, Union

from litellm import ChatCompletionRequest, ModelResponse
from litellm.types.utils import Delta, StreamingChoices
from llama_cpp.llama_types import (
    ChatCompletionStreamResponseChoice,
    ChatCompletionStreamResponseDelta,
    ChatCompletionStreamResponseDeltaEmpty,
    CreateChatCompletionStreamResponse,
)

from codegate.providers.normalizer import ModelInputNormalizer, ModelOutputNormalizer


class LLamaCppInputNormalizer(ModelInputNormalizer):
    def normalize(self, data: Dict) -> ChatCompletionRequest:
        """
        Normalize the input data
        """
        # Make a copy of the data to avoid modifying the original and normalize the message content
        return self._normalize_content_messages(data)

    def denormalize(self, data: ChatCompletionRequest) -> Dict:
        """
        Denormalize the input data
        """
        return data


class ModelToLlamaCpp(AsyncIterator[CreateChatCompletionStreamResponse]):
    def __init__(self, normalized_reply: AsyncIterable[ModelResponse]):
        self.normalized_reply = normalized_reply
        self._aiter = normalized_reply.__aiter__()

    def __aiter__(self):
        return self

    @staticmethod
    def _create_delta(
        choice_delta: Delta,
    ) -> Union[ChatCompletionStreamResponseDelta, ChatCompletionStreamResponseDeltaEmpty]:
        if not choice_delta:
            return ChatCompletionStreamResponseDeltaEmpty()
        return ChatCompletionStreamResponseDelta(
            content=choice_delta.content,
            role=choice_delta.role,
        )

    async def __anext__(self) -> CreateChatCompletionStreamResponse:
        try:
            chunk = await self._aiter.__anext__()
            return CreateChatCompletionStreamResponse(
                id=chunk["id"],
                model=chunk["model"],
                object="chat.completion.chunk",
                created=chunk["created"],
                choices=[
                    ChatCompletionStreamResponseChoice(
                        index=choice.index,
                        delta=self._create_delta(choice.delta),
                        finish_reason=choice.finish_reason,
                        logprobs=None,
                    )
                    for choice in chunk["choices"]
                ],
            )
        except StopAsyncIteration:
            raise StopAsyncIteration


class LlamaCppToModel(AsyncIterator[ModelResponse]):
    def __init__(self, normalized_reply: AsyncIterable[CreateChatCompletionStreamResponse]):
        self.normalized_reply = normalized_reply
        self._aiter = normalized_reply.__aiter__()

    def __aiter__(self):
        return self

    @staticmethod
    def _create_delta(
        choice_delta: Union[
            ChatCompletionStreamResponseDelta, ChatCompletionStreamResponseDeltaEmpty
        ]
    ) -> Delta:
        if not choice_delta:  # Handles empty dict case
            return Delta(content=None, role=None)
        return Delta(content=choice_delta.get("content"), role=choice_delta.get("role"))

    async def __anext__(self) -> ModelResponse:
        try:
            chunk = await self._aiter.__anext__()
            return ModelResponse(
                id=chunk["id"],
                choices=[
                    StreamingChoices(
                        finish_reason=choice.get("finish_reason", None),
                        index=choice["index"],
                        delta=self._create_delta(choice.get("delta")),
                        logprobs=None,
                    )
                    for choice in chunk["choices"]
                ],
                created=chunk["created"],
                model=chunk["model"],
                object=chunk["object"],
            )
        except StopAsyncIteration:
            raise StopAsyncIteration


class LLamaCppOutputNormalizer(ModelOutputNormalizer):
    def normalize_streaming(
        self,
        llamacpp_stream: AsyncIterable[CreateChatCompletionStreamResponse],
    ) -> AsyncIterator[ModelResponse]:
        """
        Normalize the output stream. This is a pass-through for liteLLM output normalizer
        as the liteLLM output is already in the normalized format.
        """
        return LlamaCppToModel(llamacpp_stream)

    def normalize(self, model_reply: Any) -> ModelResponse:
        """
        Normalize the output data. This is a pass-through for liteLLM output normalizer
        as the liteLLM output is already in the normalized format.
        """
        return model_reply

    def denormalize(self, normalized_reply: ModelResponse) -> Any:
        """
        Denormalize the output data from the completion function to the format
        expected by the client
        """
        return normalized_reply

    def denormalize_streaming(
        self,
        model_stream: AsyncIterable[ModelResponse],
    ) -> AsyncIterator[CreateChatCompletionStreamResponse]:
        """
        Denormalize the output stream from the completion function to the format
        expected by the client
        """
        return ModelToLlamaCpp(model_stream)
