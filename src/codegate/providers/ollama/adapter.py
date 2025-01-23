import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Union

from ollama import ChatResponse, Message

from codegate.providers.normalizer.base import ModelInputNormalizer, ModelOutputNormalizer
from codegate.types.common import (
    ChatCompletionRequest,
    Delta,
    ModelResponse,
    StreamingChoices,
)


class OllamaInputNormalizer(ModelInputNormalizer):

    def normalize(self, data: Dict) -> ChatCompletionRequest:
        """
        Normalize the input data to the format expected by Ollama.
        """
        # Make a copy of the data to avoid modifying the original and normalize the message content
        normalized_data = self._normalize_content_messages(data)
        normalized_data["model"] = data.get("model", "").strip()
        normalized_data["options"] = data.get("options", {})

        if "prompt" in normalized_data:
            normalized_data["messages"] = [
                {"content": normalized_data.pop("prompt"), "role": "user"}
            ]

        # if we have the stream flag in data we set it, otherwise defaults to true
        normalized_data["stream"] = data.get("stream", True)

        # This would normally be the required to get the token usage.
        # However Ollama python client doesn't support it. We would be able to get the response
        # with a direct HTTP request. Since Ollama is local this is not critical.
        # if normalized_data.get("stream", False):
        #     normalized_data["stream_options"] = {"include_usage": True}
        return ChatCompletionRequest(**normalized_data)

    def denormalize(self, data: ChatCompletionRequest) -> Dict:
        """
        Convert back to raw format for the API request
        """
        return data


class OLlamaToModel(AsyncIterator[ModelResponse]):
    def __init__(self, ollama_response: AsyncIterator[ChatResponse]):
        self.ollama_response = ollama_response
        self._aiter = ollama_response.__aiter__()

    @staticmethod
    def normalize_chunk(chunk: ChatResponse) -> ModelResponse:
        finish_reason = None
        role = "assistant"

        # Convert the datetime object to a timestamp in seconds
        datetime_obj = datetime.fromisoformat(chunk.created_at)
        timestamp_seconds = int(datetime_obj.timestamp())

        if chunk.done:
            finish_reason = "stop"
            role = None

        model_response = ModelResponse(
            id=f"ollama-chat-{str(uuid.uuid4())}",
            created=timestamp_seconds,
            model=chunk.model,
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    finish_reason=finish_reason,
                    index=0,
                    delta=Delta(content=chunk.message.content, role=role),
                    logprobs=None,
                )
            ],
        )
        return model_response

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await self._aiter.__anext__()
            if not isinstance(chunk, ChatResponse):
                return chunk

            return self.normalize_chunk(chunk)
        except StopAsyncIteration:
            raise StopAsyncIteration


class ModelToOllama(AsyncIterator[ChatResponse]):

    def __init__(self, normalized_reply: AsyncIterator[ModelResponse]):
        self.normalized_reply = normalized_reply
        self._aiter = normalized_reply.__aiter__()

    def __aiter__(self):
        return self

    async def __anext__(self) -> Union[ChatResponse]:
        try:
            chunk = await self._aiter.__anext__()
            if not isinstance(chunk, ModelResponse):
                return chunk
            # Convert the timestamp to a datetime object
            datetime_obj = datetime.fromtimestamp(chunk.created, tz=timezone.utc)
            created_at = datetime_obj.isoformat()

            message = chunk.choices[0].delta.content
            done = False
            if chunk.choices[0].finish_reason == "stop":
                done = True
                message = ""

            # Convert the model response to an Ollama response
            ollama_response = ChatResponse(
                model=chunk.model,
                created_at=created_at,
                done=done,
                message=Message(content=message, role="assistant"),
            )
            return ollama_response
        except StopAsyncIteration:
            raise StopAsyncIteration


class OllamaOutputNormalizer(ModelOutputNormalizer):
    def __init__(self):
        super().__init__()

    def normalize_streaming(
        self,
        model_reply: AsyncIterator[ChatResponse],
    ) -> AsyncIterator[ModelResponse]:
        """
        Pass through Ollama response
        """
        return OLlamaToModel(model_reply)

    def normalize(self, model_reply: Any) -> Any:
        """
        Pass through Ollama response
        """
        return model_reply

    def denormalize(self, normalized_reply: Any) -> Any:
        """
        Pass through Ollama response
        """
        return normalized_reply

    def denormalize_streaming(
        self, normalized_reply: AsyncIterator[ModelResponse]
    ) -> AsyncIterator[ChatResponse]:
        """
        Pass through Ollama response
        """
        return ModelToOllama(normalized_reply)
