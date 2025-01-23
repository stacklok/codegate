from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Optional, Tuple, Union

from ollama import ChatResponse, Message

from codegate.providers.normalizer.base import ModelInputNormalizer, ModelOutputNormalizer
from codegate.types.common import (
    Delta,
    ModelResponse,
    StreamingChoices,
)


class OLlamaToModel(AsyncIterator[ModelResponse]):
    def __init__(self, ollama_response: AsyncIterator[ChatResponse]):
        self.ollama_response = ollama_response
        self._aiter = ollama_response.__aiter__()

    @classmethod
    def _transform_to_int_secs(cls, chunk_created_at: str) -> int:
        """
        Convert the datetime to a timestamp in seconds.
        """
        datetime_obj = datetime.fromisoformat(chunk_created_at)
        return int(datetime_obj.timestamp())

    @classmethod
    def _get_finish_reason_assistant(cls, is_chunk_done: bool) -> Tuple[str, Optional[str]]:
        """
        Get the role and finish reason for the assistant based on the chunk done status.
        """
        finish_reason = None
        role = "assistant"
        if is_chunk_done:
            finish_reason = "stop"
            role = None
        return role, finish_reason

    @classmethod
    def _get_chat_id_from_timestamp(cls, timestamp_seconds: int) -> str:
        """
        Getting a string representation of the timestamp in seconds used as the chat id.

        This needs to be done so that all chunks of a chat have the same id.
        """
        timestamp_str = str(timestamp_seconds)
        return timestamp_str[:9]

    @classmethod
    def normalize_chat_chunk(cls, chunk: ChatResponse) -> ModelResponse:
        """
        Transform an ollama chat chunk to an OpenAI one
        """
        timestamp_seconds = cls._transform_to_int_secs(chunk.created_at)
        role, finish_reason = cls._get_finish_reason_assistant(chunk.done)
        chat_id = cls._get_chat_id_from_timestamp(timestamp_seconds)

        model_response = ModelResponse(
            id=f"ollama-chat-{chat_id}",
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

    @classmethod
    def normalize_fim_chunk(cls, chunk) -> Dict:
        """
        Transform an ollama generation chunk to an OpenAI one
        """
        timestamp_seconds = cls._transform_to_int_secs(chunk.created_at)
        _, finish_reason = cls._get_finish_reason_assistant(chunk.done)
        chat_id = cls._get_chat_id_from_timestamp(timestamp_seconds)

        model_response = {
            "id": f"chatcmpl-{chat_id}",
            "object": "text_completion",
            "created": timestamp_seconds,
            "model": chunk.model,
            "choices": [{"index": 0, "text": chunk.response}],
            "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
        }
        if finish_reason:
            model_response["choices"][0]["finish_reason"] = finish_reason
            del model_response["choices"][0]["text"]
        return model_response

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await self._aiter.__anext__()
            if isinstance(chunk, ChatResponse):
                return self.normalize_chat_chunk(chunk)
            return chunk
        except StopAsyncIteration:
            raise StopAsyncIteration
