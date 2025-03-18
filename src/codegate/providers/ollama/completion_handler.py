from typing import (
    AsyncIterator,
    Callable,
    Optional,
    Union,
)

import structlog
from fastapi.responses import JSONResponse, StreamingResponse
from ollama import ChatResponse, GenerateResponse

from codegate.clients.clients import ClientType
from codegate.providers.base import BaseCompletionHandler
from codegate.types.ollama import (
    StreamingChatCompletion,
    StreamingGenerateCompletion,
    chat_streaming,
    generate_streaming,
)
from codegate.types.ollama import (
    stream_generator as ollama_stream_generator,
)
from codegate.types.openai import (
    ChatCompletionRequest,
    completions_streaming,
)
from codegate.types.openai import (
    StreamingChatCompletion as OpenAIStreamingChatCompletion,
)
from codegate.types.openai import (
    stream_generator as openai_stream_generator,
)

logger = structlog.get_logger("codegate")


T = Union[
    StreamingChatCompletion,
    StreamingGenerateCompletion,
    OpenAIStreamingChatCompletion,
]


async def prepend(
    first: T,
    stream: AsyncIterator[T],
) -> AsyncIterator[T]:
    yield first
    async for item in stream:
        yield item


async def _ollama_dispatcher(  # noqa: C901
    stream: AsyncIterator[T],
) -> AsyncIterator[str]:
    """OpenAI-style SSE format"""
    first = await anext(stream)

    if isinstance(first, StreamingChatCompletion):
        stream = ollama_stream_generator(prepend(first, stream))

    if isinstance(first, StreamingGenerateCompletion):
        stream = ollama_stream_generator(prepend(first, stream))

    if isinstance(first, OpenAIStreamingChatCompletion):
        stream = openai_stream_generator(prepend(first, stream))

    async for item in stream:
        yield item


class OllamaShim(BaseCompletionHandler):

    async def execute_completion(
        self,
        request,
        base_url: Optional[str],
        api_key: Optional[str],
        stream: bool = False,
        is_fim_request: bool = False,
    ) -> Union[ChatResponse, GenerateResponse]:
        """Stream response directly from Ollama API."""
        if isinstance(request, ChatCompletionRequest):  # case for OpenAI-style requests
            return completions_streaming(request, api_key, base_url)
        if is_fim_request:
            return generate_streaming(request, api_key, base_url)
        return chat_streaming(request, api_key, base_url)

    def _create_streaming_response(
        self,
        stream: AsyncIterator[ChatResponse],
        client_type: ClientType,
        stream_generator: Callable | None = None,
    ) -> StreamingResponse:
        """
        Create a streaming response from a stream generator. The StreamingResponse
        is the format that FastAPI expects for streaming responses.
        """
        return StreamingResponse(
            stream_generator(stream) if stream_generator else _ollama_dispatcher(stream),
            media_type="application/x-ndjson; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Transfer-Encoding": "chunked",
            },
            status_code=200,
        )

    def _create_json_response(
        self, response: Union[GenerateResponse, ChatResponse]
    ) -> JSONResponse:
        return JSONResponse(status_code=200, content=response.model_dump())
