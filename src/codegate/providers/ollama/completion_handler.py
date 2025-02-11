import json
import os
from typing import AsyncIterator, Optional, Union

import httpx
import structlog
from fastapi.responses import JSONResponse, StreamingResponse
from ollama import AsyncClient, ChatResponse, GenerateResponse

from codegate.clients.clients import ClientType
from codegate.providers.base import BaseCompletionHandler
from codegate.types.common import ChatCompletionRequest
from codegate.types.ollama import chat_streaming, generate_streaming


logger = structlog.get_logger("codegate")


async def ollama_stream_generator(  # noqa: C901
    stream: AsyncIterator[ChatResponse],
    client_type: ClientType,
) -> AsyncIterator[str]:
    """OpenAI-style SSE format"""
    try:
        async for chunk in stream:
            try:
                # if we do not have response, we set it
                body = chunk.model_dump_json(exclude_none=True, exclude_unset=True)
                if os.getenv("CODEGATE_DEBUG_OLLAMA") is not None:
                    print(body)
                yield f"{body}\n"
            except Exception as e:
                logger.error("failed serializing payload", exc_info=e)
                yield f"\ndata: {json.dumps({'error': str(e), 'type': 'error', 'choices': []})}\n"
    except Exception as e:
        logger.error("failed generating output payloads", exc_info=e)
        yield f"\ndata: {json.dumps({'error': str(e)})}\n"


class OllamaShim(BaseCompletionHandler):

    async def execute_completion(
        self,
        request: ChatCompletionRequest,
        base_url: Optional[str],
        api_key: Optional[str],
        stream: bool = False,
        is_fim_request: bool = False,
    ) -> Union[ChatResponse, GenerateResponse]:
        """Stream response directly from Ollama API."""
        if is_fim_request:
            return await generate_streaming(request, api_key, base_url)
        return await chat_streaming(request, api_key, base_url)

    def _create_streaming_response(
        self,
        stream: AsyncIterator[ChatResponse],
        client_type: ClientType,
    ) -> StreamingResponse:
        """
        Create a streaming response from a stream generator. The StreamingResponse
        is the format that FastAPI expects for streaming responses.
        """
        return StreamingResponse(
            ollama_stream_generator(stream, client_type),
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
