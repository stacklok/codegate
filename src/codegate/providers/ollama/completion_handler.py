from typing import AsyncIterator, Optional, Union

import structlog
from fastapi.responses import JSONResponse, StreamingResponse
from litellm import ChatCompletionRequest
from ollama import AsyncClient, ChatResponse, GenerateResponse

from codegate.providers.base import BaseCompletionHandler

logger = structlog.get_logger("codegate")


async def ollama_stream_generator(stream: AsyncIterator[ChatResponse]) -> AsyncIterator[str]:
    """OpenAI-style SSE format"""
    try:
        async for chunk in stream:
            try:
                content = chunk.model_dump_json()
                if content:
                    yield f"{chunk.model_dump_json()}\n"
            except Exception as e:
                if str(e):
                    yield f"{str(e)}\n"
    except Exception as e:
        if str(e):
            yield f"{str(e)}\n"


class OllamaShim(BaseCompletionHandler):

    def __init__(self, base_url):
        self.client = AsyncClient(host=base_url, timeout=300)

    async def execute_completion(
        self,
        request: ChatCompletionRequest,
        api_key: Optional[str],
        stream: bool = False,
        is_fim_request: bool = False,
    ) -> Union[ChatResponse, GenerateResponse]:
        """Stream response directly from Ollama API."""
        if is_fim_request:
            prompt = request["messages"][0].get("content", "")
            response = await self.client.generate(
                model=request["model"], prompt=prompt, stream=stream, options=request["options"]  # type: ignore
            )
        else:
            response = await self.client.chat(
                model=request["model"],
                messages=request["messages"],
                stream=stream,  # type: ignore
                options=request["options"],  # type: ignore
            )  # type: ignore
        return response

    def _create_streaming_response(self, stream: AsyncIterator[ChatResponse]) -> StreamingResponse:
        """
        Create a streaming response from a stream generator. The StreamingResponse
        is the format that FastAPI expects for streaming responses.
        """
        return StreamingResponse(
            ollama_stream_generator(stream),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    def _create_json_response(
        self, response: Union[GenerateResponse, ChatResponse]
    ) -> JSONResponse:
        return JSONResponse(content=response.model_dump_json(), status_code=200)
