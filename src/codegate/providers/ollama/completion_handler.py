import json
from typing import AsyncIterator, Optional, Union

import structlog
from fastapi.responses import JSONResponse, StreamingResponse
from litellm import ChatCompletionRequest
from ollama import AsyncClient, ChatResponse, GenerateResponse

from codegate.clients.clients import ClientType
from codegate.providers.base import BaseCompletionHandler

logger = structlog.get_logger("codegate")


async def ollama_stream_generator(  # noqa: C901
    stream: AsyncIterator[ChatResponse],
    client_type: ClientType,
) -> AsyncIterator[str]:
    """OpenAI-style SSE format"""
    try:
        async for chunk in stream:
            try:
                # TODO We should wire in the client info so we can respond with
                # the correct format and start to handle multiple clients
                # in a more robust way.
                if client_type in [ClientType.CLINE, ClientType.KODU]:
                    # First get the raw dict from the chunk
                    chunk_dict = chunk.model_dump()
                    # Create response dictionary in OpenAI-like format
                    response = {
                        "id": f"chatcmpl-{chunk_dict.get('created_at', '')}",
                        "object": "chat.completion.chunk",
                        "created": chunk_dict.get("created_at"),
                        "model": chunk_dict.get("model"),
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": chunk_dict.get("message", {}).get("content", ""),
                                    "role": chunk_dict.get("message", {}).get("role", "assistant"),
                                },
                                "finish_reason": (
                                    chunk_dict.get("done_reason")
                                    if chunk_dict.get("done", False)
                                    else None
                                ),
                            }
                        ],
                    }
                    # Preserve existing type or add default if missing
                    response["type"] = chunk_dict.get("type", "stream")

                    # Add optional fields that might be present in the final message
                    optional_fields = [
                        "total_duration",
                        "load_duration",
                        "prompt_eval_count",
                        "prompt_eval_duration",
                        "eval_count",
                        "eval_duration",
                    ]
                    for field in optional_fields:
                        if field in chunk_dict:
                            response[field] = chunk_dict[field]

                    yield f"\ndata: {json.dumps(response)}\n"
                else:
                    # if we do not have response, we set it
                    chunk_dict = chunk.model_dump()
                    if "response" not in chunk_dict:
                        chunk_dict["response"] = chunk_dict.get("message", {}).get("content", "\n")
                    if not chunk_dict["response"]:
                        chunk_dict["response"] = "\n"
                    yield f"{json.dumps(chunk_dict)}\n"
            except Exception as e:
                logger.error(f"Error in stream generator: {str(e)}")
                yield f"\ndata: {json.dumps({'error': str(e), 'type': 'error', 'choices': []})}\n"
    except Exception as e:
        logger.error(f"Stream error: {str(e)}")
        yield f"\ndata: {json.dumps({'error': str(e), 'type': 'error', 'choices': []})}\n"


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
        if not base_url:
            raise ValueError("base_url is required for Ollama")

        # TODO: Add CodeGate user agent.
        headers = dict()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        client = AsyncClient(host=base_url, timeout=300, headers=headers)

        try:
            if is_fim_request:
                prompt = ""
                for i in reversed(range(len(request["messages"]))):
                    if request["messages"][i]["role"] == "user":
                        prompt = request["messages"][i]["content"]  # type: ignore
                        break
                if not prompt:
                    raise ValueError("No user message found in FIM request")

                response = await client.generate(
                    model=request["model"],
                    prompt=prompt,
                    raw=request.get("raw", False),
                    suffix=request.get("suffix", ""),
                    stream=stream,
                    options=request["options"],  # type: ignore
                )
            else:
                response = await client.chat(
                    model=request["model"],
                    messages=request["messages"],
                    stream=stream,  # type: ignore
                    options=request["options"],  # type: ignore
                )  # type: ignore
            return response
        except Exception as e:
            logger.error(f"Error in Ollama completion: {str(e)}")
            raise e

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
