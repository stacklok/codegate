import asyncio
import json
from typing import Any, AsyncIterator, Iterator, Optional, Union

from fastapi.responses import JSONResponse, StreamingResponse
from litellm import ChatCompletionRequest, ModelResponse
from llama_cpp.llama_types import (
    CreateChatCompletionStreamResponse,
)

from codegate.config import Config
from codegate.inference.inference_engine import LlamaCppInferenceEngine
from codegate.providers.base import BaseCompletionHandler


async def llamacpp_stream_generator(
    stream: AsyncIterator[CreateChatCompletionStreamResponse],
) -> AsyncIterator[str]:
    """OpenAI-style SSE format"""
    try:
        async for chunk in stream:
            chunk = json.dumps(chunk)
            try:
                yield f"data:{chunk}\n\n"
            except Exception as e:
                yield f"data:{str(e)}\n\n"
    except Exception as e:
        yield f"data: {str(e)}\n\n"
    finally:
        yield "data: [DONE]\n\n"


async def convert_to_async_iterator(
    sync_iterator: Iterator[CreateChatCompletionStreamResponse],
) -> AsyncIterator[CreateChatCompletionStreamResponse]:
    """
    Convert a synchronous iterator to an asynchronous iterator. This makes the logic easier
    because both the pipeline and the completion handler can use async iterators.
    """
    for item in sync_iterator:
        yield item
        await asyncio.sleep(0)


class LlamaCppCompletionHandler(BaseCompletionHandler):
    def __init__(self):
        self.inference_engine = LlamaCppInferenceEngine()

    async def execute_completion(
        self,
        request: ChatCompletionRequest,
        api_key: Optional[str],
        stream: bool = False,
        is_fim_request: bool = False,
        base_tool: Optional[str] = "",
    ) -> Union[ModelResponse, AsyncIterator[ModelResponse]]:
        """
        Execute the completion request with inference engine API
        """
        model_path = f"{Config.get_config().model_base_path}/{request['model']}.gguf"

        # Create a copy of the request dict and remove stream_options
        # Reason - Request error as JSON:
        # {'error': "Llama.create_completion() got an unexpected keyword argument 'stream_options'"}
        request_dict = dict(request)
        request_dict.pop("stream_options", None)

        if is_fim_request:
            response = await self.inference_engine.complete(
                model_path,
                Config.get_config().chat_model_n_ctx,
                Config.get_config().chat_model_n_gpu_layers,
                **request_dict,
            )
        else:
            response = await self.inference_engine.chat(
                model_path,
                Config.get_config().chat_model_n_ctx,
                Config.get_config().chat_model_n_gpu_layers,
                **request_dict,
            )

        return convert_to_async_iterator(response) if stream else response

    def _create_streaming_response(self, stream: AsyncIterator[Any]) -> StreamingResponse:
        """
        Create a streaming response from a stream generator. The StreamingResponse
        is the format that FastAPI expects for streaming responses.
        """
        return StreamingResponse(
            llamacpp_stream_generator(stream),
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Transfer-Encoding": "chunked",
            },
            status_code=200,
        )

    def _create_json_response(self, response: Any) -> JSONResponse:
        return JSONResponse(status_code=200, content=response)
