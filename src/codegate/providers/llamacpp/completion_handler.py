from typing import Any, AsyncIterator, Callable, Iterator, Optional, Union

from fastapi.responses import JSONResponse, StreamingResponse

from codegate.clients.clients import ClientType
from codegate.config import Config
from codegate.inference.inference_engine import LlamaCppInferenceEngine
from codegate.providers.base import BaseCompletionHandler
from codegate.types.openai import (
    LegacyCompletion,
    StreamingChatCompletion,
)
from codegate.types.openai import (
    stream_generator as openai_stream_generator,
)

# async def llamacpp_stream_generator(
#     stream: AsyncIterator[CreateChatCompletionStreamResponse],
# ) -> AsyncIterator[str]:
#     """OpenAI-style SSE format"""
#     try:
#         async for chunk in stream:
#             chunk = json.dumps(chunk)
#             try:
#                 yield f"data:{chunk}\n\n"
#             except Exception as e:
#                 yield f"data:{str(e)}\n\n"
#     except Exception as e:
#         yield f"data: {str(e)}\n\n"
#     finally:
#         yield "data: [DONE]\n\n"


async def completion_to_async_iterator(
    sync_iterator: Iterator[dict],
) -> AsyncIterator[LegacyCompletion]:
    """
    Convert a synchronous iterator to an asynchronous iterator. This makes the logic easier
    because both the pipeline and the completion handler can use async iterators.
    """
    for item in sync_iterator:
        yield LegacyCompletion(**item)


async def chat_to_async_iterator(
    sync_iterator: Iterator[dict],
) -> AsyncIterator[StreamingChatCompletion]:
    for item in sync_iterator:
        yield StreamingChatCompletion(**item)


ENGINE = LlamaCppInferenceEngine()


async def complete(request, api_key, model_path):
    stream = request.get_stream()
    full_path = f"{model_path}/{request.get_model()}.gguf"
    request_dict = request.dict(
        exclude={
            "best_of",
            "frequency_penalty",
            "n",
            "stream_options",
            "user",
        }
    )

    response = await ENGINE.complete(
        full_path,
        Config.get_config().chat_model_n_ctx,
        Config.get_config().chat_model_n_gpu_layers,
        **request_dict,
    )

    if stream:
        return completion_to_async_iterator(response)
    # TODO fix this code path is broken
    return LegacyCompletion(**response)


async def chat(request, api_key, model_path):
    stream = request.get_stream()
    full_path = f"{model_path}/{request.get_model()}.gguf"
    request_dict = request.dict(
        exclude={
            "audio",
            "frequency_penalty",
            "include_reasoning",
            "metadata",
            "max_completion_tokens",
            "modalities",
            "n",
            "parallel_tool_calls",
            "prediction",
            "prompt",
            "reasoning_effort",
            "service_tier",
            "store",
            "stream_options",
            "user",
        }
    )

    response = await ENGINE.chat(
        full_path,
        Config.get_config().chat_model_n_ctx,
        Config.get_config().chat_model_n_gpu_layers,
        **request_dict,
    )

    if stream:
        return chat_to_async_iterator(response)
    else:
        # TODO fix this code path is broken
        return StreamingChatCompletion(**response)


class LlamaCppCompletionHandler(BaseCompletionHandler):
    def __init__(self, base_url):
        self.inference_engine = ENGINE
        self.base_url = base_url

    async def execute_completion(
        self,
        request: Any,
        base_url: Optional[str],
        api_key: Optional[str],
        stream: bool = False,
        is_fim_request: bool = False,
    ) -> Union[Any, AsyncIterator[Any]]:
        """
        Execute the completion request with inference engine API
        """
        # Create a copy of the request dict and remove stream_options
        # Reason - Request error as JSON:
        # {'error': "Llama.create_completion() got an unexpected keyword argument 'stream_options'"}
        if is_fim_request:
            # base_url == model_path in this case
            return await complete(request, api_key, self.base_url)
        else:
            # base_url == model_path in this case
            return await chat(request, api_key, self.base_url)

    def _create_streaming_response(
        self,
        stream: AsyncIterator[Any],
        client_type: ClientType = ClientType.GENERIC,
        stream_generator: Callable | None = None,
    ) -> StreamingResponse:
        """
        Create a streaming response from a stream generator. The StreamingResponse
        is the format that FastAPI expects for streaming responses.
        """
        return StreamingResponse(
            stream_generator(stream) if stream_generator else openai_stream_generator(stream),
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Transfer-Encoding": "chunked",
            },
            status_code=200,
        )

    def _create_json_response(self, response: Any) -> JSONResponse:
        return JSONResponse(status_code=200, content=response)
