import json
import asyncio
from typing import Any, AsyncIterator, Iterator, Optional, Union

from fastapi.responses import StreamingResponse
from litellm import ChatCompletionRequest, ModelResponse

from codegate.config import Config
from codegate.inference.inference_engine import LlamaCppInferenceEngine
from codegate.providers.base import BaseCompletionHandler


async def llamacpp_stream_generator(stream: Iterator[Any]) -> AsyncIterator[str]:
    """OpenAI-style SSE format"""
    try:
        for chunk in stream:
            if hasattr(chunk, "model_dump_json"):
                chunk = chunk.model_dump_json(exclude_none=True, exclude_unset=True)
            try:
                yield f"data:{json.dumps(chunk)}\n\n"
                await asyncio.sleep(0)
            except Exception as e:
                yield f"data:{str(e)}\n\n"
    except Exception as e:
        yield f"data: {str(e)}\n\n"
    finally:
        yield "data: [DONE]\n\n"

class LlamaCppCompletionHandler(BaseCompletionHandler):
    def __init__(self):
        self.inference_engine = LlamaCppInferenceEngine()

    async def execute_completion(
            self,
            request: ChatCompletionRequest,
            api_key: Optional[str],
            stream: bool = False
    ) -> Union[ModelResponse, AsyncIterator[ModelResponse]]:
        """
        Execute the completion request with inference engine API
        """
        model_path = f"{Config.get_config().model_base_path}/{request['model']}.gguf"

        if 'prompt' in request:
            response = await self.inference_engine.complete(model_path,
                                                        Config.get_config().chat_model_n_ctx,
                                                        Config.get_config().chat_model_n_gpu_layers,
                                                        **request)
        else:
            response = await self.inference_engine.chat(model_path,
                                                        Config.get_config().chat_model_n_ctx,
                                                        Config.get_config().chat_model_n_gpu_layers,
                                                        **request)
        return response

    def create_streaming_response(
        self, stream: Iterator[Any]
    ) -> StreamingResponse:
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
