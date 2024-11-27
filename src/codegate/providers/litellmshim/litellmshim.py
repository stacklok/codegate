from typing import Any, AsyncIterator, Optional, Union

from fastapi.responses import StreamingResponse
from litellm import ChatCompletionRequest, ModelResponse, acompletion

from codegate.providers.base import BaseCompletionHandler, StreamGenerator


class LiteLLmShim(BaseCompletionHandler):
    """
    LiteLLM Shim is a wrapper around LiteLLM's API that allows us to use it with
    our own completion handler interface without exposing the underlying
    LiteLLM API.
    """

    def __init__(self, stream_generator: StreamGenerator, completion_func=acompletion):
        self._stream_generator = stream_generator
        self._completion_func = completion_func

    async def execute_completion(
        self,
        request: ChatCompletionRequest,
        api_key: Optional[str],
        stream: bool = False,
    ) -> Union[ModelResponse, AsyncIterator[ModelResponse]]:
        """
        Execute the completion request with LiteLLM's API
        """
        request["api_key"] = api_key
        return await self._completion_func(**request)

    def create_streaming_response(
        self, stream: AsyncIterator[Any]
    ) -> StreamingResponse:
        """
        Create a streaming response from a stream generator. The StreamingResponse
        is the format that FastAPI expects for streaming responses.
        """
        return StreamingResponse(
            self._stream_generator(stream),
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Transfer-Encoding": "chunked",
            },
            status_code=200,
        )
