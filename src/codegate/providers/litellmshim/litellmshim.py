from typing import Any, AsyncIterator, Dict

from fastapi.responses import StreamingResponse
from litellm import ModelResponse, acompletion

from ..base import BaseCompletionHandler
from .adapter import BaseAdapter


class LiteLLmShim(BaseCompletionHandler):
    """
    LiteLLM Shim is a wrapper around LiteLLM's API that allows us to use it with
    our own completion handler interface without exposing the underlying
    LiteLLM API.
    """

    def __init__(self, adapter: BaseAdapter, completion_func=acompletion):
        self._adapter = adapter
        self._completion_func = completion_func

    async def complete(self, data: Dict, api_key: str) -> AsyncIterator[Any]:
        """
        Translate the input parameters to LiteLLM's format using the adapter and
        call the LiteLLM API. Then translate the response back to our format using
        the adapter.
        """
        data["api_key"] = api_key
        completion_request = self._adapter.translate_completion_input_params(data)
        if completion_request is None:
            raise Exception("Couldn't translate the request")

        response = await self._completion_func(**completion_request)

        if isinstance(response, ModelResponse):
            return self._adapter.translate_completion_output_params(response)
        return self._adapter.translate_completion_output_params_streaming(response)

    def create_streaming_response(
        self, stream: AsyncIterator[Any]
    ) -> StreamingResponse:
        """
        Create a streaming response from a stream generator. The StreamingResponse
        is the format that FastAPI expects for streaming responses.
        """
        return StreamingResponse(
            self._adapter.stream_generator(stream),
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Transfer-Encoding": "chunked",
            },
            status_code=200,
        )
