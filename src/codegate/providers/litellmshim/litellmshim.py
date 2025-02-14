from typing import Any, AsyncIterator, Callable, Optional, Union

import litellm
import structlog
from fastapi.responses import JSONResponse, StreamingResponse

from codegate.clients.clients import ClientType
from codegate.providers.base import BaseCompletionHandler, StreamGenerator
from codegate.types.common import (
    ChatCompletionRequest,
    ModelResponse,
)
from codegate.types.anthropic import acompletion


logger = structlog.get_logger("codegate")
litellm.drop_params = True


class LiteLLmShim(BaseCompletionHandler):
    """
    LiteLLM Shim is a wrapper around LiteLLM's API that allows us to use it with
    our own completion handler interface without exposing the underlying
    LiteLLM API.
    """

    def __init__(
        self,
        stream_generator: StreamGenerator,
        completion_func: Callable = acompletion,
        fim_completion_func: Optional[Callable] = None,
    ):
        self._stream_generator = stream_generator
        self._completion_func = completion_func
        # Use the same function for FIM completion if one is not specified
        if fim_completion_func is None:
            self._fim_completion_func = completion_func
        else:
            self._fim_completion_func = fim_completion_func

    async def execute_completion(
        self,
        request: ChatCompletionRequest,
        base_url: Optional[str],
        api_key: Optional[str],
        stream: bool = False,
        is_fim_request: bool = False,
    ) -> Union[ModelResponse, AsyncIterator[ModelResponse]]:
        """
        Execute the completion request with LiteLLM's API
        """
        if is_fim_request:
            return self._fim_completion_func(request, api_key=api_key, base_url=base_url)
        return self._completion_func(request, api_key=api_key, base_url=base_url)

    def _create_streaming_response(
        self,
        stream: AsyncIterator[Any],
        _: ClientType = ClientType.GENERIC,
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

    def _create_json_response(self, response: ModelResponse) -> JSONResponse:
        """
        Create a JSON FastAPI response from a ModelResponse object.
        ModelResponse is obtained when the request is not streaming.
        """
        # ModelResponse is not a Pydantic object but has a json method we can use to serialize
        if isinstance(response, ModelResponse):
            return JSONResponse(status_code=200, content=response.json())
        # Most of others objects in LiteLLM are Pydantic, we can use the model_dump method
        return JSONResponse(status_code=200, content=response.model_dump())
