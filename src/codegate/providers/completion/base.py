import inspect
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, AsyncIterator, Callable, Optional, Union

from fastapi.responses import JSONResponse, StreamingResponse

from codegate.clients.clients import ClientType


class BaseCompletionHandler(ABC):
    """
    The completion handler is responsible for executing the completion request
    and creating the streaming response.
    """

    @abstractmethod
    async def execute_completion(
        self,
        request: Any,
        base_url: Optional[str],
        api_key: Optional[str],
        stream: bool = False,  # TODO: remove this param?
        is_fim_request: bool = False,
    ) -> Union[Any, AsyncIterator[Any]]:
        """Execute the completion request"""
        pass

    @abstractmethod
    def _create_streaming_response(
        self,
        stream: AsyncIterator[Any],
        client_type: ClientType = ClientType.GENERIC,
        stream_generator: Callable | None = None,
    ) -> StreamingResponse:
        pass

    @abstractmethod
    def _create_json_response(self, response: Any) -> JSONResponse:
        pass

    def create_response(
        self,
        response: Any,
        client_type: ClientType,
        stream_generator: Callable | None = None,
    ) -> Union[JSONResponse, StreamingResponse]:
        """
        Create a FastAPI response from the completion response.
        """
        if (
            isinstance(response, Iterator)
            or isinstance(response, AsyncIterator)
            or inspect.isasyncgen(response)
        ):
            return self._create_streaming_response(
                response,
                client_type,
                stream_generator=stream_generator,
            )
        return self._create_json_response(response)
