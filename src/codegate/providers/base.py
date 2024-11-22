from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Callable, Dict

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

StreamGenerator = Callable[[AsyncIterator[Any]], AsyncIterator[str]]


class BaseCompletionHandler(ABC):
    """
    The completion handler is responsible for executing the completion request
    and creating the streaming response.
    """

    @abstractmethod
    async def complete(self, data: Dict, api_key: str) -> AsyncIterator[Any]:
        pass

    @abstractmethod
    def create_streaming_response(
        self, stream: AsyncIterator[Any]
    ) -> StreamingResponse:
        pass


class BaseProvider(ABC):
    """
    The provider class is responsible for defining the API routes and
    calling the completion method using the completion handler.
    """

    def __init__(self, completion_handler: BaseCompletionHandler):
        self.router = APIRouter()
        self._completion_handler = completion_handler
        self._setup_routes()

    @abstractmethod
    def _setup_routes(self) -> None:
        pass

    async def complete(self, data: Dict, api_key: str) -> AsyncIterator[Any]:
        return await self._completion_handler.complete(data, api_key)

    def get_routes(self) -> APIRouter:
        return self.router
