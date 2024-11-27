from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Optional, Union

from fastapi.responses import StreamingResponse
from litellm import ChatCompletionRequest, ModelResponse


class BaseCompletionHandler(ABC):
    """
    The completion handler is responsible for executing the completion request
    and creating the streaming response.
    """

    @abstractmethod
    async def execute_completion(
        self,
        request: ChatCompletionRequest,
        api_key: Optional[str],
        stream: bool = False, # TODO: remove this param?
    ) -> Union[ModelResponse, AsyncIterator[ModelResponse]]:
        """Execute the completion request"""
        pass

    @abstractmethod
    def create_streaming_response(
        self, stream: AsyncIterator[Any]
    ) -> StreamingResponse:
        pass
