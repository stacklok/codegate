from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict

from ..types import Message, NormalizedRequest, ChatResponse

class ModelInputNormalizer(ABC):
    @abstractmethod
    def normalize(self, data: Dict[str, Any]) -> NormalizedRequest:
        """Convert provider-specific request format to SimpleModelRouter format."""
        pass

    @abstractmethod
    def denormalize(self, data: NormalizedRequest) -> Dict[str, Any]:
        """Convert SimpleModelRouter format back to provider-specific request format."""
        pass

class ModelOutputNormalizer(ABC):
    @abstractmethod
    def normalize_streaming(
        self,
        model_reply: AsyncIterator[Any]
    ) -> AsyncIterator[ChatResponse]:
        """Convert provider-specific streaming response to SimpleModelRouter format."""
        pass

    @abstractmethod
    def normalize(self, model_reply: Any) -> ChatResponse:
        """Convert provider-specific response to SimpleModelRouter format."""
        pass

    @abstractmethod
    def denormalize(self, normalized_reply: ChatResponse) -> Dict[str, Any]:
        """Convert SimpleModelRouter format back to provider-specific response format."""
        pass

    @abstractmethod
    def denormalize_streaming(
        self,
        normalized_reply: AsyncIterator[ChatResponse]
    ) -> AsyncIterator[Any]:
        """Convert SimpleModelRouter streaming response back to provider-specific format."""
        pass
