from abc import ABC, abstractmethod
from typing import Any, AsyncIterable, AsyncIterator, Dict, Iterable, Iterator, Union

from litellm import ChatCompletionRequest, ModelResponse


class ModelInputNormalizer(ABC):
    """
    The normalizer class is responsible for normalizing the input data
    before it is passed to the pipeline. It converts the input data (raw request)
    to the format expected by the pipeline.
    """

    @abstractmethod
    def normalize(self, data: Dict) -> ChatCompletionRequest:
        """Normalize the input data"""
        pass

    @abstractmethod
    def denormalize(self, data: ChatCompletionRequest) -> Dict:
        """Denormalize the input data"""
        pass


class ModelOutputNormalizer(ABC):
    """
    The output normalizer class is responsible for normalizing the output data
    from a model to the format expected by the output pipeline.

    The normalize methods are not implemented yet - they will be when we get
    around to implementing output pipelines.
    """

    @abstractmethod
    def normalize_streaming(
        self,
        model_reply: Union[AsyncIterable[Any], Iterable[Any]],
    ) -> Union[AsyncIterator[ModelResponse], Iterator[ModelResponse]]:
        """Normalize the output data"""
        pass

    @abstractmethod
    def normalize(self, model_reply: Any) -> ModelResponse:
        """Normalize the output data"""
        pass

    @abstractmethod
    def denormalize(self, normalized_reply: ModelResponse) -> Any:
        """Denormalize the output data"""
        pass

    @abstractmethod
    def denormalize_streaming(
        self,
        normalized_reply: Union[AsyncIterable[ModelResponse], Iterable[ModelResponse]],
    ) -> Union[AsyncIterator[Any], Iterator[Any]]:
        """Denormalize the output data"""
        pass
