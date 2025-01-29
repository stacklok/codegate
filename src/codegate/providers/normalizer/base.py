from abc import ABC, abstractmethod
from typing import Any, AsyncIterable, AsyncIterator, Dict, Iterable, Iterator, Union

from litellm import ChatCompletionRequest, ModelResponse


class ModelInputNormalizer(ABC):
    """
    The normalizer class is responsible for normalizing the input data
    before it is passed to the pipeline. It converts the input data (raw request)
    to the format expected by the pipeline.
    """

    def _normalize_content_messages(self, data: Dict) -> Dict:
        """
        If the request contains the "messages" key, make sure that it's content is a string.
        """
        # Anyways copy the original data to avoid modifying it
        if "messages" not in data:
            return data.copy()

        normalized_data = data.copy()
        messages = normalized_data["messages"]
        converted_messages = []
        for msg in messages:
            new_msg = msg.copy()
            content = msg.get("content", "")
            if isinstance(content, list):
                # Convert list format to string
                content_parts = []
                for part in msg["content"]:
                    if isinstance(part, dict) and part.get("type") == "text":
                        content_parts.append(part["text"])
                new_msg["content"] = " ".join(content_parts)
            converted_messages.append(new_msg)
        normalized_data["messages"] = converted_messages
        return normalized_data

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
