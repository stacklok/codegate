from abc import ABC, abstractmethod
from typing import Any, AsyncIterable, AsyncIterator, Dict, Iterable, Iterator, Optional, Union

import structlog

from codegate.providers.base import StreamGenerator
from codegate.providers.normalizer.base import ModelInputNormalizer, ModelOutputNormalizer
from codegate.types.common import CodegateModelResponseStream, ModelResponse
from codegate.types.anthropic import ChatCompletionRequest


logger = structlog.get_logger("codegate")


class BaseAdapter(ABC):
    """
    The adapter class is responsible for translating input and output
    parameters between the provider-specific on-the-wire API and the
    underlying model. We use LiteLLM's ChatCompletionRequest and ModelResponse
    is our data model.

    The methods in this class implement LiteLLM's Adapter interface and are
    not our own. This is to allow us to use LiteLLM's adapter classes as a
    drop-in replacement for our own adapters.
    """

    def __init__(self, stream_generator: StreamGenerator):
        self.stream_generator = stream_generator

    @abstractmethod
    def translate_completion_input_params(self, kwargs: Dict) -> Optional[ChatCompletionRequest]:
        """Convert input parameters to LiteLLM's ChatCompletionRequest format"""
        pass

    @abstractmethod
    def translate_completion_output_params(self, response: ModelResponse) -> Any:
        """Convert non-streaming response from LiteLLM ModelResponse format"""
        pass

    @abstractmethod
    def translate_completion_output_params_streaming(self, completion_stream: Any) -> Any:
        """
        Convert streaming response from LiteLLM format to a format that
        can be passed to a stream generator and to the client.
        """
        pass


class LiteLLMAdapterInputNormalizer(ModelInputNormalizer):
    def __init__(self, adapter: BaseAdapter):
        self._adapter = adapter

    def normalize(self, data: Dict) -> ChatCompletionRequest:
        """
        Uses an LiteLLM adapter to translate the request data from the native
        LLM format to the OpenAI API format used by LiteLLM internally.
        """
        # Make a copy of the data to avoid modifying the original and normalize the message content
        normalized_data = self._normalize_content_messages(data)
        ret = self._adapter.translate_completion_input_params(normalized_data)

        return ret

    def denormalize(self, data: ChatCompletionRequest) -> Dict:
        """
        For LiteLLM, we don't have to de-normalize as the input format is
        always ChatCompletionRequest which is a TypedDict which is a Dict
        """

        return data


class LiteLLMAdapterOutputNormalizer(ModelOutputNormalizer):
    def __init__(self, adapter: BaseAdapter):
        self._adapter = adapter

    def normalize_streaming(
        self,
        model_reply: Union[AsyncIterable[Any], Iterable[Any]],
    ) -> Union[AsyncIterator[ModelResponse], Iterator[ModelResponse]]:
        """
        Normalize the output stream. This is a pass-through for liteLLM output normalizer
        as the liteLLM output is already in the normalized format.
        """
        return model_reply

    def normalize(self, model_reply: Any) -> ModelResponse:
        """
        Normalize the output data. This is a pass-through for liteLLM output normalizer
        as the liteLLM output is already in the normalized format.
        """
        return model_reply

    def denormalize(self, normalized_reply: ModelResponse) -> Any:
        """
        Denormalize the output data from the completion function to the format
        expected by the client
        """
        if isinstance(normalized_reply, CodegateModelResponseStream):
            return normalized_reply
        return self._adapter.translate_completion_output_params(normalized_reply)

    def denormalize_streaming(
        self,
        normalized_reply: Union[AsyncIterable[ModelResponse], Iterable[ModelResponse]],
    ) -> Union[AsyncIterator[Any], Iterator[Any]]:
        """
        Denormalize the output stream from the completion function to the format
        expected by the client
        """
        if isinstance(normalized_reply, CodegateModelResponseStream):
            return normalized_reply
        return self._adapter.translate_completion_output_params_streaming(normalized_reply)
