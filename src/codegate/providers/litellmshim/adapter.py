from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from litellm import ChatCompletionRequest, ModelResponse

from codegate.providers.base import StreamGenerator


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
    def translate_completion_input_params(
        self, kwargs: Dict
    ) -> Optional[ChatCompletionRequest]:
        """Convert input parameters to LiteLLM's ChatCompletionRequest format"""
        pass

    @abstractmethod
    def translate_completion_output_params(self, response: ModelResponse) -> Any:
        """Convert non-streaming response from LiteLLM ModelResponse format"""
        pass

    @abstractmethod
    def translate_completion_output_params_streaming(
        self, completion_stream: Any
    ) -> Any:
        """
        Convert streaming response from LiteLLM format to a format that
        can be passed to a stream generator and to the client.
        """
        pass
