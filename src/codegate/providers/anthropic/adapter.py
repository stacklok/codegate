from typing import Any, Dict, Optional

from litellm import AdapterCompletionStreamWrapper, ChatCompletionRequest, ModelResponse
from litellm.adapters.anthropic_adapter import (
    AnthropicAdapter as LitellmAnthropicAdapter,
)
from litellm.types.llms.anthropic import AnthropicResponse

from ..base import StreamGenerator
from ..litellmshim import anthropic_stream_generator
from ..litellmshim.litellmshim import BaseAdapter


class AnthropicAdapter(BaseAdapter):
    """
    LiteLLM's adapter class interface is used to translate between the Anthropic data
    format and the underlying model. The AnthropicAdapter class contains the actual
    implementation of the interface methods, we just forward the calls to it.
    """
    def __init__(self, stream_generator: StreamGenerator = anthropic_stream_generator):
        self.litellm_anthropic_adapter = LitellmAnthropicAdapter()
        super().__init__(stream_generator)

    def translate_completion_input_params(
        self,
        completion_request: Dict,
    ) -> Optional[ChatCompletionRequest]:
        return self.litellm_anthropic_adapter.translate_completion_input_params(
            completion_request
        )

    def translate_completion_output_params(
        self, response: ModelResponse
    ) -> Optional[AnthropicResponse]:
        return self.litellm_anthropic_adapter.translate_completion_output_params(
            response
        )

    def translate_completion_output_params_streaming(
        self, completion_stream: Any
    ) -> AdapterCompletionStreamWrapper | None:
        return (
            self.litellm_anthropic_adapter.translate_completion_output_params_streaming(
                completion_stream
            )
        )
