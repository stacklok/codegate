from typing import Optional

from codegate.providers.litellmshim.adapter import (
    LiteLLMAdapterInputNormalizer,
    LiteLLMAdapterOutputNormalizer,
)
from codegate.types.anthropic import (
    AnthropicExperimentalPassThroughConfig,
    AnthropicMessagesRequest,
    LitellmAnthropicAdapter,
)
from codegate.types.anthropic import ChatCompletionRequest


class AnthropicAdapter(LitellmAnthropicAdapter):
    def __init__(self) -> None:
        super().__init__()

    def translate_completion_input_params(self, kwargs) -> Optional[ChatCompletionRequest]:
        # request_body = AnthropicMessagesRequest(**kwargs)  # type: ignore
        # if not request_body.get("system"):
        #     request_body["system"] = "System prompt"
        # translated_body = (
        #     AnthropicExperimentalPassThroughConfig().translate_anthropic_to_openai(
        #         anthropic_message_request=request_body
        #     )
        # )
        translated_body = ChatCompletionRequest(**kwargs)
        return translated_body


class AnthropicInputNormalizer(LiteLLMAdapterInputNormalizer):
    """
    LiteLLM's adapter class interface is used to translate between the Anthropic data
    format and the underlying model. The AnthropicAdapter class contains the actual
    implementation of the interface methods, we just forward the calls to it.
    """

    def __init__(self):
        self.adapter = AnthropicAdapter()
        super().__init__(self.adapter)


class AnthropicOutputNormalizer(LiteLLMAdapterOutputNormalizer):
    """
    LiteLLM's adapter class interface is used to translate between the Anthropic data
    format and the underlying model. The AnthropicAdapter class contains the actual
    implementation of the interface methods, we just forward the calls to it.
    """

    def __init__(self):
        super().__init__(LitellmAnthropicAdapter())

    def denormalize_streaming(self, stream):
        return stream
