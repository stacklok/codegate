from litellm.adapters.anthropic_adapter import (
    AnthropicAdapter as LitellmAnthropicAdapter,
)

from codegate.providers.litellmshim.adapter import (
    LiteLLMAdapterInputNormalizer,
    LiteLLMAdapterOutputNormalizer,
)


class AnthropicInputNormalizer(LiteLLMAdapterInputNormalizer):
    """
    LiteLLM's adapter class interface is used to translate between the Anthropic data
    format and the underlying model. The AnthropicAdapter class contains the actual
    implementation of the interface methods, we just forward the calls to it.
    """

    def __init__(self):
        super().__init__(LitellmAnthropicAdapter())

class AnthropicOutputNormalizer(LiteLLMAdapterOutputNormalizer):
    """
    LiteLLM's adapter class interface is used to translate between the Anthropic data
    format and the underlying model. The AnthropicAdapter class contains the actual
    implementation of the interface methods, we just forward the calls to it.
    """

    def __init__(self):
        super().__init__(LitellmAnthropicAdapter())
