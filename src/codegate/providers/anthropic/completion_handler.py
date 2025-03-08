from typing import AsyncIterator, Optional, Union

from codegate.providers.litellmshim import LiteLLmShim
from codegate.types.common import ChatCompletionRequest, ModelResponse


class AnthropicCompletion(LiteLLmShim):
    """
    AnthropicCompletion used by the Anthropic provider to execute completions
    """

    async def execute_completion(
        self,
        request: ChatCompletionRequest,
        base_url: Optional[str],
        api_key: Optional[str],
        stream: bool = False,
        is_fim_request: bool = False,
    ) -> Union[ModelResponse, AsyncIterator[ModelResponse]]:
        """
        Ensures the model name is prefixed with 'anthropic/' to explicitly route to Anthropic's API.

        LiteLLM automatically maps most model names, but prepending 'anthropic/' forces the request
        to Anthropic.  This avoids issues with unrecognized names like 'claude-3-5-sonnet-latest',
        which LiteLLM doesn't accept as a valid Anthropic model. This safeguard may be unnecessary
        but ensures compatibility.

        For more details, refer to the
        [LiteLLM Documentation](https://docs.litellm.ai/docs/providers/anthropic).
        """
        return await super().execute_completion(
            request,
            base_url,
            api_key,
            stream,
            is_fim_request,
        )
