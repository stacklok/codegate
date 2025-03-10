from typing import Any, AsyncIterator, Optional, Union

from codegate.providers.litellmshim import LiteLLmShim


class AnthropicCompletion(LiteLLmShim):
    """
    AnthropicCompletion used by the Anthropic provider to execute completions
    """

    async def execute_completion(
        self,
        request: Any,
        base_url: Optional[str],
        api_key: Optional[str],
        stream: bool = False,
        is_fim_request: bool = False,
    ) -> Union[Any, AsyncIterator[Any]]:
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
