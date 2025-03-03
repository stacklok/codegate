from typing import AsyncIterator, Optional, Union

import structlog
from litellm import ChatCompletionRequest, ModelResponse

from codegate.providers.litellmshim import LiteLLmShim

logger = structlog.get_logger("codegate")


class GeminiCompletion(LiteLLmShim):
    """
    GeminiCompletion used by the Gemini provider to execute completions.

    This class extends LiteLLmShim to handle Gemini-specific completion logic.
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
        Execute the completion request with LiteLLM's API.

        Ensures the model name is prefixed with the appropriate prefix to route to Google's API:
        - 'openai/' for the OpenAI-compatible endpoint (v1beta/openai)
        - 'gemini/' for the native Gemini API
        """
        model_in_request = request["model"]

        # Check if we're using the OpenAI-compatible endpoint
        is_openai_compatible = False
        if "_is_openai_compatible" in request and request["_is_openai_compatible"]:
            is_openai_compatible = True
        elif base_url and "v1beta/openai" in base_url:
            is_openai_compatible = True

        # Apply the appropriate prefix based on the endpoint
        if is_openai_compatible:
            # For OpenAI-compatible endpoint, use 'openai/' prefix
            if not model_in_request.startswith("openai/"):
                request["model"] = f"openai/{model_in_request}"
                logger.debug(
                    "Using OpenAI-compatible endpoint, prefixed model name with 'openai/': %s",
                    request["model"],
                )
        else:
            # For native Gemini API, use 'gemini/' prefix
            if not model_in_request.startswith("gemini/"):
                request["model"] = f"gemini/{model_in_request}"
                logger.debug(
                    "Using native Gemini API, prefixed model name with 'gemini/': %s",
                    request["model"],
                )

        # Set the API key and base URL
        request["api_key"] = api_key
        request["base_url"] = base_url

        # Execute the completion
        return await super().execute_completion(
            request=request,
            api_key=api_key,
            stream=stream,
            is_fim_request=is_fim_request,
            base_url=base_url,
        )
