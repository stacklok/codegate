from typing import Any, Dict, Optional

import structlog
from litellm import ChatCompletionRequest

from codegate.providers.litellmshim import sse_stream_generator
from codegate.providers.litellmshim.adapter import (
    BaseAdapter,
    LiteLLMAdapterInputNormalizer,
    LiteLLMAdapterOutputNormalizer,
)

logger = structlog.get_logger("codegate")


class GeminiAdapter(BaseAdapter):
    """
    Adapter for Gemini API to translate between Gemini's format and OpenAI's format.
    """

    def __init__(self) -> None:
        super().__init__(sse_stream_generator)

    def translate_completion_input_params(self, kwargs) -> Optional[ChatCompletionRequest]:
        """
        Translate Gemini API parameters to OpenAI format.

        Gemini API uses a similar format to OpenAI, but with some differences:
        - 'contents' instead of 'messages'
        - Different role names
        - Different parameter names for temperature, etc.
        """
        # Make a copy to avoid modifying the original
        translated_params = dict(kwargs)

        # Handle Gemini-specific parameters
        if "contents" in translated_params:
            # Convert Gemini 'contents' to OpenAI 'messages'
            contents = translated_params.pop("contents")
            messages = []

            for content in contents:
                role = content.get("role", "user")
                # Map Gemini roles to OpenAI roles
                if role == "model":
                    role = "assistant"

                message = {
                    "role": role,
                    "content": content.get("parts", [{"text": ""}])[0].get("text", ""),
                }
                messages.append(message)

            translated_params["messages"] = messages

        # Map other parameters
        if "temperature" in translated_params:
            # Temperature is the same in both APIs
            pass

        if "topP" in translated_params:
            translated_params["top_p"] = translated_params.pop("topP")

        if "topK" in translated_params:
            translated_params["top_k"] = translated_params.pop("topK")

        if "maxOutputTokens" in translated_params:
            translated_params["max_tokens"] = translated_params.pop("maxOutputTokens")

        # Check if we're using the OpenAI-compatible endpoint
        is_openai_compatible = False
        if (
            "_is_openai_compatible" in translated_params
            and translated_params["_is_openai_compatible"]
        ):
            is_openai_compatible = True
            # Remove the custom field to avoid sending it to the API
            translated_params.pop("_is_openai_compatible")
        elif (
            "base_url" in translated_params
            and translated_params["base_url"]
            and "v1beta/openai" in translated_params["base_url"]
        ):
            is_openai_compatible = True

        # Apply the appropriate prefix based on the endpoint
        if "model" in translated_params:
            model_in_request = translated_params["model"]
            if is_openai_compatible:
                # For OpenAI-compatible endpoint, use 'openai/' prefix
                if not model_in_request.startswith("openai/"):
                    translated_params["model"] = f"openai/{model_in_request}"
                    logger.debug(
                        "Using OpenAI-compatible endpoint, prefixed model name with 'openai/': %s",
                        translated_params["model"],
                    )
            else:
                # For native Gemini API, use 'gemini/' prefix
                if not model_in_request.startswith("gemini/"):
                    translated_params["model"] = f"gemini/{model_in_request}"
                    logger.debug(
                        "Using native Gemini API, prefixed model name with 'gemini/': %s",
                        translated_params["model"],
                    )

        return ChatCompletionRequest(**translated_params)

    def translate_completion_output_params(self, response: Any) -> Dict:
        """
        Translate OpenAI format response to Gemini format.
        """
        # For non-streaming responses, we can just return the response as is
        # LiteLLM should handle the conversion
        return response

    def translate_completion_output_params_streaming(self, completion_stream: Any) -> Any:
        """
        Translate streaming response from OpenAI format to Gemini format.
        """
        # For streaming, we can just return the stream as is
        # The stream generator will handle the conversion
        return completion_stream


class GeminiInputNormalizer(LiteLLMAdapterInputNormalizer):
    """
    Normalizer for Gemini API input.
    """

    def __init__(self):
        self.adapter = GeminiAdapter()
        super().__init__(self.adapter)


class GeminiOutputNormalizer(LiteLLMAdapterOutputNormalizer):
    """
    Normalizer for Gemini API output.
    """

    def __init__(self):
        super().__init__(GeminiAdapter())
