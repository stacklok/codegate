from typing import Any, Dict, List

from litellm import AllMessageValues, ChatCompletionRequest, OpenAIMessageContent

from codegate.providers.normalizer.base import ModelInputNormalizer, ModelOutputNormalizer


class ChatMlInputNormalizer(ModelInputNormalizer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _str_from_message(message: OpenAIMessageContent) -> str:
        """
        LiteLLM has a weird Union wrapping their messages, so we need to extract the text from it.
        """
        if isinstance(message, str):
            return message
        text_parts = []
        try:
            for item in message:
                try:
                    if isinstance(item, dict):
                        if item.get("type") == "text" and "text" in item:
                            text_parts.append(item["text"])
                except (AttributeError, TypeError):
                    # Skip items that can't be processed as dicts
                    continue
        except TypeError:
            # Handle case where content is not actually iterable
            return ""

        return " ".join(text_parts)

    def split_chat_ml_request(self, request: str) -> List[AllMessageValues]:
        """
        Split a ChatML request into a list of ChatCompletionTextObjects.
        """
        messages: List[AllMessageValues] = []

        parts = request.split("<|im_start|>")
        for part in parts[1:]:
            # Skip if there's no im_end tag
            if "<|im_end|>" not in part:
                continue

            # Split by im_end to get the message content
            message_part = part.split("<|im_end|>")[0]

            # Split the first line which contains the role
            lines = message_part.split("\n", 1)

            if len(lines) != 2:
                continue

            messages.append({"role": lines[0].strip(), "content": lines[1].strip()})

        return messages

    def normalize(self, data: Dict) -> ChatCompletionRequest:
        """
        Normalize the input data to the format expected by ChatML.
        """
        # Make a copy of the data to avoid modifying the original
        normalized_data = data.copy()

        # ChatML requests have a single message separated by tags and newlines
        # if it's not the case, just return the input data and hope for the best
        input_chat_request = ChatCompletionRequest(**normalized_data)
        input_messages = input_chat_request.get("messages", [])
        if len(input_messages) != 1:
            return input_chat_request
        input_chat_request["messages"] = self.split_chat_ml_request(
            self._str_from_message(input_messages[0]["content"])
        )
        return input_chat_request

    def denormalize(self, data: ChatCompletionRequest) -> Dict:
        """
        Convert back to raw format for the API request
        """
        # we don't have to denormalize since we are using litellm later on.
        # For completeness we should if we are # talking to the LLM directly
        # but for now we don't need to
        return data


class VLLMInputNormalizer(ModelInputNormalizer):
    def __init__(self):
        self._chat_ml_normalizer = ChatMlInputNormalizer()
        super().__init__()

    @staticmethod
    def _has_chat_ml_format(data: Dict) -> bool:
        """
        Determine if the input data is in ChatML format.
        """
        input_chat_request = ChatCompletionRequest(**data)
        if len(input_chat_request.get("messages", [])) != 1:
            # ChatML requests have a single message
            return False
        content = input_chat_request["messages"][0]["content"]
        if isinstance(content, str) and "<|im_start|>" in content:
            return True

    def normalize(self, data: Dict) -> ChatCompletionRequest:
        """
        Normalize the input data to the format expected by LiteLLM.
        Ensures the model name has the hosted_vllm prefix and base_url has /v1.
        """
        # Make a copy of the data to avoid modifying the original
        normalized_data = data.copy()

        # Format the model name to include the provider
        if "model" in normalized_data:
            model_name = normalized_data["model"]
            if not model_name.startswith("hosted_vllm/"):
                normalized_data["model"] = f"hosted_vllm/{model_name}"

        # Ensure the base_url ends with /v1 if provided
        if "base_url" in normalized_data:
            base_url = normalized_data["base_url"].rstrip("/")
            if not base_url.endswith("/v1"):
                normalized_data["base_url"] = f"{base_url}/v1"

        ret_data = normalized_data
        if self._has_chat_ml_format(normalized_data):
            ret_data = self._chat_ml_normalizer.normalize(normalized_data)
        return ret_data

    def denormalize(self, data: ChatCompletionRequest) -> Dict:
        """
        Convert back to raw format for the API request
        """
        return data


class VLLMOutputNormalizer(ModelOutputNormalizer):
    def __init__(self):
        super().__init__()

    def normalize_streaming(
        self,
        model_reply: Any,
    ) -> Any:
        """
        No normalizing needed for streaming responses
        """
        return model_reply

    def normalize(self, model_reply: Any) -> Any:
        """
        No normalizing needed for responses
        """
        return model_reply

    def denormalize(self, normalized_reply: Any) -> Any:
        """
        No denormalizing needed for responses
        """
        return normalized_reply

    def denormalize_streaming(
        self,
        normalized_reply: Any,
    ) -> Any:
        """
        No denormalizing needed for streaming responses
        """
        return normalized_reply
