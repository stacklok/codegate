from typing import Any, Dict

from litellm import ChatCompletionRequest

from codegate.providers.normalizer.base import ModelInputNormalizer, ModelOutputNormalizer


class OpenAIInputNormalizer(ModelInputNormalizer):
    def __init__(self):
        super().__init__()

    def normalize(self, data: Dict) -> ChatCompletionRequest:
        """
        No normalizing needed, already OpenAI format
        """
        normalized_data = self._normalize_content_messages(data)
        if normalized_data.get("stream", False):
            normalized_data["stream_options"] = {"include_usage": True}
        return ChatCompletionRequest(**normalized_data)

    def denormalize(self, data: ChatCompletionRequest) -> Dict:
        """
        No denormalizing needed, already OpenAI format
        """
        return data


class OpenAIOutputNormalizer(ModelOutputNormalizer):
    def __init__(self):
        super().__init__()

    def normalize_streaming(
        self,
        model_reply: Any,
    ) -> Any:
        """
        No normalizing needed, already OpenAI format
        """
        return model_reply

    def normalize(self, model_reply: Any) -> Any:
        """
        No normalizing needed, already OpenAI format
        """
        return model_reply

    def denormalize(self, normalized_reply: Any) -> Any:
        """
        No denormalizing needed, already OpenAI format
        """
        return normalized_reply

    def denormalize_streaming(
        self,
        normalized_reply: Any,
    ) -> Any:
        """
        No denormalizing needed, already OpenAI format
        """
        return normalized_reply
