from typing import Any, AsyncIterable, AsyncIterator, Dict, Iterable, Iterator, Union

from litellm import ChatCompletionRequest, ModelResponse

from codegate.providers.normalizer import ModelInputNormalizer, ModelOutputNormalizer


class LLamaCppInputNormalizer(ModelInputNormalizer):
    def normalize(self, data: Dict) -> ChatCompletionRequest:
        """
        Normalize the input data
        """
        try:
            return ChatCompletionRequest(**data)
        except Exception as e:
            raise ValueError(f"Invalid completion parameters: {str(e)}")

    def denormalize(self, data: ChatCompletionRequest) -> Dict:
        """
        Denormalize the input data
        """
        return data


class LLamaCppOutputNormalizer(ModelOutputNormalizer):
    def normalize_streaming(
        self,
        model_reply: Union[AsyncIterable[Any], Iterable[Any]],
    ) -> Union[AsyncIterator[ModelResponse], Iterator[ModelResponse]]:
        """
        Normalize the output stream. This is a pass-through for liteLLM output normalizer
        as the liteLLM output is already in the normalized format.
        """
        return model_reply

    def normalize(self, model_reply: Any) -> ModelResponse:
        """
        Normalize the output data. This is a pass-through for liteLLM output normalizer
        as the liteLLM output is already in the normalized format.
        """
        return model_reply

    def denormalize(self, normalized_reply: ModelResponse) -> Any:
        """
        Denormalize the output data from the completion function to the format
        expected by the client
        """
        return normalized_reply

    def denormalize_streaming(
        self,
        normalized_reply: Union[AsyncIterable[ModelResponse], Iterable[ModelResponse]],
    ) -> Union[AsyncIterator[Any], Iterator[Any]]:
        """
        Denormalize the output stream from the completion function to the format
        expected by the client
        """
        return normalized_reply
