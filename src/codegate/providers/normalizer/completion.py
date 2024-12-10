from typing import Dict

from litellm.types.llms.openai import (
    ChatCompletionRequest,
)

from codegate.providers.normalizer import ModelInputNormalizer


class CompletionNormalizer(ModelInputNormalizer):
    def __init__(self):
        super().__init__()

    def normalize(self, data: Dict) -> ChatCompletionRequest:
        """
        Turn a completion request into a ChatCompletionRequest object.
        """
        # When doing FIM, we receive "prompt" instead of messages. Normalizing.
        if "prompt" in data:
            data["messages"] = [{"content": data.pop("prompt"), "role": "user"}]
            # We can add as many parameters as we like to data. ChatCompletionRequest is not strict.
            data["had_prompt_before"] = True
        try:
            return ChatCompletionRequest(**data)
        except Exception as e:
            raise ValueError(f"Invalid completion parameters: {str(e)}")

    def denormalize(self, data: ChatCompletionRequest) -> Dict:
        """
        Turn a ChatCompletionRequest object back into a completion request.
        """
        # If we receive "prompt" in FIM, we need convert it back.
        # Take care to not convert a system message into a prompt.
        # TODO: we should prepend the system message into the prompt.
        if data.get("had_prompt_before", False):
            for message in data["messages"]:
                if message["role"] == "user":
                    data["prompt"] = message["content"]
                    break
            del data["had_prompt_before"]
            del data["messages"]
        return data
