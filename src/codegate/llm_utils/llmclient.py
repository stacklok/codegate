import json
from typing import Any, Dict, Optional

import litellm
import structlog
from ollama import Client as OllamaClient

from codegate.config import Config
from codegate.inference import LlamaCppInferenceEngine
from codegate.types.generators import acompletion

logger = structlog.get_logger("codegate")

litellm.drop_params = True


class LLMClient:
    """
    Base class for LLM interactions handling both local and cloud providers.

    This is a kludge before we refactor our providers a bit to be able to pass
    in all the parameters we need.
    """

    @staticmethod
    async def complete(
        content: str,
        system_prompt: str,
        provider: str,
        model: str = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Send a completion request to either local or cloud LLM.

        Args:
            content: The user message content
            system_prompt: The system prompt to use
            provider: "local" or "litellm"
            model: Model identifier
            api_key: API key for cloud providers
            base_url: Base URL for cloud providers
            **kwargs: Additional arguments for the completion request

        Returns:
            Parsed response from the LLM
        """
        if provider == "llamacpp":
            return await LLMClient._complete_local(content, system_prompt, model, **kwargs)
        return await LLMClient._complete_litellm(
            content,
            system_prompt,
            provider,
            model,
            api_key,
            base_url,
            extra_headers,
            **kwargs,
        )

    @staticmethod
    async def _create_request(
        content: str, system_prompt: str, model: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Private method to create a request dictionary for LLM completion.
        """
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "model": model,
            "stream": False,
            "response_format": {"type": "json_object"},
            "temperature": kwargs.get("temperature", 0),
        }

    @staticmethod
    async def _complete_local(
        content: str,
        system_prompt: str,
        model: str,
        **kwargs,
    ) -> Dict[str, Any]:
        # Use the private method to create the request
        request = await LLMClient._create_request(content, system_prompt, model, **kwargs)

        inference_engine = LlamaCppInferenceEngine()
        result = await inference_engine.chat(
            f"{Config.get_config().model_base_path}/{request['model']}.gguf",
            n_ctx=Config.get_config().chat_model_n_ctx,
            n_gpu_layers=Config.get_config().chat_model_n_gpu_layers,
            **request,
        )

        return json.loads(result["choices"][0]["message"]["content"])

    @staticmethod
    async def _complete_litellm(
        content: str,
        system_prompt: str,
        provider: str,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # Use the private method to create the request
        request = await LLMClient._create_request(content, system_prompt, model, **kwargs)

        # We should reuse the same logic in the provider
        # but let's do that later
        if provider == "vllm":
            if not base_url.endswith("/v1"):
                base_url = f"{base_url}/v1"
        else:
            if not model.startswith(f"{provider}/"):
                model = f"{provider}/{model}"

        try:
            if provider == "ollama":
                model = model.split("/")[-1]
                response = OllamaClient(host=base_url).chat(
                    model=model,
                    messages=request["messages"],
                    format="json",
                    options={"temperature": request["temperature"]},
                )
                content = response.message.content
            else:
                response = await acompletion(
                    model=model,
                    messages=request["messages"],
                    api_key=api_key,
                    temperature=request["temperature"],
                    base_url=base_url,
                    response_format=request["response_format"],
                    extra_headers=extra_headers,
                )
                content = response["choices"][0]["message"]["content"]

            # Clean up code blocks if present
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            return json.loads(content)

        except Exception as e:
            logger.error(f"LiteLLM completion failed {model} ({content}): {e}")
            raise e
