import json
from typing import Any, Dict, Optional

import structlog

from codegate.config import Config
from codegate.inference import LlamaCppInferenceEngine
from codegate.llmclient.base import Message, LLMProvider
from codegate.providers.litellmshim.bridge import LiteLLMBridgeProvider

logger = structlog.get_logger("codegate")

class LLMClient:
    """Base class for LLM interactions handling both local and cloud providers."""
    
    @staticmethod
    def _create_provider(
        provider: str,
        model: str = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Optional[LLMProvider]:
        if provider == "llamacpp":
            return None  # Handled separately for now
        return LiteLLMBridgeProvider(
            api_key=api_key or "",
            base_url=base_url,
            default_model=model
        )

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
        if provider == "llamacpp":
            return await LLMClient._complete_local(content, system_prompt, model, **kwargs)
            
        llm_provider = LLMClient._create_provider(provider, model, api_key, base_url)
        
        try:
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=content)
            ]
            
            response = await llm_provider.chat(
                messages=messages,
                temperature=kwargs.get("temperature", 0),
                stream=False,
                extra_headers=extra_headers,
                **kwargs
            )
            
            return json.loads(response.message.content)
            
        except Exception as e:
            logger.error(f"LLM completion failed {model} ({content}): {e}")
            raise e
        finally:
            await llm_provider.close()

    @staticmethod
    async def _complete_local(
        content: str,
        system_prompt: str,
        model: str,
        **kwargs,
    ) -> Dict[str, Any]:
        request = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "model": model,
            "stream": False,
            "response_format": {"type": "json_object"},
            "temperature": kwargs.get("temperature", 0),
        }

        inference_engine = LlamaCppInferenceEngine()
        result = await inference_engine.chat(
            f"{Config.get_config().model_base_path}/{request['model']}.gguf",
            n_ctx=Config.get_config().chat_model_n_ctx,
            n_gpu_layers=Config.get_config().chat_model_n_gpu_layers,
            **request,
        )

        return json.loads(result["choices"][0]["message"]["content"])
