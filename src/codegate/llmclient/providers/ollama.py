import httpx
from typing import AsyncIterator, Dict, List, Optional, Union
import json

from codegate.llmclient.base import LLMProvider, Message, ChatResponse, CompletionResponse

class OllamaProvider(LLMProvider):
    """Ollama API provider implementation."""

    def __init__(
        self,
        api_key: str = "",  # Ollama doesn't use API keys by default
        base_url: Optional[str] = "http://localhost:11434",
        default_model: Optional[str] = "llama2"
    ):
        """Initialize the Ollama provider.

        Args:
            api_key: Not used by default in Ollama
            base_url: Optional API base URL override
            default_model: Default model to use
        """
        super().__init__(api_key, base_url, default_model)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Content-Type": "application/json"}
        )

    async def chat(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> Union[ChatResponse, AsyncIterator[ChatResponse]]:
        """Send a chat request to Ollama."""
        model = model or self.default_model

        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
            "options": {
                "temperature": temperature,
                **kwargs
            }
        }

        if stream:
            return self._stream_chat(payload)

        async with self._client as client:
            response = await client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

            return ChatResponse(
                message=Message(
                    role="assistant",
                    content=data["message"]["content"]
                ),
                model=model,
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                }
            )

    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> Union[CompletionResponse, AsyncIterator[CompletionResponse]]:
        """Send a completion request to Ollama."""
        model = model or self.default_model

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                **kwargs
            }
        }

        if stream:
            return self._stream_completion(payload)

        async with self._client as client:
            response = await client.post("/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()

            return CompletionResponse(
                text=data["response"],
                model=model,
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                }
            )

    async def _stream_chat(self, payload: Dict) -> AsyncIterator[ChatResponse]:
        """Handle streaming chat responses."""
        async with self._client as client:
            async with client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    data = json.loads(line)
                    if "done" in data and data["done"]:
                        break

                    yield ChatResponse(
                        message=Message(
                            role="assistant",
                            content=data["message"]["content"]
                        ),
                        model=payload["model"],
                        usage={
                            "prompt_tokens": data.get("prompt_eval_count", 0),
                            "completion_tokens": data.get("eval_count", 0),
                            "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                        }
                    )

    async def _stream_completion(self, payload: Dict) -> AsyncIterator[CompletionResponse]:
        """Handle streaming completion responses."""
        async with self._client as client:
            async with client.stream("POST", "/api/generate", json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    data = json.loads(line)
                    if "done" in data and data["done"]:
                        break

                    yield CompletionResponse(
                        text=data["response"],
                        model=payload["model"],
                        usage={
                            "prompt_tokens": data.get("prompt_eval_count", 0),
                            "completion_tokens": data.get("eval_count", 0),
                            "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                        }
                    )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()