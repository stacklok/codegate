import httpx
from typing import AsyncIterator, Dict, List, Optional, Union
import json

from codegate.llmclient.base import LLMProvider, Message, ChatResponse, CompletionResponse

class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation."""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = "https://api.openai.com/v1",
        default_model: Optional[str] = "gpt-3.5-turbo"
    ):
        """Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key
            base_url: Optional API base URL override
            default_model: Default model to use
        """
        super().__init__(api_key, base_url, default_model)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=60.0  # 60 second timeout for requests
        )

    async def chat(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> Union[ChatResponse, AsyncIterator[ChatResponse]]:
        """Send a chat request to OpenAI."""
        model = model or self.default_model

        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "stream": stream,
            **kwargs
        }

        if stream:
            return self._stream_chat(payload)

        response = await self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        return ChatResponse(
                message=Message(
                    role=data["choices"][0]["message"]["role"],
                    content=data["choices"][0]["message"]["content"]
                ),
                model=data["model"],
                usage=data["usage"]
            )

    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> Union[CompletionResponse, AsyncIterator[CompletionResponse]]:
        """Send a completion request to OpenAI."""
        model = model or self.default_model

        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "stream": stream,
            **kwargs
        }

        if stream:
            return self._stream_completion(payload)

        response = await self._client.post("/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        return CompletionResponse(
                text=data["choices"][0]["text"],
                model=data["model"],
                usage=data["usage"]
            )

    async def _stream_chat(self, payload: Dict) -> AsyncIterator[ChatResponse]:
        """Handle streaming chat responses."""
        async with self._client.stream("POST", "/chat/completions", json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        if line.strip() == "data: [DONE]":
                            break

                        data = json.loads(line[6:])
                        if not data["choices"]:
                            continue

                        delta = data["choices"][0]["delta"]
                        if "content" not in delta:
                            continue

                        yield ChatResponse(
                            message=Message(
                                role=delta.get("role", "assistant"),
                                content=delta["content"]
                            ),
                            model=data["model"],
                            usage={}  # Usage stats only available at end of stream
                        )

    async def _stream_completion(self, payload: Dict) -> AsyncIterator[CompletionResponse]:
        """Handle streaming completion responses."""
        async with self._client.stream("POST", "/completions", json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        if line.strip() == "data: [DONE]":
                            break

                        data = json.loads(line[6:])
                        if not data["choices"]:
                            continue

                        yield CompletionResponse(
                            text=data["choices"][0]["text"],
                            model=data["model"],
                            usage={}  # Usage stats only available at end of stream
                        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
