import httpx
from typing import AsyncIterator, Dict, List, Optional, Union
import json

from codegate.llmclient.base import LLMProvider, Message, ChatResponse, CompletionResponse

class AnthropicProvider(LLMProvider):
    """Anthropic API provider implementation."""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = "https://api.anthropic.com/v1",
        default_model: Optional[str] = "claude-3-opus-20240229"
    ):
        """Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key
            base_url: Optional API base URL override
            default_model: Default model to use
        """
        super().__init__(api_key, base_url, default_model)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }
        )

    async def chat(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> Union[ChatResponse, AsyncIterator[ChatResponse]]:
        """Send a chat request to Anthropic."""
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

        async with self._client as client:
            response = await client.post("/messages", json=payload)
            response.raise_for_status()
            data = response.json()

            return ChatResponse(
                message=Message(
                    role="assistant",
                    content=data["content"][0]["text"]
                ),
                model=data["model"],
                usage={
                    "prompt_tokens": data.get("usage", {}).get("input_tokens", 0),
                    "completion_tokens": data.get("usage", {}).get("output_tokens", 0),
                    "total_tokens": data.get("usage", {}).get("total_tokens", 0)
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
        """Send a completion request to Anthropic."""
        # Convert completion request to chat format since Anthropic uses unified endpoint
        messages = [Message(role="user", content=prompt)]
        chat_response = await self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            stream=stream,
            **kwargs
        )

        if stream:
            async def convert_stream():
                async for chunk in chat_response:
                    yield CompletionResponse(
                        text=chunk.message.content,
                        model=chunk.model,
                        usage=chunk.usage
                    )
            return convert_stream()
        else:
            return CompletionResponse(
                text=chat_response.message.content,
                model=chat_response.model,
                usage=chat_response.usage
            )

    async def _stream_chat(self, payload: Dict) -> AsyncIterator[ChatResponse]:
        """Handle streaming chat responses."""
        async with self._client as client:
            async with client.stream("POST", "/messages", json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        if line.strip() == "data: [DONE]":
                            break

                        data = json.loads(line[6:])
                        if "delta" not in data:
                            continue

                        delta = data["delta"]
                        if "text" not in delta:
                            continue

                        yield ChatResponse(
                            message=Message(
                                role="assistant",
                                content=delta["text"]
                            ),
                            model=data["model"],
                            usage={}  # Usage stats only available at end of stream
                        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()