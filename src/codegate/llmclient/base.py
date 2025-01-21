from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, List, Optional, Union
from dataclasses import dataclass

@dataclass
class Message:
    """Represents a chat message."""
    role: str
    content: str

@dataclass
class CompletionResponse:
    """Represents a completion response from an LLM."""
    text: str
    model: str
    usage: Dict[str, int]

@dataclass
class ChatResponse:
    """Represents a chat response from an LLM."""
    message: Message
    model: str
    usage: Dict[str, int]

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None
    ):
        """Initialize the LLM provider.

        Args:
            api_key: API key for authentication
            base_url: Optional custom base URL for the API
            default_model: Optional default model to use
        """
        self.api_key = api_key
        self.base_url = base_url
        self.default_model = default_model

    @abstractmethod
    async def chat(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> Union[ChatResponse, AsyncIterator[ChatResponse]]:
        """Send a chat request to the LLM.

        Args:
            messages: List of messages in the conversation
            model: Optional model override
            temperature: Sampling temperature
            stream: Whether to stream the response
            **kwargs: Additional provider-specific parameters

        Returns:
            ChatResponse or AsyncIterator[ChatResponse] if streaming
        """
        pass

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> Union[CompletionResponse, AsyncIterator[CompletionResponse]]:
        """Send a completion request to the LLM.

        Args:
            prompt: The text prompt
            model: Optional model override
            temperature: Sampling temperature
            stream: Whether to stream the response
            **kwargs: Additional provider-specific parameters

        Returns:
            CompletionResponse or AsyncIterator[CompletionResponse] if streaming
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close any open connections."""
        pass