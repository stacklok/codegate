import time
from typing import AsyncIterator, Dict, List, Optional, Union

import litellm
from litellm import acompletion
from litellm.types.utils import Delta, StreamingChoices

from codegate.llmclient.base import (
    ChatResponse,
    CompletionResponse,
    LLMProvider,
    Message,
)

litellm.drop_params = True

class LiteLLMBridgeProvider(LLMProvider):
    """Bridge provider that implements the new LLMProvider interface using LiteLLM."""
    
    async def chat(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> Union[ChatResponse, AsyncIterator[ChatResponse]]:
        """Send a chat request using LiteLLM."""
        
        # Convert messages to LiteLLM format
        litellm_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        # Use default model if none specified
        model_name = model or self.default_model
        if not model_name:
            raise ValueError("No model specified")
            
        # Prepare request
        request = {
            "model": model_name,
            "messages": litellm_messages,
            "temperature": temperature,
            "stream": stream,
            "api_key": self.api_key,
            **kwargs
        }
        
        if self.base_url:
            request["base_url"] = self.base_url
            
        async def process_stream() -> AsyncIterator[ChatResponse]:
            async for chunk in await acompletion(**request):
                if not chunk.choices:
                    continue
                    
                choice = chunk.choices[0]
                if not choice.delta or not choice.delta.content:
                    continue
                    
                yield ChatResponse(
                    message=Message(
                        role="assistant",
                        content=choice.delta.content
                    ),
                    model=chunk.model,
                    usage={}  # Usage stats only available in final response
                )
                
        if stream:
            return process_stream()
        else:
            response = await acompletion(**request)
            if not response.choices:
                raise ValueError("No choices in response")
                
            choice = response.choices[0]
            return ChatResponse(
                message=Message(
                    role=choice.message.role,
                    content=choice.message.content
                ),
                model=response.model,
                usage=response.usage
            )
            
    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> Union[CompletionResponse, AsyncIterator[CompletionResponse]]:
        """Send a completion request using LiteLLM."""
        # Convert to chat format since that's what most models use now
        messages = [{"role": "user", "content": prompt}]
        
        # Use default model if none specified  
        model_name = model or self.default_model
        if not model_name:
            raise ValueError("No model specified")
            
        # Prepare request
        request = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            "api_key": self.api_key,
            **kwargs
        }
        
        if self.base_url:
            request["base_url"] = self.base_url
            
        async def process_stream() -> AsyncIterator[CompletionResponse]:
            async for chunk in await acompletion(**request):
                if not chunk.choices:
                    continue
                    
                choice = chunk.choices[0]
                if not choice.delta or not choice.delta.content:
                    continue
                    
                yield CompletionResponse(
                    text=choice.delta.content,
                    model=chunk.model,
                    usage={}  # Usage stats only available in final response
                )
                
        if stream:
            return process_stream()
        else:
            response = await acompletion(**request)
            if not response.choices:
                raise ValueError("No choices in response")
                
            choice = response.choices[0]
            return CompletionResponse(
                text=choice.message.content,
                model=response.model,
                usage=response.usage
            )
            
    async def close(self) -> None:
        """Nothing to close for LiteLLM."""
        pass 