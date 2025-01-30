from unittest.mock import AsyncMock, MagicMock

import pytest
from litellm import ChatCompletionRequest
from ollama import ChatResponse, GenerateResponse, Message

from codegate.providers.ollama.completion_handler import OllamaShim


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.generate = AsyncMock(return_value=GenerateResponse(response="FIM response"))
    client.chat = AsyncMock(
        return_value=ChatResponse(message=Message(content="Chat response", role="assistant"))
    )
    return client


@pytest.fixture
def handler(mock_client):
    ollama_shim = OllamaShim("http://ollama:11434")
    ollama_shim.client = mock_client
    return ollama_shim


@pytest.fixture
def chat_request():
    return ChatCompletionRequest(
        model="test-model", messages=[{"role": "user", "content": "Hello"}], options={}
    )


@pytest.mark.asyncio
async def test_execute_completion_is_fim_request(handler, chat_request):
    chat_request["messages"][0]["content"] = "FIM prompt"
    await handler.execute_completion(chat_request, api_key=None, stream=False, is_fim_request=True)
    handler.client.generate.assert_called_once_with(
        model=chat_request["model"],
        prompt="FIM prompt",
        stream=False,
        options=chat_request["options"],
        suffix="",
    )


@pytest.mark.asyncio
async def test_execute_completion_not_is_fim_request(handler, chat_request):
    await handler.execute_completion(chat_request, api_key=None, stream=False, is_fim_request=False)
    handler.client.chat.assert_called_once_with(
        model=chat_request["model"],
        messages=chat_request["messages"],
        stream=False,
        options=chat_request["options"],
    )
