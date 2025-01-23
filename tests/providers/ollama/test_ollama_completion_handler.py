from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ollama import ChatResponse, GenerateResponse, Message

from codegate.providers.ollama.completion_handler import OllamaShim
from codegate.types.common import ChatCompletionRequest


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
    ollama_shim = OllamaShim()
    return ollama_shim


@pytest.fixture
def chat_request():
    return ChatCompletionRequest(
        model="test-model", messages=[{"role": "user", "content": "Hello"}], options={}
    )


@patch("codegate.providers.ollama.completion_handler.AsyncClient.generate", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_execute_completion_is_fim_request(mock_client_generate, handler, chat_request):
    chat_request["messages"][0]["content"] = "FIM prompt"
    await handler.execute_completion(
        chat_request,
        base_url="http://ollama:11434",
        api_key=None,
        stream=False,
        is_fim_request=True,
    )
    mock_client_generate.assert_called_once_with(
        model=chat_request["model"],
        prompt="FIM prompt",
        stream=False,
        options=chat_request["options"],
        suffix="",
        raw=False,
    )


@patch("codegate.providers.ollama.completion_handler.AsyncClient.chat", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_execute_completion_not_is_fim_request(mock_client_chat, handler, chat_request):
    await handler.execute_completion(
        chat_request,
        base_url="http://ollama:11434",
        api_key=None,
        stream=False,
        is_fim_request=False,
    )
    mock_client_chat.assert_called_once_with(
        model=chat_request["model"],
        messages=chat_request["messages"],
        stream=False,
        options=chat_request["options"],
    )
