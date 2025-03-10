from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ollama import ChatResponse, GenerateResponse, Message

from codegate.providers.ollama.completion_handler import OllamaShim
from codegate.types import ollama, openai


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


@patch("codegate.providers.ollama.completion_handler.completions_streaming", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_execute_completion_is_openai_fim_request(mock_streaming, handler):
    openai_request = openai.ChatCompletionRequest(
        model="model",
        messages=[
            openai.UserMessage(
                role="user",
                content="FIM prompt",
            ),
        ],
    )
    await handler.execute_completion(
        openai_request,
        base_url="http://ollama:11434",
        api_key="key",
        stream=False,
        is_fim_request=True,
    )
    mock_streaming.assert_called_once_with(
        openai_request,
        "key",
        "http://ollama:11434",
    )


@patch("codegate.providers.ollama.completion_handler.generate_streaming", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_execute_completion_is_ollama_fim_request(mock_streaming, handler):
    ollama_request = ollama.GenerateRequest(
        model="model",
        prompt="FIM prompt",
    )
    await handler.execute_completion(
        ollama_request,
        base_url="http://ollama:11434",
        api_key="key",
        stream=False,
        is_fim_request=True,
    )
    mock_streaming.assert_called_once_with(
        ollama_request,
        "key",
        "http://ollama:11434",
    )


@patch("codegate.providers.ollama.completion_handler.chat_streaming", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_execute_completion_not_is_ollama_fim_request(mock_streaming, handler):
    ollama_request = ollama.ChatRequest(
        model="model",
        messages=[
            ollama.UserMessage(
                role="user",
                content="Chat prompt",
            ),
        ],
    )
    await handler.execute_completion(
        ollama_request,
        base_url="http://ollama:11434",
        api_key="key",
        stream=False,
        is_fim_request=False,
    )
    mock_streaming.assert_called_once_with(
        ollama_request,
        "key",
        "http://ollama:11434",
    )
