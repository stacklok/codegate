import pytest
from typing import AsyncIterator, List
from unittest.mock import AsyncMock, MagicMock

from codegate.llmclient.base import ChatResponse, CompletionResponse, LLMProvider, Message

class MockLLMProvider(LLMProvider):
    """Mock provider for testing the base LLM functionality."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chat_mock = AsyncMock()
        self.complete_mock = AsyncMock()
        self.close_mock = AsyncMock()
        
    async def chat(
        self,
        messages: List[Message],
        model: str = None,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ):
        return await self.chat_mock(messages, model, temperature, stream, **kwargs)
        
    async def complete(
        self,
        prompt: str,
        model: str = None,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ):
        return await self.complete_mock(prompt, model, temperature, stream, **kwargs)
        
    async def close(self):
        await self.close_mock()

@pytest.fixture
def mock_provider():
    return MockLLMProvider(api_key="test-key", base_url="http://test", default_model="test-model")

@pytest.mark.asyncio
async def test_provider_initialization():
    """Test that provider is initialized with correct parameters."""
    provider = MockLLMProvider(
        api_key="test-key",
        base_url="http://test",
        default_model="test-model"
    )
    
    assert provider.api_key == "test-key"
    assert provider.base_url == "http://test"
    assert provider.default_model == "test-model"

@pytest.mark.asyncio
async def test_chat_non_streaming(mock_provider):
    """Test non-streaming chat completion."""
    expected_response = ChatResponse(
        message=Message(role="assistant", content="Test response"),
        model="test-model",
        usage={"prompt_tokens": 10, "completion_tokens": 5}
    )
    mock_provider.chat_mock.return_value = expected_response
    
    messages = [
        Message(role="system", content="You are a test assistant"),
        Message(role="user", content="Hello")
    ]
    
    response = await mock_provider.chat(messages)
    assert response == expected_response
    mock_provider.chat_mock.assert_called_once_with(
        messages, None, 0.7, False
    )

@pytest.mark.asyncio
async def test_chat_streaming(mock_provider):
    """Test streaming chat completion."""
    async def mock_stream():
        responses = [
            ChatResponse(
                message=Message(role="assistant", content="Test"),
                model="test-model",
                usage={}
            ),
            ChatResponse(
                message=Message(role="assistant", content=" response"),
                model="test-model",
                usage={}
            )
        ]
        for response in responses:
            yield response
    
    mock_provider.chat_mock.return_value = mock_stream()
    
    messages = [Message(role="user", content="Hello")]
    responses = []
    
    async for response in await mock_provider.chat(messages, stream=True):
        responses.append(response)
        
    assert len(responses) == 2
    assert responses[0].message.content == "Test"
    assert responses[1].message.content == " response"
    mock_provider.chat_mock.assert_called_once_with(
        messages, None, 0.7, True
    )

@pytest.mark.asyncio
async def test_complete_non_streaming(mock_provider):
    """Test non-streaming text completion."""
    expected_response = CompletionResponse(
        text="Test response",
        model="test-model",
        usage={"prompt_tokens": 5, "completion_tokens": 2}
    )
    mock_provider.complete_mock.return_value = expected_response
    
    response = await mock_provider.complete("Hello")
    assert response == expected_response
    mock_provider.complete_mock.assert_called_once_with(
        "Hello", None, 0.7, False
    )

@pytest.mark.asyncio
async def test_complete_streaming(mock_provider):
    """Test streaming text completion."""
    async def mock_stream():
        responses = [
            CompletionResponse(text="Test", model="test-model", usage={}),
            CompletionResponse(text=" response", model="test-model", usage={})
        ]
        for response in responses:
            yield response
    
    mock_provider.complete_mock.return_value = mock_stream()
    
    responses = []
    async for response in await mock_provider.complete("Hello", stream=True):
        responses.append(response)
        
    assert len(responses) == 2
    assert responses[0].text == "Test"
    assert responses[1].text == " response"
    mock_provider.complete_mock.assert_called_once_with(
        "Hello", None, 0.7, True
    )

@pytest.mark.asyncio
async def test_close(mock_provider):
    """Test that close is called."""
    await mock_provider.close()
    mock_provider.close_mock.assert_called_once() 