import json
import pytest
from codegate.llmclient.types import (
    Message,
    NormalizedRequest,
    ChatResponse,
    Delta,
    Choice,
    Response
)

def test_message():
    """Test Message dataclass."""
    msg = Message(content="Hello", role="user")
    assert msg.content == "Hello"
    assert msg.role == "user"

def test_normalized_request():
    """Test NormalizedRequest dataclass."""
    messages = [
        Message(content="Hello", role="user"),
        Message(content="Hi", role="assistant")
    ]
    request = NormalizedRequest(
        messages=messages,
        model="test-model",
        stream=True,
        options={"temperature": 0.7}
    )
    
    assert request.messages == messages
    assert request.model == "test-model"
    assert request.stream is True
    assert request.options == {"temperature": 0.7}

def test_chat_response():
    """Test ChatResponse model."""
    messages = [Message(content="Hello", role="user")]
    response = ChatResponse(
        id="test-id",
        messages=messages,
        created=1234567890,
        model="test-model",
        done=True
    )
    
    assert response.id == "test-id"
    assert response.messages == messages
    assert response.created == 1234567890
    assert response.model == "test-model"
    assert response.done is True

def test_delta():
    """Test Delta model."""
    delta = Delta(content="Hello", role="assistant")
    assert delta.content == "Hello"
    assert delta.role == "assistant"
    
    # Test optional fields
    delta = Delta()
    assert delta.content is None
    assert delta.role is None

def test_choice():
    """Test Choice model."""
    delta = Delta(content="Hello")
    choice = Choice(
        finish_reason="stop",
        index=0,
        delta=delta
    )
    
    assert choice.finish_reason == "stop"
    assert choice.index == 0
    assert choice.delta == delta
    assert choice.logprobs is None

def test_response():
    """Test Response model."""
    delta = Delta(content="Hello")
    choice = Choice(delta=delta)
    response = Response(
        id="test-id",
        choices=[choice],
        created=1234567890,
        model="test-model"
    )
    
    assert response.id == "test-id"
    assert len(response.choices) == 1
    assert response.created == 1234567890
    assert response.model == "test-model"
    assert response.object == "chat.completion.chunk"
    assert response.stream is False

def test_response_json():
    """Test Response JSON serialization."""
    delta = Delta(content="Hello")
    choice = Choice(delta=delta)
    response = Response(
        id="test-id",
        choices=[choice],
        created=1234567890,
        model="test-model"
    )
    
    json_str = response.json()
    data = json.loads(json_str)
    
    assert data["id"] == "test-id"
    assert len(data["choices"]) == 1
    assert data["created"] == 1234567890
    assert data["model"] == "test-model"
    assert data["object"] == "chat.completion.chunk"
    assert data["stream"] is False

def test_response_message():
    """Test Response message property."""
    delta = Delta(content="Hello")
    choice = Choice(delta=delta)
    response = Response(
        id="test-id",
        choices=[choice],
        created=1234567890,
        model="test-model"
    )
    
    assert response.message == choice
    
    # Test empty choices
    empty_response = Response(
        id="test-id",
        choices=[],
        created=1234567890,
        model="test-model"
    )
    assert empty_response.message is None 