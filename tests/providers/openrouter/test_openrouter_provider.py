import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.requests import Request

from codegate.config import DEFAULT_PROVIDER_URLS
from codegate.pipeline.factory import PipelineFactory
from codegate.providers.openrouter.provider import OpenRouterProvider


@pytest.fixture
def mock_factory():
    return MagicMock(spec=PipelineFactory)


@pytest.fixture
def provider(mock_factory):
    return OpenRouterProvider(mock_factory)


def test_get_base_url(provider):
    """Test that _get_base_url returns the correct OpenRouter API URL"""
    assert provider._get_base_url() == DEFAULT_PROVIDER_URLS["openrouter"]


@pytest.mark.asyncio
@patch("codegate.providers.openai.OpenAIProvider.process_request")
async def test_model_prefix_added(mocked_parent_process_request):
    """Test that model name gets prefixed with openrouter/ when not already present"""
    mock_factory = MagicMock(spec=PipelineFactory)
    provider = OpenRouterProvider(mock_factory)

    # Mock request
    mock_request = MagicMock(spec=Request)
    mock_request.body = AsyncMock(return_value=json.dumps({"model": "gpt-4"}).encode())
    mock_request.url.path = "/openrouter/chat/completions"
    mock_request.state.detected_client = "test-client"

    # Get the route handler function
    route_handlers = [
        route for route in provider.router.routes if route.path == "/openrouter/chat/completions"
    ]
    create_completion = route_handlers[0].endpoint

    await create_completion(request=mock_request, authorization="Bearer test-token")

    # Verify process_request was called with prefixed model
    call_args = mocked_parent_process_request.call_args[0]
    assert call_args[0]["model"] == "openrouter/gpt-4"


@pytest.mark.asyncio
async def test_model_prefix_preserved():
    """Test that model name is not modified when openrouter/ prefix is already present"""
    mock_factory = MagicMock(spec=PipelineFactory)
    provider = OpenRouterProvider(mock_factory)
    provider.process_request = AsyncMock()

    # Mock request
    mock_request = MagicMock(spec=Request)
    mock_request.body = AsyncMock(return_value=json.dumps({"model": "openrouter/gpt-4"}).encode())
    mock_request.url.path = "/openrouter/chat/completions"
    mock_request.state.detected_client = "test-client"

    # Get the route handler function
    route_handlers = [
        route for route in provider.router.routes if route.path == "/openrouter/chat/completions"
    ]
    create_completion = route_handlers[0].endpoint

    await create_completion(request=mock_request, authorization="Bearer test-token")

    # Verify process_request was called with unchanged model name
    call_args = provider.process_request.call_args[0]
    assert call_args[0]["model"] == "openrouter/gpt-4"


@pytest.mark.asyncio
async def test_invalid_auth_header():
    """Test that invalid authorization header format raises HTTPException"""
    mock_factory = MagicMock(spec=PipelineFactory)
    provider = OpenRouterProvider(mock_factory)

    mock_request = MagicMock(spec=Request)

    # Get the route handler function
    route_handlers = [
        route for route in provider.router.routes if route.path == "/openrouter/chat/completions"
    ]
    create_completion = route_handlers[0].endpoint

    with pytest.raises(HTTPException) as exc_info:
        await create_completion(request=mock_request, authorization="InvalidToken")

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid authorization header"
