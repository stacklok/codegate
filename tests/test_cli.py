"""Tests for the server module."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from httpx import AsyncClient

from codegate import __version__
from codegate.pipeline.factory import PipelineFactory
from codegate.pipeline.secrets.manager import SecretsManager
from codegate.providers.registry import ProviderRegistry
from codegate.server import init_app


@pytest.fixture
def mock_secrets_manager():
    """Create a mock secrets manager."""
    return MagicMock(spec=SecretsManager)


@pytest.fixture
def mock_provider_registry():
    """Create a mock provider registry."""
    return MagicMock(spec=ProviderRegistry)


@pytest.fixture
def mock_pipeline_factory():
    """Create a mock pipeline factory."""
    mock_factory = MagicMock(spec=PipelineFactory)
    # Mock the methods that are called on the pipeline factory
    mock_factory.create_input_pipeline.return_value = MagicMock()
    mock_factory.create_fim_pipeline.return_value = MagicMock()
    mock_factory.create_output_pipeline.return_value = MagicMock()
    mock_factory.create_fim_output_pipeline.return_value = MagicMock()
    return mock_factory


@pytest.fixture
def test_client(mock_pipeline_factory) -> TestClient:
    """Create a test client for the FastAPI application."""
    app = init_app(mock_pipeline_factory)
    return TestClient(app)


def test_app_initialization(mock_pipeline_factory) -> None:
    """Test that the FastAPI application initializes correctly."""
    app = init_app(mock_pipeline_factory)
    assert app is not None
    assert app.title == "CodeGate"
    assert app.version == __version__


def test_cors_middleware(mock_pipeline_factory) -> None:
    """Test that CORS middleware is properly configured."""
    app = init_app(mock_pipeline_factory)
    cors_middleware = None
    for middleware in app.user_middleware:
        if isinstance(middleware.cls, type) and issubclass(middleware.cls, CORSMiddleware):
            cors_middleware = middleware
            break
    assert cors_middleware is not None
    assert cors_middleware.kwargs.get("allow_origins") == ["*"]
    assert cors_middleware.kwargs.get("allow_credentials") is True
    assert cors_middleware.kwargs.get("allow_methods") == ["*"]
    assert cors_middleware.kwargs.get("allow_headers") == ["*"]


def test_health_check(test_client: TestClient) -> None:
    """Test the health check endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@patch("codegate.pipeline.secrets.manager.SecretsManager")
@patch("codegate.server.ProviderRegistry")
def test_provider_registration(mock_registry, mock_secrets_mgr, mock_pipeline_factory) -> None:
    """Test that all providers are registered correctly."""
    init_app(mock_pipeline_factory)

    # Verify ProviderRegistry was initialized with the app
    mock_registry.assert_called_once()

    # Verify all providers were registered
    registry_instance = mock_registry.return_value
    assert (
        registry_instance.add_provider.call_count == 5
    )  # openai, anthropic, llamacpp, vllm, ollama

    # Verify specific providers were registered
    provider_names = [call.args[0] for call in registry_instance.add_provider.call_args_list]
    assert "openai" in provider_names
    assert "anthropic" in provider_names
    assert "llamacpp" in provider_names
    assert "vllm" in provider_names
    assert "ollama" in provider_names


@patch("codegate.server.CodegateSignatures")
def test_signatures_initialization(mock_signatures, mock_pipeline_factory) -> None:
    """Test that signatures are initialized correctly."""
    init_app(mock_pipeline_factory)
    mock_signatures.initialize.assert_called_once_with("signatures.yaml")


def test_pipeline_initialization(mock_pipeline_factory) -> None:
    """Test that pipelines are initialized correctly."""
    app = init_app(mock_pipeline_factory)

    # Access the provider registry to check pipeline configuration
    registry = next((route for route in app.routes if hasattr(route, "registry")), None)

    if registry:
        for provider in registry.registry.values():
            # Verify each provider has the required pipelines
            assert hasattr(provider, "pipeline_processor")
            assert hasattr(provider, "fim_pipeline_processor")
            assert hasattr(provider, "output_pipeline_processor")


def test_dashboard_routes(mock_pipeline_factory) -> None:
    """Test that dashboard routes are included."""
    app = init_app(mock_pipeline_factory)
    routes = [route.path for route in app.routes]

    # Verify dashboard endpoints are included
    dashboard_routes = [route for route in routes if route.startswith("/dashboard")]
    assert len(dashboard_routes) > 0


def test_system_routes(mock_pipeline_factory) -> None:
    """Test that system routes are included."""
    app = init_app(mock_pipeline_factory)
    routes = [route.path for route in app.routes]

    # Verify system endpoints are included
    assert "/health" in routes


@pytest.mark.asyncio
async def test_async_health_check(mock_pipeline_factory) -> None:
    """Test the health check endpoint with async client."""
    app = init_app(mock_pipeline_factory)
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


def test_error_handling(test_client: TestClient) -> None:
    """Test error handling for non-existent endpoints."""
    response = test_client.get("/nonexistent")
    assert response.status_code == 404

    # Test method not allowed
    response = test_client.post("/health")  # Health endpoint only allows GET
    assert response.status_code == 405
