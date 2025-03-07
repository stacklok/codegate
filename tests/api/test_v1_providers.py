from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4 as uuid

import httpx
import pytest
import structlog
from httpx import AsyncClient

from codegate.db import connection
from codegate.pipeline.factory import PipelineFactory
from codegate.providers.crud.crud import ProviderCrud
from codegate.server import init_app
from codegate.workspaces.crud import WorkspaceCrud

logger = structlog.get_logger("codegate")

# TODO: Abstract the mock DB setup


@pytest.fixture
def db_path():
    """Creates a temporary database file path."""
    current_test_dir = Path(__file__).parent
    db_filepath = current_test_dir / f"codegate_test_{uuid()}.db"
    db_fullpath = db_filepath.absolute()
    connection.init_db_sync(str(db_fullpath))
    yield db_fullpath
    if db_fullpath.is_file():
        db_fullpath.unlink()


@pytest.fixture()
def db_recorder(db_path) -> connection.DbRecorder:
    """Creates a DbRecorder instance with test database."""
    return connection.DbRecorder(sqlite_path=db_path, _no_singleton=True)


@pytest.fixture()
def db_reader(db_path) -> connection.DbReader:
    """Creates a DbReader instance with test database."""
    return connection.DbReader(sqlite_path=db_path, _no_singleton=True)


@pytest.fixture()
def mock_workspace_crud(db_recorder, db_reader) -> WorkspaceCrud:
    """Creates a WorkspaceCrud instance with test database."""
    ws_crud = WorkspaceCrud()
    ws_crud._db_reader = db_reader
    ws_crud._db_recorder = db_recorder
    return ws_crud


@pytest.fixture()
def mock_provider_crud(db_recorder, db_reader, mock_workspace_crud) -> ProviderCrud:
    """Creates a ProviderCrud instance with test database."""
    p_crud = ProviderCrud()
    p_crud._db_reader = db_reader
    p_crud._db_writer = db_recorder
    p_crud._ws_crud = mock_workspace_crud
    return p_crud


@pytest.fixture
def mock_pipeline_factory():
    """Create a mock pipeline factory."""
    mock_factory = MagicMock(spec=PipelineFactory)
    mock_factory.create_input_pipeline.return_value = MagicMock()
    mock_factory.create_fim_pipeline.return_value = MagicMock()
    mock_factory.create_output_pipeline.return_value = MagicMock()
    mock_factory.create_fim_output_pipeline.return_value = MagicMock()
    return mock_factory


@pytest.mark.asyncio
async def test_providers_crud(
    mock_pipeline_factory, mock_workspace_crud, mock_provider_crud
) -> None:
    with (
        patch("codegate.api.v1.wscrud", mock_workspace_crud),
        patch("codegate.api.v1.pcrud", mock_provider_crud),
        patch(
            "codegate.providers.openai.provider.OpenAIProvider.models",
            return_value=["gpt-4", "gpt-3.5-turbo"],
        ),
        patch(
            "codegate.providers.openrouter.provider.OpenRouterProvider.models",
            return_value=["anthropic/claude-2", "deepseek/deepseek-r1"],
        ),
    ):
        """Test creating multiple providers and listing them."""
        app = init_app(mock_pipeline_factory)

        async with AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as ac:
            # Create first provider (OpenAI)
            provider_payload_1 = {
                "name": "openai-provider",
                "description": "OpenAI provider description",
                "auth_type": "none",
                "provider_type": "openai",
                "endpoint": "https://api.openai.com",
                "api_key": "sk-proj-foo-bar-123-xyz",
            }

            response = await ac.post("/api/v1/provider-endpoints", json=provider_payload_1)
            assert response.status_code == 201
            provider1_response = response.json()
            assert provider1_response["name"] == provider_payload_1["name"]
            assert provider1_response["description"] == provider_payload_1["description"]
            assert provider1_response["auth_type"] == provider_payload_1["auth_type"]
            assert provider1_response["provider_type"] == provider_payload_1["provider_type"]
            assert provider1_response["endpoint"] == provider_payload_1["endpoint"]
            assert isinstance(provider1_response.get("id", ""), str) and provider1_response["id"]

            # Create second provider (OpenRouter)
            provider_payload_2 = {
                "name": "openrouter-provider",
                "description": "OpenRouter provider description",
                "auth_type": "none",
                "provider_type": "openrouter",
                "endpoint": "https://openrouter.ai/api",
                "api_key": "sk-or-foo-bar-456-xyz",
            }

            response = await ac.post("/api/v1/provider-endpoints", json=provider_payload_2)
            assert response.status_code == 201
            provider2_response = response.json()
            assert provider2_response["name"] == provider_payload_2["name"]
            assert provider2_response["description"] == provider_payload_2["description"]
            assert provider2_response["auth_type"] == provider_payload_2["auth_type"]
            assert provider2_response["provider_type"] == provider_payload_2["provider_type"]
            assert provider2_response["endpoint"] == provider_payload_2["endpoint"]
            assert isinstance(provider2_response.get("id", ""), str) and provider2_response["id"]

            # List all providers
            response = await ac.get("/api/v1/provider-endpoints")
            assert response.status_code == 200
            providers = response.json()

            # Verify both providers exist in the list
            assert isinstance(providers, list)
            assert len(providers) == 2

            # Verify fields for first provider
            provider1 = next(p for p in providers if p["name"] == "openai-provider")
            assert provider1["description"] == provider_payload_1["description"]
            assert provider1["auth_type"] == provider_payload_1["auth_type"]
            assert provider1["provider_type"] == provider_payload_1["provider_type"]
            assert provider1["endpoint"] == provider_payload_1["endpoint"]
            assert isinstance(provider1.get("id", ""), str) and provider1["id"]

            # Verify fields for second provider
            provider2 = next(p for p in providers if p["name"] == "openrouter-provider")
            assert provider2["description"] == provider_payload_2["description"]
            assert provider2["auth_type"] == provider_payload_2["auth_type"]
            assert provider2["provider_type"] == provider_payload_2["provider_type"]
            assert provider2["endpoint"] == provider_payload_2["endpoint"]
            assert isinstance(provider2.get("id", ""), str) and provider2["id"]

            # Get OpenAI provider by name
            response = await ac.get("/api/v1/provider-endpoints/openai-provider")
            assert response.status_code == 200
            provider = response.json()
            assert provider["name"] == provider_payload_1["name"]
            assert provider["description"] == provider_payload_1["description"]
            assert provider["auth_type"] == provider_payload_1["auth_type"]
            assert provider["provider_type"] == provider_payload_1["provider_type"]
            assert provider["endpoint"] == provider_payload_1["endpoint"]
            assert isinstance(provider["id"], str) and provider["id"]

            # Get OpenRouter provider by name
            response = await ac.get("/api/v1/provider-endpoints/openrouter-provider")
            assert response.status_code == 200
            provider = response.json()
            assert provider["name"] == provider_payload_2["name"]
            assert provider["description"] == provider_payload_2["description"]
            assert provider["auth_type"] == provider_payload_2["auth_type"]
            assert provider["provider_type"] == provider_payload_2["provider_type"]
            assert provider["endpoint"] == provider_payload_2["endpoint"]
            assert isinstance(provider["id"], str) and provider["id"]

            # Test getting non-existent provider
            response = await ac.get("/api/v1/provider-endpoints/non-existent")
            assert response.status_code == 404
            assert response.json()["detail"] == "Provider endpoint not found"

            # Test deleting providers
            response = await ac.delete("/api/v1/provider-endpoints/openai-provider")
            assert response.status_code == 204

            # Verify provider was deleted by trying to get it
            response = await ac.get("/api/v1/provider-endpoints/openai-provider")
            assert response.status_code == 404
            assert response.json()["detail"] == "Provider endpoint not found"

            # Delete second provider
            response = await ac.delete("/api/v1/provider-endpoints/openrouter-provider")
            assert response.status_code == 204

            # Verify second provider was deleted
            response = await ac.get("/api/v1/provider-endpoints/openrouter-provider")
            assert response.status_code == 404
            assert response.json()["detail"] == "Provider endpoint not found"

            # Test deleting non-existent provider
            response = await ac.delete("/api/v1/provider-endpoints/non-existent")
            assert response.status_code == 404
            assert response.json()["detail"] == "Provider endpoint not found"

            # Verify providers list is empty
            response = await ac.get("/api/v1/provider-endpoints")
            assert response.status_code == 200
            providers = response.json()
            assert len(providers) == 0


@pytest.mark.asyncio
async def test_list_providers_by_name(
    mock_pipeline_factory, mock_workspace_crud, mock_provider_crud
) -> None:
    with (
        patch("codegate.api.v1.wscrud", mock_workspace_crud),
        patch("codegate.api.v1.pcrud", mock_provider_crud),
        patch(
            "codegate.providers.openai.provider.OpenAIProvider.models",
            return_value=["gpt-4", "gpt-3.5-turbo"],
        ),
        patch(
            "codegate.providers.openrouter.provider.OpenRouterProvider.models",
            return_value=["anthropic/claude-2", "deepseek/deepseek-r1"],
        ),
    ):
        """Test creating multiple providers and listing them by name."""
        app = init_app(mock_pipeline_factory)

        async with AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as ac:
            # Create first provider (OpenAI)
            provider_payload_1 = {
                "name": "openai-provider",
                "description": "OpenAI provider description",
                "auth_type": "none",
                "provider_type": "openai",
                "endpoint": "https://api.openai.com",
                "api_key": "sk-proj-foo-bar-123-xyz",
            }

            response = await ac.post("/api/v1/provider-endpoints", json=provider_payload_1)
            assert response.status_code == 201

            # Create second provider (OpenRouter)
            provider_payload_2 = {
                "name": "openrouter-provider",
                "description": "OpenRouter provider description",
                "auth_type": "none",
                "provider_type": "openrouter",
                "endpoint": "https://openrouter.ai/api",
                "api_key": "sk-or-foo-bar-456-xyz",
            }

            response = await ac.post("/api/v1/provider-endpoints", json=provider_payload_2)
            assert response.status_code == 201

            # Test querying providers by name
            response = await ac.get("/api/v1/provider-endpoints?name=openai-provider")
            assert response.status_code == 200
            providers = response.json()
            assert len(providers) == 1
            assert providers[0]["name"] == "openai-provider"
            assert isinstance(providers[0]["id"], str) and providers[0]["id"]

            response = await ac.get("/api/v1/provider-endpoints?name=openrouter-provider")
            assert response.status_code == 200
            providers = response.json()
            assert len(providers) == 1
            assert providers[0]["name"] == "openrouter-provider"
            assert isinstance(providers[0]["id"], str) and providers[0]["id"]


@pytest.mark.asyncio
async def test_list_all_provider_models(
    mock_pipeline_factory, mock_workspace_crud, mock_provider_crud
) -> None:
    with (
        patch("codegate.api.v1.wscrud", mock_workspace_crud),
        patch("codegate.api.v1.pcrud", mock_provider_crud),
        patch(
            "codegate.providers.openai.provider.OpenAIProvider.models",
            return_value=["gpt-4", "gpt-3.5-turbo"],
        ),
        patch(
            "codegate.providers.openrouter.provider.OpenRouterProvider.models",
            return_value=["anthropic/claude-2", "deepseek/deepseek-r1"],
        ),
    ):
        """Test listing all models from all providers."""
        app = init_app(mock_pipeline_factory)

        async with AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as ac:
            # Create OpenAI provider
            provider_payload_1 = {
                "name": "openai-provider",
                "description": "OpenAI provider description",
                "auth_type": "none",
                "provider_type": "openai",
                "endpoint": "https://api.openai.com",
                "api_key": "sk-proj-foo-bar-123-xyz",
            }

            response = await ac.post("/api/v1/provider-endpoints", json=provider_payload_1)
            assert response.status_code == 201

            # Create OpenRouter provider
            provider_payload_2 = {
                "name": "openrouter-provider",
                "description": "OpenRouter provider description",
                "auth_type": "none",
                "provider_type": "openrouter",
                "endpoint": "https://openrouter.ai/api",
                "api_key": "sk-or-foo-bar-456-xyz",
            }

            response = await ac.post("/api/v1/provider-endpoints", json=provider_payload_2)
            assert response.status_code == 201

            # Get all models
            response = await ac.get("/api/v1/provider-endpoints/models")
            assert response.status_code == 200
            models = response.json()

            # Verify response structure and content
            assert isinstance(models, list)
            assert len(models) == 4

            # Verify models list structure
            assert all(isinstance(model, dict) for model in models)
            assert all("name" in model for model in models)
            assert all("provider_type" in model for model in models)
            assert all("provider_name" in model for model in models)

            # Verify OpenAI provider models
            openai_models = [m for m in models if m["provider_name"] == "openai-provider"]
            assert len(openai_models) == 2
            assert all(m["provider_type"] == "openai" for m in openai_models)

            # Verify OpenRouter provider models
            openrouter_models = [m for m in models if m["provider_name"] == "openrouter-provider"]
            assert len(openrouter_models) == 2
            assert all(m["provider_type"] == "openrouter" for m in openrouter_models)


@pytest.mark.asyncio
async def test_list_models_by_provider(
    mock_pipeline_factory, mock_workspace_crud, mock_provider_crud
) -> None:
    with (
        patch("codegate.api.v1.wscrud", mock_workspace_crud),
        patch("codegate.api.v1.pcrud", mock_provider_crud),
        patch(
            "codegate.providers.openai.provider.OpenAIProvider.models",
            return_value=["gpt-4", "gpt-3.5-turbo"],
        ),
        patch(
            "codegate.providers.openrouter.provider.OpenRouterProvider.models",
            return_value=["anthropic/claude-2", "deepseek/deepseek-r1"],
        ),
    ):
        """Test listing models for a specific provider."""
        app = init_app(mock_pipeline_factory)

        async with AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as ac:
            # Create OpenAI provider
            provider_payload = {
                "name": "openai-provider",
                "description": "OpenAI provider description",
                "auth_type": "none",
                "provider_type": "openai",
                "endpoint": "https://api.openai.com",
                "api_key": "sk-proj-foo-bar-123-xyz",
            }

            response = await ac.post("/api/v1/provider-endpoints", json=provider_payload)
            assert response.status_code == 201
            provider = response.json()
            provider_id = provider["id"]

            # Get models for the provider
            response = await ac.get(f"/api/v1/provider-endpoints/{provider_id}/models")
            assert response.status_code == 200
            models = response.json()

            # Verify response structure and content
            assert isinstance(models, list)
            assert len(models) == 2
            assert all(isinstance(model, dict) for model in models)
            assert all("name" in model for model in models)
            assert all("provider_type" in model for model in models)
            assert all("provider_name" in model for model in models)
            assert all(model["provider_type"] == "openai" for model in models)
            assert all(model["provider_name"] == "openai-provider" for model in models)

            # Test with non-existent provider ID
            fake_uuid = str(uuid())
            response = await ac.get(f"/api/v1/provider-endpoints/{fake_uuid}/models")
            assert response.status_code == 404
            assert response.json()["detail"] == "Provider not found"
