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
async def test_workspace_crud(
    mock_pipeline_factory, mock_workspace_crud, mock_provider_crud, db_reader
) -> None:
    with (
        patch("codegate.api.v1.dbreader", db_reader),
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
        """Test creating, updating and reading a workspace."""

        app = init_app(mock_pipeline_factory)

        provider_payload_1 = {
            "name": "openai-provider",
            "description": "OpenAI provider description",
            "auth_type": "none",
            "provider_type": "openai",
            "endpoint": "https://api.openai.com",
            "api_key": "sk-proj-foo-bar-123-xyz",
        }

        provider_payload_2 = {
            "name": "openrouter-provider",
            "description": "OpenRouter provider description",
            "auth_type": "none",
            "provider_type": "openrouter",
            "endpoint": "https://openrouter.ai/api",
            "api_key": "sk-or-foo-bar-456-xyz",
        }

        async with AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.post("/api/v1/provider-endpoints", json=provider_payload_1)
            assert response.status_code == 201

            response = await ac.post("/api/v1/provider-endpoints", json=provider_payload_2)
            assert response.status_code == 201

            # Create workspace

            name_1: str = str(uuid())
            custom_instructions_1: str = "Respond to every request in iambic pentameter"
            muxing_rules_1 = [
                {
                    "provider_name": "openai-provider",
                    "provider_type": "openai",
                    "model": "gpt-4",
                    "matcher": "*.ts",
                    "matcher_type": "filename_match",
                },
                {
                    "provider_name": "openai-provider",
                    "provider_type": "openai",
                    "model": "gpt-3.5-turbo",
                    "matcher_type": "catch_all",
                    "matcher": "",
                },
            ]

            payload_create = {
                "name": name_1,
                "config": {
                    "custom_instructions": custom_instructions_1,
                    "muxing_rules": muxing_rules_1,
                },
            }

            response = await ac.post("/api/v1/workspaces", json=payload_create)
            assert response.status_code == 201

            # Verify created workspace
            response = await ac.get(f"/api/v1/workspaces/{name_1}")
            assert response.status_code == 200
            response_body = response.json()

            assert response_body["name"] == name_1
            assert response_body["config"]["custom_instructions"] == custom_instructions_1
            for i, rule in enumerate(response_body["config"]["muxing_rules"]):
                assert rule["provider_name"] == muxing_rules_1[i]["provider_name"]
                assert rule["provider_type"] == muxing_rules_1[i]["provider_type"]
                assert rule["model"] == muxing_rules_1[i]["model"]
                assert rule["matcher"] == muxing_rules_1[i]["matcher"]
                assert rule["matcher_type"] == muxing_rules_1[i]["matcher_type"]

            name_2: str = str(uuid())
            custom_instructions_2: str = "Respond to every request in cockney rhyming slang"
            muxing_rules_2 = [
                {
                    "provider_name": "openrouter-provider",
                    "provider_type": "openrouter",
                    "model": "anthropic/claude-2",
                    "matcher": "*.ts",
                    "matcher_type": "filename_match",
                },
                {
                    "provider_name": "openrouter-provider",
                    "provider_type": "openrouter",
                    "model": "deepseek/deepseek-r1",
                    "matcher_type": "catch_all",
                    "matcher": "",
                },
            ]

            payload_update = {
                "name": name_2,
                "config": {
                    "custom_instructions": custom_instructions_2,
                    "muxing_rules": muxing_rules_2,
                },
            }

            response = await ac.put(f"/api/v1/workspaces/{name_1}", json=payload_update)
            assert response.status_code == 200

            # Verify updated workspace
            response = await ac.get(f"/api/v1/workspaces/{name_2}")
            assert response.status_code == 200
            response_body = response.json()

            assert response_body["name"] == name_2
            assert response_body["config"]["custom_instructions"] == custom_instructions_2
            for i, rule in enumerate(response_body["config"]["muxing_rules"]):
                assert rule["provider_name"] == muxing_rules_2[i]["provider_name"]
                assert rule["provider_type"] == muxing_rules_2[i]["provider_type"]
                assert rule["model"] == muxing_rules_2[i]["model"]
                assert rule["matcher"] == muxing_rules_2[i]["matcher"]
                assert rule["matcher_type"] == muxing_rules_2[i]["matcher_type"]


@pytest.mark.asyncio
async def test_rename_workspace(
    mock_pipeline_factory, mock_workspace_crud, mock_provider_crud, db_reader
) -> None:
    with (
        patch("codegate.api.v1.dbreader", db_reader),
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
        """Test renaming a workspace."""

        app = init_app(mock_pipeline_factory)

        provider_payload_1 = {
            "name": "openai-provider",
            "description": "OpenAI provider description",
            "auth_type": "none",
            "provider_type": "openai",
            "endpoint": "https://api.openai.com",
            "api_key": "sk-proj-foo-bar-123-xyz",
        }

        provider_payload_2 = {
            "name": "openrouter-provider",
            "description": "OpenRouter provider description",
            "auth_type": "none",
            "provider_type": "openrouter",
            "endpoint": "https://openrouter.ai/api",
            "api_key": "sk-or-foo-bar-456-xyz",
        }

        async with AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.post("/api/v1/provider-endpoints", json=provider_payload_1)
            assert response.status_code == 201

            response = await ac.post("/api/v1/provider-endpoints", json=provider_payload_2)
            assert response.status_code == 201

            # Create workspace

            name_1: str = str(uuid())
            custom_instructions: str = "Respond to every request in iambic pentameter"
            muxing_rules = [
                {
                    "provider_name": "openai-provider",
                    "provider_type": "openai",
                    "model": "gpt-4",
                    "matcher": "*.ts",
                    "matcher_type": "filename_match",
                },
                {
                    "provider_name": "openai-provider",
                    "provider_type": "openai",
                    "model": "gpt-3.5-turbo",
                    "matcher_type": "catch_all",
                    "matcher": "",
                },
            ]

            payload_create = {
                "name": name_1,
                "config": {
                    "custom_instructions": custom_instructions,
                    "muxing_rules": muxing_rules,
                },
            }

            response = await ac.post("/api/v1/workspaces", json=payload_create)
            assert response.status_code == 201
            response_body = response.json()
            assert response_body["name"] == name_1

            # Verify created workspace
            response = await ac.get(f"/api/v1/workspaces/{name_1}")
            assert response.status_code == 200
            response_body = response.json()
            assert response_body["name"] == name_1

            name_2: str = str(uuid())

            payload_update = {
                "name": name_2,
            }

            response = await ac.put(f"/api/v1/workspaces/{name_1}", json=payload_update)
            assert response.status_code == 200
            response_body = response.json()
            assert response_body["name"] == name_2

            # other fields shouldn't have been touched
            assert response_body["config"]["custom_instructions"] == custom_instructions
            for i, rule in enumerate(response_body["config"]["muxing_rules"]):
                assert rule["provider_name"] == muxing_rules[i]["provider_name"]
                assert rule["provider_type"] == muxing_rules[i]["provider_type"]
                assert rule["model"] == muxing_rules[i]["model"]
                assert rule["matcher"] == muxing_rules[i]["matcher"]
                assert rule["matcher_type"] == muxing_rules[i]["matcher_type"]

            # Verify updated workspace
            response = await ac.get(f"/api/v1/workspaces/{name_2}")
            assert response.status_code == 200
            response_body = response.json()
            assert response_body["name"] == name_2


@pytest.mark.asyncio
async def test_create_workspace_name_already_in_use(
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
        """Test creating a workspace when the name is already in use."""

        app = init_app(mock_pipeline_factory)

        async with AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as ac:
            name: str = str(uuid())

            payload_create = {
                "name": name,
            }

            # Create the workspace for the first time
            response = await ac.post("/api/v1/workspaces", json=payload_create)
            assert response.status_code == 201

            # Try to create the workspace again with the same name
            response = await ac.post("/api/v1/workspaces", json=payload_create)
            assert response.status_code == 409
            assert response.json()["detail"] == "Workspace name already in use"


@pytest.mark.asyncio
async def test_rename_workspace_name_already_in_use(
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
        """Test renaming a workspace when the new name is already in use."""

        app = init_app(mock_pipeline_factory)

        async with AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as ac:
            name_1: str = str(uuid())
            name_2: str = str(uuid())

            payload_create_1 = {
                "name": name_1,
            }

            payload_create_2 = {
                "name": name_2,
            }

            # Create two workspaces
            response = await ac.post("/api/v1/workspaces", json=payload_create_1)
            assert response.status_code == 201

            response = await ac.post("/api/v1/workspaces", json=payload_create_2)
            assert response.status_code == 201

            # Try to rename the first workspace to the name of the second workspace
            payload_update = {
                "name": name_2,
            }

            response = await ac.put(f"/api/v1/workspaces/{name_1}", json=payload_update)
            assert response.status_code == 409
            assert response.json()["detail"] == "Workspace name already in use"


@pytest.mark.asyncio
async def test_create_workspace_with_nonexistent_model_in_muxing_rule(
    mock_pipeline_factory, mock_workspace_crud, mock_provider_crud, db_reader
) -> None:
    with (
        patch("codegate.api.v1.dbreader", db_reader),
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
        """Test creating a workspace with a muxing rule that uses a nonexistent model."""

        app = init_app(mock_pipeline_factory)

        provider_payload = {
            "name": "openai-provider",
            "description": "OpenAI provider description",
            "auth_type": "none",
            "provider_type": "openai",
            "endpoint": "https://api.openai.com",
            "api_key": "sk-proj-foo-bar-123-xyz",
        }

        async with AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.post("/api/v1/provider-endpoints", json=provider_payload)
            assert response.status_code == 201

            name: str = str(uuid())
            custom_instructions: str = "Respond to every request in iambic pentameter"
            muxing_rules = [
                {
                    "provider_name": "openai-provider",
                    "provider_type": "openai",
                    "model": "nonexistent-model",
                    "matcher": "*.ts",
                    "matcher_type": "filename_match",
                },
            ]

            payload_create = {
                "name": name,
                "config": {
                    "custom_instructions": custom_instructions,
                    "muxing_rules": muxing_rules,
                },
            }

            response = await ac.post("/api/v1/workspaces", json=payload_create)
            assert response.status_code == 400
            assert "does not exist" in response.json()["detail"]
