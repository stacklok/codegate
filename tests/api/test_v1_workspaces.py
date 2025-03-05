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
async def test_get_workspaces(
    mock_pipeline_factory, mock_workspace_crud, mock_provider_crud
) -> None:
    with (
        patch("codegate.api.v1.wscrud", mock_workspace_crud),
        patch("codegate.api.v1.pcrud", mock_provider_crud),
        patch(
            "codegate.providers.openai.provider.OpenAIProvider.models",
            return_value=["foo-bar-001", "foo-bar-002"],
        ),
    ):
        """Test getting all workspaces."""
        app = init_app(mock_pipeline_factory)

        async with AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as ac:
            # Create a provider for muxing rules
            provider_payload = {
                "name": "test-provider",
                "description": "",
                "auth_type": "none",
                "provider_type": "openai",
                "endpoint": "https://api.openai.com",
                "api_key": "sk-proj-foo-bar-123-xzy",
            }
            response = await ac.post("/api/v1/provider-endpoints", json=provider_payload)
            assert response.status_code == 201
            provider = response.json()

            # Create first workspace
            name_1 = str(uuid())
            workspace_1 = {
                "name": name_1,
                "config": {
                    "custom_instructions": "Respond in haiku format",
                    "muxing_rules": [
                        {
                            "provider_id": provider["id"],
                            "model": "foo-bar-001",
                            "matcher": "*.py",
                            "matcher_type": "filename_match",
                        }
                    ],
                },
            }
            response = await ac.post("/api/v1/workspaces", json=workspace_1)
            assert response.status_code == 201

            # Create second workspace
            name_2 = str(uuid())
            workspace_2 = {
                "name": name_2,
                "config": {
                    "custom_instructions": "Respond in prose",
                    "muxing_rules": [
                        {
                            "provider_id": provider["id"],
                            "model": "foo-bar-002",
                            "matcher": "*.js",
                            "matcher_type": "filename_match",
                        }
                    ],
                },
            }
            response = await ac.post("/api/v1/workspaces", json=workspace_2)
            assert response.status_code == 201

            response = await ac.get("/api/v1/workspaces")
            assert response.status_code == 200
            workspaces = response.json()["workspaces"]

            # Verify response structure
            assert isinstance(workspaces, list)
            assert len(workspaces) >= 2

            workspace_names = [w["name"] for w in workspaces]
            assert name_1 in workspace_names
            assert name_2 in workspace_names
            assert len([n for n in workspace_names if n in [name_1, name_2]]) == 2


@pytest.mark.asyncio
async def test_get_workspaces_filter_by_provider(
    mock_pipeline_factory, mock_workspace_crud, mock_provider_crud
) -> None:
    with (
        patch("codegate.api.v1.wscrud", mock_workspace_crud),
        patch("codegate.api.v1.pcrud", mock_provider_crud),
        patch(
            "codegate.providers.openai.provider.OpenAIProvider.models",
            return_value=["foo-bar-001", "foo-bar-002"],
        ),
    ):
        """Test filtering workspaces by provider ID."""
        app = init_app(mock_pipeline_factory)

        async with AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as ac:
            # Create first provider
            provider_payload_1 = {
                "name": "provider-1",
                "description": "",
                "auth_type": "none",
                "provider_type": "openai",
                "endpoint": "https://api.openai.com",
                "api_key": "sk-proj-foo-bar-123-xyz",
            }
            response = await ac.post("/api/v1/provider-endpoints", json=provider_payload_1)
            assert response.status_code == 201
            provider_1 = response.json()

            # Create second provider
            provider_payload_2 = {
                "name": "provider-2",
                "description": "",
                "auth_type": "none",
                "provider_type": "openai",
                "endpoint": "https://api.openai.com",
                "api_key": "sk-proj-foo-bar-456-xyz",
            }
            response = await ac.post("/api/v1/provider-endpoints", json=provider_payload_2)
            assert response.status_code == 201
            provider_2 = response.json()

            # Create workspace using provider 1
            workspace_1 = {
                "name": str(uuid()),
                "config": {
                    "custom_instructions": "Instructions 1",
                    "muxing_rules": [
                        {
                            "provider_id": provider_1["id"],
                            "model": "foo-bar-001",
                            "matcher": "*.py",
                            "matcher_type": "filename_match",
                        }
                    ],
                },
            }
            response = await ac.post("/api/v1/workspaces", json=workspace_1)
            assert response.status_code == 201

            # Create workspace using provider 2
            workspace_2 = {
                "name": str(uuid()),
                "config": {
                    "custom_instructions": "Instructions 2",
                    "muxing_rules": [
                        {
                            "provider_id": provider_2["id"],
                            "model": "foo-bar-002",
                            "matcher": "*.js",
                            "matcher_type": "filename_match",
                        }
                    ],
                },
            }
            response = await ac.post("/api/v1/workspaces", json=workspace_2)
            assert response.status_code == 201

            # Test filtering by provider 1
            response = await ac.get(f"/api/v1/workspaces?provider_id={provider_1['id']}")
            assert response.status_code == 200
            workspaces = response.json()["workspaces"]
            assert len(workspaces) == 1
            assert workspaces[0]["name"] == workspace_1["name"]

            # Test filtering by provider 2
            response = await ac.get(f"/api/v1/workspaces?provider_id={provider_2['id']}")
            assert response.status_code == 200
            workspaces = response.json()["workspaces"]
            assert len(workspaces) == 1
            assert workspaces[0]["name"] == workspace_2["name"]


@pytest.mark.asyncio
async def test_create_update_workspace_happy_path(
    mock_pipeline_factory, mock_workspace_crud, mock_provider_crud
) -> None:
    with (
        patch("codegate.api.v1.wscrud", mock_workspace_crud),
        patch("codegate.api.v1.pcrud", mock_provider_crud),
        patch(
            "codegate.providers.openai.provider.OpenAIProvider.models",
            return_value=["foo-bar-001", "foo-bar-002"],
        ),
    ):
        """Test creating & updating a workspace (happy path)."""

        app = init_app(mock_pipeline_factory)

        provider_payload_1 = {
            "name": "foo",
            "description": "",
            "auth_type": "none",
            "provider_type": "openai",
            "endpoint": "https://api.openai.com",
            "api_key": "sk-proj-foo-bar-123-xzy",
        }

        provider_payload_2 = {
            "name": "bar",
            "description": "",
            "auth_type": "none",
            "provider_type": "openai",
            "endpoint": "https://api.openai.com",
            "api_key": "sk-proj-foo-bar-123-xzy",
        }

        async with AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as ac:
            # Create the first provider
            response = await ac.post("/api/v1/provider-endpoints", json=provider_payload_1)
            assert response.status_code == 201
            provider_1 = response.json()

            # Create the second provider
            response = await ac.post("/api/v1/provider-endpoints", json=provider_payload_2)
            assert response.status_code == 201
            provider_2 = response.json()

            name_1: str = str(uuid())
            custom_instructions_1: str = "Respond to every request in iambic pentameter"
            muxing_rules_1 = [
                {
                    "provider_name": None,  # optional & not implemented yet
                    "provider_id": provider_1["id"],
                    "model": "foo-bar-001",
                    "matcher": "*.ts",
                    "matcher_type": "filename_match",
                },
                {
                    "provider_name": None,  # optional & not implemented yet
                    "provider_id": provider_2["id"],
                    "model": "foo-bar-002",
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
                assert rule["model"] == muxing_rules_1[i]["model"]
                assert rule["matcher"] == muxing_rules_1[i]["matcher"]
                assert rule["matcher_type"] == muxing_rules_1[i]["matcher_type"]

            name_2: str = str(uuid())
            custom_instructions_2: str = "Respond to every request in cockney rhyming slang"
            muxing_rules_2 = [
                {
                    "provider_name": None,  # optional & not implemented yet
                    "provider_id": provider_2["id"],
                    "model": "foo-bar-002",
                    "matcher": "*.ts",
                    "matcher_type": "filename_match",
                },
                {
                    "provider_name": None,  # optional & not implemented yet
                    "provider_id": provider_1["id"],
                    "model": "foo-bar-001",
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
            assert response.status_code == 201

            # Verify updated workspace
            response = await ac.get(f"/api/v1/workspaces/{name_2}")
            assert response.status_code == 200
            response_body = response.json()

            assert response_body["name"] == name_2
            assert response_body["config"]["custom_instructions"] == custom_instructions_2
            for i, rule in enumerate(response_body["config"]["muxing_rules"]):
                assert rule["model"] == muxing_rules_2[i]["model"]
                assert rule["matcher"] == muxing_rules_2[i]["matcher"]
                assert rule["matcher_type"] == muxing_rules_2[i]["matcher_type"]


@pytest.mark.asyncio
async def test_create_update_workspace_name_only(
    mock_pipeline_factory, mock_workspace_crud, mock_provider_crud
) -> None:
    with (
        patch("codegate.api.v1.wscrud", mock_workspace_crud),
        patch("codegate.api.v1.pcrud", mock_provider_crud),
        patch(
            "codegate.providers.openai.provider.OpenAIProvider.models",
            return_value=["foo-bar-001", "foo-bar-002"],
        ),
    ):
        """Test creating & updating a workspace (happy path)."""

        app = init_app(mock_pipeline_factory)

        async with AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as ac:
            name_1: str = str(uuid())

            payload_create = {
                "name": name_1,
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
            assert response_body["config"]["custom_instructions"] == ""
            assert response_body["config"]["muxing_rules"] == []

            name_2: str = str(uuid())

            payload_update = {
                "name": name_2,
            }

            response = await ac.put(f"/api/v1/workspaces/{name_1}", json=payload_update)
            assert response.status_code == 201
            response_body = response.json()
            assert response_body["name"] == name_2

            # Verify updated workspace
            response = await ac.get(f"/api/v1/workspaces/{name_2}")
            assert response.status_code == 200
            response_body = response.json()
            assert response_body["name"] == name_2
            assert response_body["config"]["custom_instructions"] == ""
            assert response_body["config"]["muxing_rules"] == []


@pytest.mark.asyncio
async def test_create_workspace_name_already_in_use(
    mock_pipeline_factory, mock_workspace_crud, mock_provider_crud
) -> None:
    with (
        patch("codegate.api.v1.wscrud", mock_workspace_crud),
        patch("codegate.api.v1.pcrud", mock_provider_crud),
        patch(
            "codegate.providers.openai.provider.OpenAIProvider.models",
            return_value=["foo-bar-001", "foo-bar-002"],
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
            return_value=["foo-bar-001", "foo-bar-002"],
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
    mock_pipeline_factory, mock_workspace_crud, mock_provider_crud
) -> None:
    with (
        patch("codegate.api.v1.wscrud", mock_workspace_crud),
        patch("codegate.api.v1.pcrud", mock_provider_crud),
        patch(
            "codegate.providers.openai.provider.OpenAIProvider.models",
            return_value=["foo-bar-001", "foo-bar-002"],
        ),
    ):
        """Test creating a workspace with a muxing rule that uses a nonexistent model."""

        app = init_app(mock_pipeline_factory)

        provider_payload = {
            "name": "foo",
            "description": "",
            "auth_type": "none",
            "provider_type": "openai",
            "endpoint": "https://api.openai.com",
            "api_key": "sk-proj-foo-bar-123-xzy",
        }

        async with AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as ac:
            # Create the first provider
            response = await ac.post("/api/v1/provider-endpoints", json=provider_payload)
            assert response.status_code == 201
            provider = response.json()

            name: str = str(uuid())
            custom_instructions: str = "Respond to every request in iambic pentameter"
            muxing_rules = [
                {
                    "provider_name": None,
                    "provider_id": provider["id"],
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
            assert "Model nonexistent-model does not exist" in response.json()["detail"]
