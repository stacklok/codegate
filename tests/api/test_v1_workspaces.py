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
from codegate.providers.registry import ProviderRegistry
from codegate.server import init_app
from codegate.workspaces.crud import WorkspaceCrud
from tests.providers.test_registry import MockProvider

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
    """Creates a DbRecorder instance with test database."""
    ws_crud = WorkspaceCrud()
    ws_crud._db_reader = db_reader
    ws_crud._db_recorder = db_recorder
    return ws_crud


@pytest.fixture()
def mock_provider_crud(db_recorder, db_reader, mock_workspace_crud) -> ProviderCrud:
    """Creates a DbRecorder instance with test database."""
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

        registry = ProviderRegistry(app)

        registry.add_provider("ollama", MockProvider())

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

            name_2: str = str(uuid())

            payload_update = {
                "name": name_2,
            }

            response = await ac.put(f"/api/v1/workspaces/{name_1}", json=payload_update)
            assert response.status_code == 201
            response_body = response.json()

            assert response_body["name"] == name_2
