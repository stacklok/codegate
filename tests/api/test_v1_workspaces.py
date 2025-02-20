from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4 as uuid

import httpx
import pytest
from httpx import AsyncClient

from codegate.db import connection
from codegate.pipeline.factory import PipelineFactory
from codegate.server import init_app


@pytest.fixture(scope="function")
def db_path():
    """Creates a temporary database file path."""
    current_test_dir = Path(__file__).parent
    db_filepath = current_test_dir / "codegate_test.db"
    db_fullpath = db_filepath.absolute()
    connection.init_db_sync(db_fullpath)
    yield db_fullpath
    if db_fullpath.is_file():
        db_fullpath.unlink()


@pytest.fixture(scope="function")
def db_recorder(db_path) -> connection.DbRecorder:
    """Creates a DbRecorder instance with test database."""
    return connection.DbRecorder(sqlite_path=db_path, _no_singleton=True)


@pytest.fixture(scope="function")
def db_reader(db_path) -> connection.DbReader:
    """Creates a DbReader instance with test database."""
    return connection.DbReader(sqlite_path=db_path, _no_singleton=True)


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
async def test_create_workspace_happy_path(mock_pipeline_factory, db_recorder, db_reader) -> None:
    """Test creating a workspace (happy path)."""
    app = init_app(mock_pipeline_factory)

    name: str = str(uuid())
    custom_instructions: str = "Wiggly jiggly"
    muxing_rules = [
        {
            "provider_name": None,
            "provider_id": "0653607e-bf47-42d8-9a9d-86cdca3cf19f",
            "model": "deepseek-r1:1.5b",
            "matcher": "*.ts",
            "matcher_type": "filename_match",
        },
        {
            "provider_name": None,
            "provider_id": "0653607e-bf47-42d8-9a9d-86cdca3cf19f",
            "model": "deepseek-r1:1.5b",
            "matcher_type": "catch_all",
            "matcher": "",
        },
    ]

    payload = {
        "name": name,
        "config": {"custom_instructions": custom_instructions, "muxing_rules": muxing_rules},
    }
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/api/v1/workspaces", json=payload)
        assert response.status_code == 201
        response_body = response.json()
        print(response_body["config"]["muxing_rules"])

        assert response_body["name"] == name
        assert response_body["config"]["custom_instructions"] == custom_instructions
        assert response_body["config"]["muxing_rules"] == muxing_rules
