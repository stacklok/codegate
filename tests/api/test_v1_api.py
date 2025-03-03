import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI
from codegate.api.v1 import v1
from codegate.db.models import Alert, AlertSeverity, GetPromptWithOutputsRow
from codegate.workspaces.crud import WorkspaceDoesNotExistError  # Import the APIRouter instance

# Create a FastAPI test app and include the APIRouter
app = FastAPI()
app.include_router(v1)
client = TestClient(app)


@pytest.fixture
def mock_ws():
    """Mock workspace object"""
    ws = AsyncMock()
    ws.id = "test_workspace_id"
    return ws


@pytest.fixture
def mock_alerts():
    """Mock alerts list"""
    return [
        Alert(
            id="1",
            prompt_id="p1",
            code_snippet="code",
            trigger_string="error",
            trigger_type="type",
            trigger_category=AlertSeverity.CRITICAL.value,
            timestamp="2024-03-03T12:34:56Z",
        ),
        Alert(
            id="2",
            prompt_id="p2",
            code_snippet="code2",
            trigger_string="error2",
            trigger_type="type2",
            trigger_category=AlertSeverity.CRITICAL.value,
            timestamp="2024-03-03T12:35:56Z",
        ),
    ]


@pytest.fixture
def mock_prompts():
    """Mock prompts output list"""
    return [
        GetPromptWithOutputsRow(
            id="p1",
            timestamp="2024-03-03T12:34:56Z",
            provider="provider",
            request="req",
            type="type",
            output_id="o1",
            output="output",
            output_timestamp="2024-03-03T12:35:56Z",
            input_tokens=10,
            output_tokens=15,
            input_cost=0.01,
            output_cost=0.02,
        ),
        GetPromptWithOutputsRow(
            id="p2",
            timestamp="2024-03-03T12:36:56Z",
            provider="provider2",
            request="req2",
            type="type2",
            output_id="o2",
            output="output2",
            output_timestamp="2024-03-03T12:37:56Z",
            input_tokens=20,
            output_tokens=25,
            input_cost=0.02,
            output_cost=0.03,
        ),
    ]


@pytest.mark.asyncio
async def test_get_workspace_alerts_not_found():
    """Test when workspace does not exist (404 error)"""
    with patch(
        "codegate.workspaces.crud.WorkspaceCrud.get_workspace_by_name",
        side_effect=WorkspaceDoesNotExistError("Workspace does not exist"),
    ):
        response = client.get("/workspaces/non_existent_workspace/alerts")
        assert response.status_code == 404
        assert response.json()["detail"] == "Workspace does not exist"


@pytest.mark.asyncio
async def test_get_workspace_alerts_internal_server_error():
    """Test when an internal error occurs (500 error)"""
    with patch(
        "codegate.workspaces.crud.WorkspaceCrud.get_workspace_by_name",
        side_effect=Exception("Unexpected error"),
    ):
        response = client.get("/workspaces/test_workspace/alerts")
        assert response.status_code == 500
        assert response.json()["detail"] == "Internal server error"


@pytest.mark.asyncio
async def test_get_workspace_alerts_empty(mock_ws):
    """Test when no alerts are found (empty list)"""
    with (
        patch("codegate.workspaces.crud.WorkspaceCrud.get_workspace_by_name", return_value=mock_ws),
        patch("codegate.db.connection.DbReader.get_alerts_by_workspace", return_value=[]),
    ):

        response = client.get("/workspaces/test_workspace/alerts?page=1&page_size=10")
        assert response.status_code == 200
        assert response.json() == {
            "page": 1,
            "alerts": [],
        }


@pytest.mark.asyncio
async def test_get_workspace_alerts_with_results(mock_ws, mock_alerts, mock_prompts):
    """Test when valid alerts are retrieved with pagination"""
    with (
        patch("codegate.workspaces.crud.WorkspaceCrud.get_workspace_by_name", return_value=mock_ws),
        patch(
            "codegate.db.connection.DbReader.get_alerts_by_workspace",
            return_value=(mock_alerts, len(mock_alerts)),
        ),
        patch("codegate.db.connection.DbReader.get_prompts_with_output", return_value=mock_prompts),
        patch("codegate.api.v1_processing.remove_duplicate_alerts", return_value=mock_alerts),
        patch("codegate.api.v1_processing.parse_get_alert_conversation", return_value=mock_alerts),
    ):

        response = client.get("/workspaces/test_workspace/alerts?page=1&page_size=2")
        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 1
        assert len(data["alerts"]) == 2


@pytest.mark.asyncio
async def test_get_workspace_alerts_deduplication(mock_ws, mock_alerts, mock_prompts):
    """Test that alerts are fetched iteratively when deduplication reduces results"""
    dedup_alerts = [mock_alerts[0]]  # Simulate deduplication removing one alert

    with (
        patch("codegate.workspaces.crud.WorkspaceCrud.get_workspace_by_name", return_value=mock_ws),
        patch(
            "codegate.db.connection.DbReader.get_alerts_by_workspace",
            side_effect=[(mock_alerts, 2), (mock_alerts, 2)],
        ),
        patch("codegate.db.connection.DbReader.get_prompts_with_output", return_value=mock_prompts),
        patch("codegate.api.v1_processing.remove_duplicate_alerts", return_value=dedup_alerts),
        patch("codegate.api.v1_processing.parse_get_alert_conversation", return_value=dedup_alerts),
    ):

        response = client.get("/workspaces/test_workspace/alerts?page=1&page_size=2")
        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 1
        assert len(data["alerts"]) == 1  # Only one alert left after deduplication
