from unittest.mock import AsyncMock, patch

import pytest

from codegate.db.models import WorkspaceActive
from codegate.pipeline.cli.commands import Workspace


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mock_workspaces, expected_output",
    [
        # Case 1: No workspaces
        ([], ""),
        # Case 2: One workspace active
        (
            [
                # We'll make a MagicMock that simulates a workspace
                # with 'name' attribute and 'active_workspace_id' set
                WorkspaceActive(id="1", name="Workspace1", active_workspace_id="100")
            ],
            "- Workspace1 **(active)**\n",
        ),
        # Case 3: Multiple workspaces, second one active
        (
            [
                WorkspaceActive(id="1", name="Workspace1", active_workspace_id=None),
                WorkspaceActive(id="2", name="Workspace2", active_workspace_id="200"),
            ],
            "- Workspace1\n- Workspace2 **(active)**\n",
        ),
    ],
)
async def test_list_workspaces(mock_workspaces, expected_output):
    """
    Test _list_workspaces with different sets of returned workspaces.
    """
    workspace_commands = Workspace()

    # Mock DbReader inside workspace_commands
    mock_get_workspaces = AsyncMock(return_value=mock_workspaces)
    workspace_commands.workspace_crud.get_workspaces = mock_get_workspaces

    # Call the method
    result = await workspace_commands._list_workspaces(None, None)

    # Check the result
    assert result == expected_output
    mock_get_workspaces.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args, existing_workspaces, expected_message",
    [
        # Case 1: No workspace name provided
        ([], [], "Please provide a name. Use `codegate workspace add your_workspace_name`"),
        # Case 2: Workspace name is empty string
        ([""], [], "Please provide a name. Use `codegate workspace add your_workspace_name`"),
        # Case 3: Successful add
        (["myworkspace"], [], "Workspace **myworkspace** has been added"),
    ],
)
async def test_add_workspaces(args, existing_workspaces, expected_message):
    """
    Test _add_workspace under different scenarios:
    - no args
    - empty string arg
    - workspace already exists
    - workspace successfully added
    """
    workspace_commands = Workspace()

    # Mock the DbReader to return existing_workspaces
    mock_db_reader = AsyncMock()
    mock_db_reader.get_workspace_by_name.return_value = existing_workspaces
    workspace_commands._db_reader = mock_db_reader

    # We'll also patch DbRecorder to ensure no real DB operations happen
    with patch("codegate.workspaces.crud.WorkspaceCrud", autospec=True) as mock_recorder_cls:
        mock_recorder = mock_recorder_cls.return_value
        workspace_commands.workspace_crud = mock_recorder
        mock_recorder.add_workspace = AsyncMock()

        # Call the method
        result = await workspace_commands._add_workspace(None, args)

        # Assertions
        assert result == expected_message

        # If expected_message indicates "added", we expect add_workspace to be called once
        if "has been added" in expected_message:
            mock_recorder.add_workspace.assert_awaited_once_with(args[0])
        else:
            mock_recorder.add_workspace.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_message, expected_command, expected_args, mocked_execute_response",
    [
        (["list"], "list", ["list"], "List workspaces output"),
        (["add", "myws"], "add", ["add", "myws"], "Added workspace"),
        (["activate", "myws"], "activate", ["activate", "myws"], "Activated workspace"),
    ],
)
async def test_parse_execute_cmd(
    user_message, expected_command, expected_args, mocked_execute_response
):
    """
    Test parse_execute_cmd to ensure it parses the user message
    and calls the correct command with the correct args.
    """
    workspace_commands = Workspace()

    with patch.object(workspace_commands, "run", return_value=mocked_execute_response) as mock_run:
        result = await workspace_commands.exec(user_message)
        assert result == mocked_execute_response

        # Verify 'execute' was called with the expected command and args
        mock_run.assert_awaited_once_with(expected_args)
