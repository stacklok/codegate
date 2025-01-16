import datetime
from unittest.mock import AsyncMock, patch

import pytest

from codegate.db.models import Session, Workspace, WorkspaceActive
from codegate.pipeline.workspace.workspace import WorkspaceCommands


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
    workspace_commands = WorkspaceCommands()

    # Mock DbReader inside workspace_commands
    mock_db_reader = AsyncMock()
    mock_db_reader.get_workspaces.return_value = mock_workspaces
    workspace_commands._db_reader = mock_db_reader

    # Call the method
    result = await workspace_commands._list_workspaces()

    # Check the result
    assert result == expected_output
    mock_db_reader.get_workspaces.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args, existing_workspaces, expected_message",
    [
        # Case 1: No workspace name provided
        ([], [], "Please provide a name. Use `codegate-workspace add your_workspace_name`"),
        # Case 2: Workspace name is empty string
        ([""], [], "Please provide a name. Use `codegate-workspace add your_workspace_name`"),
        # Case 3: Workspace already exists
        (
            ["myworkspace"],
            [Workspace(name="myworkspace", id="1")],
            "Workspace **myworkspace** already exists",
        ),
        # Case 4: Successful add
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
    workspace_commands = WorkspaceCommands()

    # Mock the DbReader to return existing_workspaces
    mock_db_reader = AsyncMock()
    mock_db_reader.get_workspace_by_name.return_value = existing_workspaces
    workspace_commands._db_reader = mock_db_reader

    # We'll also patch DbRecorder to ensure no real DB operations happen
    with patch(
        "codegate.pipeline.workspace.workspace.DbRecorder", autospec=True
    ) as mock_recorder_cls:
        mock_recorder = mock_recorder_cls.return_value
        mock_recorder.add_workspace = AsyncMock()

        # Call the method
        result = await workspace_commands._add_workspace(*args)

        # Assertions
        assert result == expected_message

        # If expected_message indicates "added", we expect add_workspace to be called once
        if "has been added" in expected_message:
            mock_recorder.add_workspace.assert_awaited_once_with(args[0])
        else:
            mock_recorder.add_workspace.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args, workspace_exists, sessions, expected_message",
    [
        # Case 1: No name provided
        ([], False, [], "Please provide a name. Use `codegate-workspace activate workspace_name`"),
        # Case 2: Workspace does not exist
        (
            ["non_existing_ws"],
            False,
            [],
            (
                "Workspace **non_existing_ws** does not exist. "
                "Use `codegate-workspace add non_existing_ws` to add it"
            ),
        ),
        # Case 3: No active session found
        (["myworkspace"], True, [], "Something went wrong. No active session found."),
        # Case 4: Workspace is already active
        (
            ["myworkspace"],
            True,
            [Session(id="1", active_workspace_id="10", last_update=datetime.datetime.now())],
            "Workspace **myworkspace** is already active",
        ),
        # Case 5: Successfully activate new workspace
        (
            ["myworkspace"],
            True,
            [
                # This session has a different active_workspace_id (99), so we can activate 10
                Session(id="1", active_workspace_id="99", last_update=datetime.datetime.now())
            ],
            "Workspace **myworkspace** has been activated",
        ),
    ],
)
async def test_activate_workspace(args, workspace_exists, sessions, expected_message):
    """
    Test _activate_workspace under various conditions:
    - no name provided
    - workspace not found
    - session not found
    - workspace already active
    - successful activation
    """
    workspace_commands = WorkspaceCommands()

    # Mock the DbReader to return either an empty list or a mock workspace
    mock_db_reader = AsyncMock()

    if workspace_exists:
        # We'll pretend we found a workspace: ID = 10
        mock_workspace = Workspace(id="10", name=args[0])
        mock_db_reader.get_workspace_by_name.return_value = [mock_workspace]
    else:
        mock_db_reader.get_workspace_by_name.return_value = []

    # Return the sessions for get_sessions
    mock_db_reader.get_sessions.return_value = sessions

    workspace_commands._db_reader = mock_db_reader

    with patch(
        "codegate.pipeline.workspace.workspace.DbRecorder", autospec=True
    ) as mock_recorder_cls:
        mock_recorder = mock_recorder_cls.return_value
        mock_recorder.update_session = AsyncMock()

        result = await workspace_commands._activate_workspace(*args)

        assert result == expected_message

        # If we expect a successful activation, check that update_session was called
        if "has been activated" in expected_message:
            mock_recorder.update_session.assert_awaited_once()
            updated_session = mock_recorder.update_session.await_args[0][0]
            # Check that active_workspace_id is changed to 10 (our mock workspace ID)
            assert updated_session.active_workspace_id == "10"
            # Check that last_update was set to now
            assert isinstance(updated_session.last_update, datetime.datetime)
        else:
            mock_recorder.update_session.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_message, expected_command, expected_args, mocked_execute_response",
    [
        ("codegate-workspace list", "list", [], "List workspaces output"),
        ("codegate-workspace add myws", "add", ["myws"], "Added workspace"),
        ("codegate-workspace activate myws", "activate", ["myws"], "Activated workspace"),
    ],
)
async def test_parse_execute_cmd(
    user_message, expected_command, expected_args, mocked_execute_response
):
    """
    Test parse_execute_cmd to ensure it parses the user message
    and calls the correct command with the correct args.
    """
    workspace_commands = WorkspaceCommands()

    with patch.object(
        workspace_commands, "execute", return_value=mocked_execute_response
    ) as mock_execute:
        result = await workspace_commands.parse_execute_cmd(user_message)
        assert result == mocked_execute_response

        # Verify 'execute' was called with the expected command and args
        mock_execute.assert_awaited_once_with(expected_command, *expected_args)
