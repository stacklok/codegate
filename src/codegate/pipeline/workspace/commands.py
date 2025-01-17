import datetime
from typing import List, Optional, Tuple

from codegate.db.connection import DbReader, DbRecorder
from codegate.db.models import ActiveWorkspace, Session, Workspace, WorkspaceActive


class WorkspaceCrud:

    def __init__(self):
        self._db_reader = DbReader()

    async def add_workspace(self, new_workspace_name: str) -> bool:
        """
        Add a workspace

        Args:
            name (str): The name of the workspace
        """
        db_recorder = DbRecorder()
        workspace_created = await db_recorder.add_workspace(
            new_workspace_name)
        return bool(workspace_created)

    async def get_workspaces(self) -> List[WorkspaceActive]:
        """
        Get all workspaces
        """
        return await self._db_reader.get_workspaces()

    async def get_active_workspace(self) -> Optional[ActiveWorkspace]:
        """
        Get the active workspace
        """
        return await self._db_reader.get_active_workspace()

    async def _is_workspace_active_or_not_exist(
        self, workspace_name: str
    ) -> Tuple[bool, Optional[Session], Optional[Workspace]]:
        """
        Check if the workspace is active

        Will return:
        - True if the workspace was activated
        - False if the workspace is already active or does not exist
        """
        selected_workspace = await self._db_reader.get_workspace_by_name(workspace_name)
        if not selected_workspace:
            return True, None, None

        sessions = await self._db_reader.get_sessions()
        # The current implementation expects only one active session
        if len(sessions) != 1:
            raise RuntimeError("Something went wrong. No active session found.")

        session = sessions[0]
        if session.active_workspace_id == selected_workspace.id:
            return True, None, None
        return False, session, selected_workspace

    async def activate_workspace(self, workspace_name: str) -> bool:
        """
        Activate a workspace

        Will return:
        - True if the workspace was activated
        - False if the workspace is already active or does not exist
        """
        is_active, session, workspace = await self._is_workspace_active_or_not_exist(workspace_name)
        if is_active:
            return False

        session.active_workspace_id = workspace.id
        session.last_update = datetime.datetime.now(datetime.timezone.utc)
        db_recorder = DbRecorder()
        await db_recorder.update_session(session)
        return True


class WorkspaceCommands:

    def __init__(self):
        self.workspace_crud = WorkspaceCrud()
        self.commands = {
            "list": self._list_workspaces,
            "add": self._add_workspace,
            "activate": self._activate_workspace,
        }

    async def _list_workspaces(self, *args) -> str:
        """
        List all workspaces
        """
        workspaces = await self.workspace_crud.get_workspaces()
        respond_str = ""
        for workspace in workspaces:
            respond_str += f"- {workspace.name}"
            if workspace.active_workspace_id:
                respond_str += " **(active)**"
            respond_str += "\n"
        return respond_str

    async def _add_workspace(self, *args) -> str:
        """
        Add a workspace
        """
        if args is None or len(args) == 0:
            return "Please provide a name. Use `codegate-workspace add your_workspace_name`"

        new_workspace_name = args[0]
        if not new_workspace_name:
            return "Please provide a name. Use `codegate-workspace add your_workspace_name`"

        workspace_created = await self.workspace_crud.add_workspace(new_workspace_name)
        if not workspace_created:
            return (
                "Something went wrong. Workspace could not be added.\n"
                "1. Check if the name is alphanumeric and only contains dashes, and underscores.\n"
                "2. Check if the workspace already exists."
            )
        return f"Workspace **{new_workspace_name}** has been added"

    async def _activate_workspace(self, *args) -> str:
        """
        Activate a workspace
        """
        if args is None or len(args) == 0:
            return "Please provide a name. Use `codegate-workspace activate workspace_name`"

        workspace_name = args[0]
        if not workspace_name:
            return "Please provide a name. Use `codegate-workspace activate workspace_name`"

        was_activated = await self.workspace_crud.activate_workspace(workspace_name)
        if not was_activated:
            return (
                f"Workspace **{workspace_name}** does not exist or was already active. "
                f"Use `codegate-workspace add {workspace_name}` to add it"
            )
        return f"Workspace **{workspace_name}** has been activated"

    async def execute(self, command: str, *args) -> str:
        """
        Execute the given command

        Args:
            command (str): The command to execute
        """
        command_to_execute = self.commands.get(command)
        if command_to_execute is not None:
            return await command_to_execute(*args)
        else:
            return "Command not found"

    async def parse_execute_cmd(self, last_user_message: str) -> str:
        """
        Parse the last user message and execute the command

        Args:
            last_user_message (str): The last user message
        """
        command_and_args = last_user_message.lower().split("codegate-workspace ")[1]
        command, *args = command_and_args.split(" ")
        return await self.execute(command, *args)
