import datetime
from typing import List, Optional, Tuple

from codegate.db.connection import DbReader, DbRecorder
from codegate.db.models import ActiveWorkspace, Session, Workspace, WorkspaceActive


class WorkspaceCrudError(Exception):
    pass


class WorkspaceDoesNotExistError(WorkspaceCrudError):
    pass


class WorkspaceAlreadyActiveError(WorkspaceCrudError):
    pass


class WorkspaceCrud:

    def __init__(self):
        self._db_reader = DbReader()

    async def add_workspace(self, new_workspace_name: str) -> Workspace:
        """
        Add a workspace

        Args:
            name (str): The name of the workspace
        """
        db_recorder = DbRecorder()
        workspace_created = await db_recorder.add_workspace(new_workspace_name)
        return workspace_created

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

    async def _is_workspace_active(
        self, workspace_name: str
    ) -> Tuple[bool, Optional[Session], Optional[Workspace]]:
        """
        Check if the workspace is active alongside the session and workspace objects
        """
        # TODO: All of this should be done within a transaction.

        selected_workspace = await self._db_reader.get_workspace_by_name(workspace_name)
        if not selected_workspace:
            raise WorkspaceDoesNotExistError(f"Workspace {workspace_name} does not exist.")

        sessions = await self._db_reader.get_sessions()
        # The current implementation expects only one active session
        if len(sessions) != 1:
            raise WorkspaceCrudError("Something went wrong. More than one session found.")

        session = sessions[0]
        return (session.active_workspace_id == selected_workspace.id, session, selected_workspace)

    async def activate_workspace(self, workspace_name: str):
        """
        Activate a workspace
        """
        is_active, session, workspace = await self._is_workspace_active(workspace_name)
        if is_active:
            raise WorkspaceAlreadyActiveError(f"Workspace {workspace_name} is already active.")

        session.active_workspace_id = workspace.id
        session.last_update = datetime.datetime.now(datetime.timezone.utc)
        db_recorder = DbRecorder()
        await db_recorder.update_session(session)
        return

    async def update_workspace_system_prompt(
        self, workspace_name: str, sys_prompt_lst: List[str]
    ) -> Workspace:
        selected_workspace = await self._db_reader.get_workspace_by_name(workspace_name)
        if not selected_workspace:
            raise WorkspaceDoesNotExistError(f"Workspace {workspace_name} does not exist.")

        system_prompt = " ".join(sys_prompt_lst)
        workspace_update = Workspace(
            id=selected_workspace.id,
            name=selected_workspace.name,
            system_prompt=system_prompt,
        )
        db_recorder = DbRecorder()
        updated_workspace = await db_recorder.update_workspace(workspace_update)
        return updated_workspace

    async def soft_delete_workspace(self, workspace_name: str):
        """
        Soft delete a workspace
        """
        if workspace_name == "":
            raise WorkspaceCrudError("Workspace name cannot be empty.")
        if workspace_name == "default":
            raise WorkspaceCrudError("Cannot delete default workspace.")

        selected_workspace = await self._db_reader.get_workspace_by_name(workspace_name)
        if not selected_workspace:
            raise WorkspaceDoesNotExistError(f"Workspace {workspace_name} does not exist.")

        # Check if workspace is active, if it is, make the default workspace active
        active_workspace = await self._db_reader.get_active_workspace()
        if active_workspace and active_workspace.id == selected_workspace.id:
            _ = await self.activate_workspace("default")

        db_recorder = DbRecorder()
        try:
            _ = await db_recorder.soft_delete_workspace(selected_workspace)
        except Exception:
            raise WorkspaceCrudError(f"Error deleting workspace {workspace_name}")
        return

    async def get_workspace_by_name(self, workspace_name: str) -> Workspace:
        workspace = await self._db_reader.get_workspace_by_name(workspace_name)
        if not workspace:
            raise WorkspaceDoesNotExistError(f"Workspace {workspace_name} does not exist.")
        return workspace
