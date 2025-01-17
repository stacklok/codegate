import datetime
from typing import List, Optional, Tuple

from codegate.db.connection import DbReader, DbRecorder
from codegate.db.models import ActiveWorkspace, Session, Workspace, WorkspaceActive


class WorkspaceCrudError(Exception):
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

    async def get_workspaces(self)-> List[WorkspaceActive]:
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
