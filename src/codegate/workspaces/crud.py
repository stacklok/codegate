import asyncio
import datetime
from typing import List, Optional, Tuple
from uuid import uuid4 as uuid

from codegate.db.connection import DbReader, DbRecorder
from codegate.db.models import (
    ActiveWorkspace,
    MuxRule,
    MuxRuleProviderEndpoint,
    Session,
    WorkspaceRow,
    WorkspaceWithSessionInfo,
)


class WorkspaceCrudError(Exception):
    pass


class WorkspaceDoesNotExistError(WorkspaceCrudError):
    pass


class WorkspaceAlreadyActiveError(WorkspaceCrudError):
    pass


class WorkspaceMuxRuleDoesNotExistError(WorkspaceCrudError):
    pass


DEFAULT_WORKSPACE_NAME = "default"

# These are reserved keywords that cannot be used for workspaces
RESERVED_WORKSPACE_KEYWORDS = [DEFAULT_WORKSPACE_NAME, "active", "archived"]


class WorkspaceCrud:

    def __init__(self):
        self._db_reader = DbReader()

    async def add_workspace(self, new_workspace_name: str) -> WorkspaceRow:
        """
        Add a workspace

        Args:
            name (str): The name of the workspace
        """
        if new_workspace_name == "":
            raise WorkspaceCrudError("Workspace name cannot be empty.")
        if new_workspace_name in RESERVED_WORKSPACE_KEYWORDS:
            raise WorkspaceCrudError(f"Workspace name {new_workspace_name} is reserved.")
        db_recorder = DbRecorder()
        workspace_created = await db_recorder.add_workspace(new_workspace_name)
        return workspace_created

    async def rename_workspace(
        self, old_workspace_name: str, new_workspace_name: str
    ) -> WorkspaceRow:
        """
        Rename a workspace

        Args:
            old_name (str): The old name of the workspace
            new_name (str): The new name of the workspace
        """
        if new_workspace_name == "":
            raise WorkspaceCrudError("Workspace name cannot be empty.")
        if old_workspace_name == "":
            raise WorkspaceCrudError("Workspace name cannot be empty.")
        if old_workspace_name in DEFAULT_WORKSPACE_NAME:
            raise WorkspaceCrudError("Cannot rename default workspace.")
        if new_workspace_name in RESERVED_WORKSPACE_KEYWORDS:
            raise WorkspaceCrudError(f"Workspace name {new_workspace_name} is reserved.")
        if old_workspace_name == new_workspace_name:
            raise WorkspaceCrudError("Old and new workspace names are the same.")
        ws = await self._db_reader.get_workspace_by_name(old_workspace_name)
        if not ws:
            raise WorkspaceDoesNotExistError(f"Workspace {old_workspace_name} does not exist.")
        db_recorder = DbRecorder()
        new_ws = WorkspaceRow(
            id=ws.id, name=new_workspace_name, custom_instructions=ws.custom_instructions
        )
        workspace_renamed = await db_recorder.update_workspace(new_ws)
        return workspace_renamed

    async def get_workspaces(self) -> List[WorkspaceWithSessionInfo]:
        """
        Get all workspaces
        """
        return await self._db_reader.get_workspaces()

    async def get_archived_workspaces(self) -> List[WorkspaceRow]:
        """
        Get all archived workspaces
        """
        return await self._db_reader.get_archived_workspaces()

    async def get_active_workspace(self) -> Optional[ActiveWorkspace]:
        """
        Get the active workspace
        """
        return await self._db_reader.get_active_workspace()

    async def _is_workspace_active(
        self, workspace_name: str
    ) -> Tuple[bool, Optional[Session], Optional[WorkspaceRow]]:
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

    async def recover_workspace(self, workspace_name: str):
        """
        Recover an archived workspace
        """
        selected_workspace = await self._db_reader.get_archived_workspace_by_name(workspace_name)
        if not selected_workspace:
            raise WorkspaceDoesNotExistError(f"Workspace {workspace_name} does not exist.")

        db_recorder = DbRecorder()
        await db_recorder.recover_workspace(selected_workspace)
        return

    async def update_workspace_custom_instructions(
        self, workspace_name: str, custom_instr_lst: List[str]
    ) -> WorkspaceRow:
        selected_workspace = await self._db_reader.get_workspace_by_name(workspace_name)
        if not selected_workspace:
            raise WorkspaceDoesNotExistError(f"Workspace {workspace_name} does not exist.")

        custom_instructions = " ".join(custom_instr_lst)
        workspace_update = WorkspaceRow(
            id=selected_workspace.id,
            name=selected_workspace.name,
            custom_instructions=custom_instructions,
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
        if workspace_name == DEFAULT_WORKSPACE_NAME:
            raise WorkspaceCrudError("Cannot archive default workspace.")

        selected_workspace = await self._db_reader.get_workspace_by_name(workspace_name)
        if not selected_workspace:
            raise WorkspaceDoesNotExistError(f"Workspace {workspace_name} does not exist.")

        # Check if workspace is active, if it is, make the default workspace active
        active_workspace = await self._db_reader.get_active_workspace()
        if active_workspace and active_workspace.id == selected_workspace.id:
            raise WorkspaceCrudError("Cannot archive active workspace.")

        db_recorder = DbRecorder()
        try:
            _ = await db_recorder.soft_delete_workspace(selected_workspace)
        except Exception:
            raise WorkspaceCrudError(f"Error deleting workspace {workspace_name}")
        return

    async def hard_delete_workspace(self, workspace_name: str):
        """
        Hard delete a workspace
        """
        if workspace_name == "":
            raise WorkspaceCrudError("Workspace name cannot be empty.")

        selected_workspace = await self._db_reader.get_archived_workspace_by_name(workspace_name)
        if not selected_workspace:
            raise WorkspaceDoesNotExistError(f"Workspace {workspace_name} does not exist.")

        db_recorder = DbRecorder()
        try:
            _ = await db_recorder.hard_delete_workspace(selected_workspace)
        except Exception:
            raise WorkspaceCrudError(f"Error deleting workspace {workspace_name}")
        return

    async def get_workspace_by_name(self, workspace_name: str) -> WorkspaceRow:
        workspace = await self._db_reader.get_workspace_by_name(workspace_name)
        if not workspace:
            raise WorkspaceDoesNotExistError(f"Workspace {workspace_name} does not exist.")
        return workspace

    # Can't use type hints since the models are not yet defined
    # Note that I'm explicitly importing the models here to avoid circular imports.
    async def get_muxes(self, workspace_name: str):
        from codegate.api import v1_models

        # Verify if workspace exists
        workspace = await self._db_reader.get_workspace_by_name(workspace_name)
        if not workspace:
            raise WorkspaceDoesNotExistError(f"Workspace {workspace_name} does not exist.")

        dbmuxes = await self._db_reader.get_muxes_by_workspace(workspace.id)

        muxes = []
        # These are already sorted by priority
        for dbmux in dbmuxes:
            muxes.append(
                v1_models.MuxRule(
                    provider_id=dbmux.provider_endpoint_id,
                    model=dbmux.provider_model_name,
                    matcher_type=dbmux.matcher_type,
                    matcher=dbmux.matcher_blob,
                )
            )

        return muxes

    async def get_muxes_with_provider_info(
        self, workspace_name: str
    ) -> List[MuxRuleProviderEndpoint]:

        # Verify if workspace exists
        workspace = await self._db_reader.get_workspace_by_name(workspace_name)
        if not workspace:
            raise WorkspaceDoesNotExistError(f"Workspace {workspace_name} does not exist.")

        try:
            dbmuxes = await self._db_reader.get_muxes_with_provider_by_workspace(workspace.id)
        except Exception:
            raise WorkspaceCrudError(f"Error getting muxes for workspace {workspace_name}")

        return dbmuxes

    # Can't use type hints since the models are not yet defined
    async def set_muxes(self, workspace_name: str, muxes):
        # Verify if workspace exists
        workspace = await self._db_reader.get_workspace_by_name(workspace_name)
        if not workspace:
            raise WorkspaceDoesNotExistError(f"Workspace {workspace_name} does not exist.")

        # Delete all muxes for the workspace
        db_recorder = DbRecorder()
        await db_recorder.delete_muxes_by_workspace(workspace.id)

        tasks = set()

        # Add the new muxes
        priority = 0

        # Verify all models are valid
        for mux in muxes:
            dbm = await self._db_reader.get_provider_model_by_provider_id_and_name(
                mux.provider_id,
                mux.model,
            )
            if not dbm:
                raise WorkspaceCrudError(
                    f"Model {mux.model} does not exist for provider {mux.provider_id}"
                )

        for mux in muxes:
            new_mux = MuxRule(
                id=str(uuid()),
                provider_endpoint_id=mux.provider_id,
                provider_model_name=mux.model,
                workspace_id=workspace.id,
                matcher_type=mux.matcher_type,
                matcher_blob=mux.matcher if mux.matcher else "",
                priority=priority,
            )
            tasks.add(db_recorder.add_mux(new_mux))

            priority += 1

        await asyncio.gather(*tasks)

    async def get_active_workspace_muxes(self) -> List[MuxRuleProviderEndpoint]:
        active_workspace = await self.get_active_workspace()
        if not active_workspace:
            raise WorkspaceCrudError("No active workspace found.")
        return await self.get_muxes_with_provider_info(active_workspace.name)
