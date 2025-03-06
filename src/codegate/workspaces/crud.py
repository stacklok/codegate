import datetime
from typing import List, Optional, Tuple
from uuid import uuid4 as uuid

import structlog

from codegate.db import models as db_models
from codegate.db.connection import AlreadyExistsError, DbReader, DbRecorder, DbTransaction
from codegate.muxing import models as mux_models
from codegate.muxing import rulematcher

logger = structlog.get_logger("codegate")


class WorkspaceCrudError(Exception):
    pass


class WorkspaceDoesNotExistError(WorkspaceCrudError):
    pass


class WorkspaceNameAlreadyInUseError(WorkspaceCrudError):
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
        self._db_recorder = DbRecorder()

    async def add_workspace(
        self,
        new_workspace_name: str,
        custom_instructions: Optional[str] = None,
        muxing_rules: Optional[List[mux_models.MuxRuleWithProviderId]] = None,
    ) -> Tuple[db_models.WorkspaceRow, List[db_models.MuxRule]]:
        """
        Add a workspace

        Args:
            new_workspace_name (str): The name of the workspace
            system_prompt (Optional[str]): The system prompt for the workspace
            muxing_rules (Optional[List[mux_models.MuxRuleWithProviderId]]): The muxing rules for the workspace
        """
        if new_workspace_name == "":
            raise WorkspaceCrudError("Workspace name cannot be empty.")
        if new_workspace_name in RESERVED_WORKSPACE_KEYWORDS:
            raise WorkspaceCrudError(f"Workspace name {new_workspace_name} is reserved.")

        async with DbTransaction() as transaction:
            try:
                existing_ws = await self._db_reader.get_workspace_by_name(new_workspace_name)
                if existing_ws:
                    raise WorkspaceNameAlreadyInUseError(
                        f"Workspace name {new_workspace_name} is already in use."
                    )

                workspace_created = await self._db_recorder.add_workspace(new_workspace_name)

                if custom_instructions:
                    workspace_created.custom_instructions = custom_instructions
                    await self._db_recorder.update_workspace(workspace_created)

                mux_rules = []
                if muxing_rules:
                    mux_rules = await self.set_muxes(new_workspace_name, muxing_rules)

                await transaction.commit()
                return workspace_created, mux_rules
            except (
                AlreadyExistsError,
                WorkspaceDoesNotExistError,
                WorkspaceNameAlreadyInUseError,
            ) as e:
                raise e
            except Exception as e:
                raise WorkspaceCrudError(f"Error adding workspace {new_workspace_name}: {str(e)}")

    async def update_workspace(
        self,
        old_workspace_name: str,
        new_workspace_name: str,
        custom_instructions: Optional[str] = None,
        muxing_rules: Optional[List[mux_models.MuxRuleWithProviderId]] = None,
    ) -> Tuple[db_models.WorkspaceRow, List[db_models.MuxRule]]:
        """
        Update a workspace

        Args:
            old_workspace_name (str): The old name of the workspace
            new_workspace_name (str): The new name of the workspace
            system_prompt (Optional[str]): The system prompt for the workspace
            muxing_rules (Optional[List[mux_models.MuxRuleWithProviderId]]): The muxing rules for the workspace
        """  # noqa: E501
        if new_workspace_name == "":
            raise WorkspaceCrudError("Workspace name cannot be empty.")
        if old_workspace_name == "":
            raise WorkspaceCrudError("Workspace name cannot be empty.")
        if old_workspace_name in DEFAULT_WORKSPACE_NAME:
            raise WorkspaceCrudError("Cannot rename default workspace.")
        if new_workspace_name in RESERVED_WORKSPACE_KEYWORDS:
            raise WorkspaceCrudError(f"Workspace name {new_workspace_name} is reserved.")

        async with DbTransaction() as transaction:
            try:
                ws = await self._db_reader.get_workspace_by_name(old_workspace_name)
                if not ws:
                    raise WorkspaceDoesNotExistError(
                        f"Workspace {old_workspace_name} does not exist."
                    )

                if old_workspace_name != new_workspace_name:
                    existing_ws = await self._db_reader.get_workspace_by_name(new_workspace_name)
                    if existing_ws:
                        raise WorkspaceNameAlreadyInUseError(
                            f"Workspace name {new_workspace_name} is already in use."
                        )

                new_ws = db_models.WorkspaceRow(
                    id=ws.id, name=new_workspace_name, custom_instructions=ws.custom_instructions
                )
                workspace_renamed = await self._db_recorder.update_workspace(new_ws)

                if custom_instructions:
                    workspace_renamed.custom_instructions = custom_instructions
                    await self._db_recorder.update_workspace(workspace_renamed)

                mux_rules = []
                if muxing_rules:
                    mux_rules = await self.set_muxes(new_workspace_name, muxing_rules)

                await transaction.commit()
                return workspace_renamed, mux_rules
            except WorkspaceDoesNotExistError as e:
                raise e
            except Exception as e:
                raise WorkspaceCrudError(f"Error updating workspace {old_workspace_name}: {str(e)}")

    async def get_workspaces(self) -> List[db_models.WorkspaceWithSessionInfo]:
        """
        Get all workspaces
        """
        return await self._db_reader.get_workspaces()

    async def get_archived_workspaces(self) -> List[db_models.WorkspaceRow]:
        """
        Get all archived workspaces
        """
        return await self._db_reader.get_archived_workspaces()

    async def get_active_workspace(self) -> Optional[db_models.ActiveWorkspace]:
        """
        Get the active workspace
        """
        return await self._db_reader.get_active_workspace()

    async def _is_workspace_active(
        self, workspace_name: str
    ) -> Tuple[bool, Optional[db_models.Session], Optional[db_models.WorkspaceRow]]:
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
        await self._db_recorder.update_session(session)

        # Ensure the mux registry is updated
        mux_registry = await rulematcher.get_muxing_rules_registry()
        await mux_registry.set_active_workspace(workspace.name)
        return

    async def recover_workspace(self, workspace_name: str):
        """
        Recover an archived workspace
        """
        selected_workspace = await self._db_reader.get_archived_workspace_by_name(workspace_name)
        if not selected_workspace:
            raise WorkspaceDoesNotExistError(f"Workspace {workspace_name} does not exist.")

        await self._db_recorder.recover_workspace(selected_workspace)
        return

    async def update_workspace_custom_instructions(
        self, workspace_name: str, custom_instr_lst: List[str]
    ) -> db_models.WorkspaceRow:
        selected_workspace = await self._db_reader.get_workspace_by_name(workspace_name)
        if not selected_workspace:
            raise WorkspaceDoesNotExistError(f"Workspace {workspace_name} does not exist.")

        custom_instructions = " ".join(custom_instr_lst)
        workspace_update = db_models.WorkspaceRow(
            id=selected_workspace.id,
            name=selected_workspace.name,
            custom_instructions=custom_instructions,
        )
        updated_workspace = await self._db_recorder.update_workspace(workspace_update)
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

        try:
            _ = await self._db_recorder.soft_delete_workspace(selected_workspace)
        except Exception:
            raise WorkspaceCrudError(f"Error deleting workspace {workspace_name}")

        # Remove the muxes from the registry
        mux_registry = await rulematcher.get_muxing_rules_registry()
        await mux_registry.delete_ws_rules(workspace_name)
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

        try:
            _ = await self._db_recorder.hard_delete_workspace(selected_workspace)
        except Exception:
            raise WorkspaceCrudError(f"Error deleting workspace {workspace_name}")
        return

    async def get_workspace_by_name(self, workspace_name: str) -> db_models.WorkspaceRow:
        workspace = await self._db_reader.get_workspace_by_name(workspace_name)
        if not workspace:
            raise WorkspaceDoesNotExistError(f"Workspace {workspace_name} does not exist.")
        return workspace

    async def workspaces_by_provider(
        self, provider_id: uuid
    ) -> List[db_models.WorkspaceWithSessionInfo]:
        """Get the workspaces by provider."""

        workspaces = await self._db_reader.get_workspaces_by_provider(str(provider_id))

        return workspaces

    async def get_muxes(self, workspace_name: str) -> List[mux_models.MuxRuleWithProviderId]:
        # Verify if workspace exists
        workspace = await self._db_reader.get_workspace_by_name(workspace_name)
        if not workspace:
            raise WorkspaceDoesNotExistError(f"Workspace {workspace_name} does not exist.")

        dbmuxes = await self._db_reader.get_muxes_by_workspace(workspace.id)

        muxes = []
        # These are already sorted by priority
        for dbmux in dbmuxes:
            muxes.append(
                mux_models.MuxRuleWithProviderId(
                    provider_name=dbmux.provider_endpoint_name,
                    provider_type=dbmux.provider_endpoint_type,
                    provider_id=dbmux.provider_endpoint_id,
                    model=dbmux.provider_model_name,
                    matcher_type=dbmux.matcher_type,
                    matcher=dbmux.matcher_blob,
                )
            )

        return muxes

    async def set_muxes(
        self, workspace_name: str, muxes: List[mux_models.MuxRuleWithProviderId]
    ) -> List[db_models.MuxRule]:
        # Verify if workspace exists
        workspace = await self._db_reader.get_workspace_by_name(workspace_name)
        if not workspace:
            raise WorkspaceDoesNotExistError(f"Workspace {workspace_name} does not exist.")

        # Delete all muxes for the workspace
        await self._db_recorder.delete_muxes_by_workspace(workspace.id)

        # Add the new muxes
        priority = 0

        muxes_with_routes: List[
            Tuple[mux_models.MuxRuleWithProviderId, rulematcher.ModelRoute]
        ] = []

        # Verify all models are valid
        for mux in muxes:
            route = await self.get_routing_for_mux(mux)
            muxes_with_routes.append((mux, route))

        matchers: List[rulematcher.MuxingRuleMatcher] = []
        dbmuxes: List[db_models.MuxRule] = []

        for mux, route in muxes_with_routes:
            new_mux = db_models.MuxRule(
                id=str(uuid()),
                provider_endpoint_id=mux.provider_id,
                provider_endpoint_type=mux.provider_type,
                provider_endpoint_name=mux.provider_name,
                provider_model_name=mux.model,
                workspace_id=workspace.id,
                matcher_type=mux.matcher_type,
                matcher_blob=mux.matcher if mux.matcher else "",
                priority=priority,
            )
            dbmux = await self._db_recorder.add_mux(new_mux)
            dbmuxes.append(dbmux)

            matchers.append(rulematcher.MuxingMatcherFactory.create(dbmux, route))

            priority += 1

        # Set routing list for the workspace
        mux_registry = await rulematcher.get_muxing_rules_registry()
        await mux_registry.set_ws_rules(workspace_name, matchers)

        return dbmuxes

    async def get_routing_for_mux(
        self, mux: mux_models.MuxRuleWithProviderId
    ) -> rulematcher.ModelRoute:
        """Get the routing for a mux

        Note that this particular mux object is the API model, not the database model.
        It's only not annotated because of a circular import issue.
        """
        dbprov = await self._db_reader.get_provider_endpoint_by_id(mux.provider_id)
        if not dbprov:
            raise WorkspaceCrudError(f'Provider "{mux.provider_name}" does not exist')

        dbm = await self._db_reader.get_provider_model_by_provider_id_and_name(
            mux.provider_id,
            mux.model,
        )
        if not dbm:
            raise WorkspaceCrudError(
                f'Model "{mux.model}" does not exist for provider "{mux.provider_name}"'
            )
        dbauth = await self._db_reader.get_auth_material_by_provider_id(mux.provider_id)
        if not dbauth:
            raise WorkspaceCrudError(
                f'Auth material for provider "{mux.provider_name}" does not exist'
            )

        return rulematcher.ModelRoute(
            endpoint=dbprov,
            model=dbm,
            auth_material=dbauth,
        )

    async def get_routing_for_db_mux(self, mux: db_models.MuxRule) -> rulematcher.ModelRoute:
        """Get the routing for a mux

        Note that this particular mux object is the database model, not the API model.
        It's only not annotated because of a circular import issue.
        """
        dbprov = await self._db_reader.get_provider_endpoint_by_id(mux.provider_endpoint_id)
        if not dbprov:
            raise WorkspaceCrudError(f'Provider "{mux.provider_endpoint_name}" does not exist')

        dbm = await self._db_reader.get_provider_model_by_provider_id_and_name(
            mux.provider_endpoint_id,
            mux.provider_model_name,
        )
        if not dbm:
            raise WorkspaceCrudError(
                f"Model {mux.provider_model_name} does not "
                "exist for provider {mux.provider_endpoint_id}"
            )
        dbauth = await self._db_reader.get_auth_material_by_provider_id(mux.provider_endpoint_id)
        if not dbauth:
            raise WorkspaceCrudError(
                f'Auth material for provider "{mux.provider_endpoint_name}" does not exist'
            )

        return rulematcher.ModelRoute(
            model=dbm,
            endpoint=dbprov,
            auth_material=dbauth,
        )

    async def initialize_mux_registry(self) -> None:
        """Initialize the mux registry with all workspaces in the database"""

        active_ws = await self.get_active_workspace()
        if active_ws:
            mux_registry = await rulematcher.get_muxing_rules_registry()
            await mux_registry.set_active_workspace(active_ws.name)

        await self.repopulate_mux_cache()

    async def repopulate_mux_cache(self) -> None:
        """Repopulate the mux cache with all muxes in the database"""

        # Get all workspaces
        workspaces = await self.get_workspaces()

        mux_registry = await rulematcher.get_muxing_rules_registry()

        # Remove any workspaces from cache that are not in the database
        ws_names = set(ws.name for ws in workspaces)
        cached_ws = set(await mux_registry.get_registries())
        ws_to_remove = cached_ws - ws_names
        for ws in ws_to_remove:
            await mux_registry.delete_ws_rules(ws)

        # For each workspace, get the muxes and set them in the registry
        for ws in workspaces:
            muxes = await self._db_reader.get_muxes_by_workspace(ws.id)

            matchers: List[rulematcher.MuxingRuleMatcher] = []

            for mux in muxes:
                route = await self.get_routing_for_db_mux(mux)
                matchers.append(rulematcher.MuxingMatcherFactory.create(mux, route))

            await mux_registry.set_ws_rules(ws.name, matchers)
