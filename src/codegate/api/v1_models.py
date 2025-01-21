from typing import Any, List, Optional

import pydantic

from codegate.db import models as db_models


class Workspace(pydantic.BaseModel):
    name: str
    is_active: bool


class SystemPrompt(pydantic.BaseModel):
    prompt: str


class ActiveWorkspace(Workspace):
    # TODO: use a more specific type for last_updated
    last_updated: Any


class ListWorkspacesResponse(pydantic.BaseModel):
    workspaces: list[Workspace]

    @classmethod
    def from_db_workspaces_active(
        cls, db_workspaces: List[db_models.WorkspaceActive]
    ) -> "ListWorkspacesResponse":
        return cls(
            workspaces=[
                Workspace(name=ws.name, is_active=ws.active_workspace_id is not None)
                for ws in db_workspaces
            ]
        )

    @classmethod
    def from_db_workspaces(
        cls, db_workspaces: List[db_models.Workspace]
    ) -> "ListWorkspacesResponse":
        return cls(workspaces=[Workspace(name=ws.name, is_active=False) for ws in db_workspaces])


class ListActiveWorkspacesResponse(pydantic.BaseModel):
    workspaces: list[ActiveWorkspace]

    @classmethod
    def from_db_workspaces(
        cls, ws: Optional[db_models.ActiveWorkspace]
    ) -> "ListActiveWorkspacesResponse":
        if ws is None:
            return cls(workspaces=[])
        return cls(
            workspaces=[ActiveWorkspace(name=ws.name, is_active=True, last_updated=ws.last_update)]
        )


class CreateOrRenameWorkspaceRequest(pydantic.BaseModel):
    name: str

    # If set, rename the workspace to this name. Note that
    # the 'name' field is still required and the workspace
    # workspace must exist.
    rename_to: Optional[str] = None


class ActivateWorkspaceRequest(pydantic.BaseModel):
    name: str
