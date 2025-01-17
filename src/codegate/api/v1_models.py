from typing import Any, List, Optional

import pydantic

from codegate.db import models as db_models


class Workspace(pydantic.BaseModel):
    name: str
    is_active: bool


class ActiveWorkspace(Workspace):
    # TODO: use a more specific type for last_updated
    last_updated: Any


class ListWorkspacesResponse(pydantic.BaseModel):
    workspaces: list[Workspace]

    @classmethod
    def from_db_workspaces(
        cls, db_workspaces: List[db_models.WorkspaceActive]
    ) -> "ListWorkspacesResponse":
        return cls(
            workspaces=[
                Workspace(name=ws.name, is_active=ws.active_workspace_id is not None)
                for ws in db_workspaces
            ]
        )


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


class CreateWorkspaceRequest(pydantic.BaseModel):
    name: str


class ActivateWorkspaceRequest(pydantic.BaseModel):
    name: str
