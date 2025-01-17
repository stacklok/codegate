from fastapi import APIRouter, Response
from fastapi.exceptions import HTTPException
from fastapi.routing import APIRoute
from pydantic import ValidationError

from codegate.api import v1_models
from codegate.db.connection import AlreadyExistsError
from codegate.workspaces.crud import WorkspaceCrud

v1 = APIRouter()
wscrud = WorkspaceCrud()


def uniq_name(route: APIRoute):
    return f"v1_{route.name}"


@v1.get("/workspaces", tags=["Workspaces"], generate_unique_id_function=uniq_name)
async def list_workspaces() -> v1_models.ListWorkspacesResponse:
    """List all workspaces."""
    wslist = await wscrud.get_workspaces()

    resp = v1_models.ListWorkspacesResponse.from_db_workspaces(wslist)

    return resp


@v1.get("/workspaces/active", tags=["Workspaces"], generate_unique_id_function=uniq_name)
async def list_active_workspaces() -> v1_models.ListActiveWorkspacesResponse:
    """List all active workspaces.

    In it's current form, this function will only return one workspace. That is,
    the globally active workspace."""
    activews = await wscrud.get_active_workspace()

    resp = v1_models.ListActiveWorkspacesResponse.from_db_workspaces(activews)

    return resp


@v1.post("/workspaces/active", tags=["Workspaces"], generate_unique_id_function=uniq_name)
async def activate_workspace(request: v1_models.ActivateWorkspaceRequest, status_code=204):
    """Activate a workspace by name."""
    activated = await wscrud.activate_workspace(request.name)

    # TODO: Refactor
    if not activated:
        return HTTPException(status_code=409, detail="Workspace already active")

    return Response(status_code=204)


@v1.post("/workspaces", tags=["Workspaces"], generate_unique_id_function=uniq_name, status_code=201)
async def create_workspace(request: v1_models.CreateWorkspaceRequest):
    """Create a new workspace."""
    # Input validation is done in the model
    try:
        created = await wscrud.add_workspace(request.name)
    except AlreadyExistsError:
        raise HTTPException(status_code=409, detail="Workspace already exists")
    except ValidationError:
        raise HTTPException(status_code=400,
                            detail=("Invalid workspace name. "
                                    "Please use only alphanumeric characters and dashes"))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

    if created:
        return v1_models.Workspace(name=created.name)


@v1.delete(
    "/workspaces/{workspace_name}",
    tags=["Workspaces"],
    generate_unique_id_function=uniq_name,
    status_code=204,
)
async def delete_workspace(workspace_name: str):
    """Delete a workspace by name."""
    raise NotImplementedError
