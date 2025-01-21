from typing import List, Optional

from fastapi import APIRouter, HTTPException, Response
from fastapi.routing import APIRoute
from pydantic import ValidationError

from codegate.api import v1_models
from codegate.api.dashboard import dashboard
from codegate.api.dashboard.request_models import AlertConversation, Conversation
from codegate.db.connection import AlreadyExistsError, DbReader
from codegate.workspaces import crud

v1 = APIRouter()
v1.include_router(dashboard.dashboard_router)
wscrud = crud.WorkspaceCrud()

# This is a singleton object
dbreader = DbReader()


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
    try:
        await wscrud.activate_workspace(request.name)
    except crud.WorkspaceAlreadyActiveError:
        raise HTTPException(status_code=409, detail="Workspace already active")
    except crud.WorkspaceDoesNotExistError:
        raise HTTPException(status_code=404, detail="Workspace does not exist")
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

    return Response(status_code=204)


@v1.post("/workspaces", tags=["Workspaces"], generate_unique_id_function=uniq_name, status_code=201)
async def create_workspace(request: v1_models.CreateWorkspaceRequest) -> v1_models.Workspace:
    """Create a new workspace."""
    # Input validation is done in the model
    try:
        _ = await wscrud.add_workspace(request.name)
    except AlreadyExistsError:
        raise HTTPException(status_code=409, detail="Workspace already exists")
    except ValidationError:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid workspace name. " "Please use only alphanumeric characters and dashes"
            ),
        )
    except crud.WorkspaceCrudError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

    return v1_models.Workspace(name=request.name, is_active=False)


@v1.delete(
    "/workspaces/{workspace_name}",
    tags=["Workspaces"],
    generate_unique_id_function=uniq_name,
)
async def delete_workspace(workspace_name: str):
    """Delete a workspace by name."""
    try:
        _ = await wscrud.soft_delete_workspace(workspace_name)
    except crud.WorkspaceDoesNotExistError:
        raise HTTPException(status_code=404, detail="Workspace does not exist")
    except crud.WorkspaceCrudError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

    return Response(status_code=204)


@v1.get(
    "/workspaces/{workspace_name}/alerts",
    tags=["Workspaces"],
    generate_unique_id_function=uniq_name,
)
async def get_workspace_alerts(workspace_name: str) -> List[Optional[AlertConversation]]:
    """Get alerts for a workspace."""
    try:
        ws = await wscrud.get_workspace_by_name(workspace_name)
    except crud.WorkspaceDoesNotExistError:
        raise HTTPException(status_code=404, detail="Workspace does not exist")
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

    try:
        alerts = await dbreader.get_alerts_with_prompt_and_output(ws.id)
        return await dashboard.parse_get_alert_conversation(alerts)
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")


@v1.get(
    "/workspaces/{workspace_name}/messages",
    tags=["Workspaces"],
    generate_unique_id_function=uniq_name,
)
async def get_workspace_messages(workspace_name: str) -> List[Conversation]:
    """Get messages for a workspace."""
    try:
        ws = await wscrud.get_workspace_by_name(workspace_name)
    except crud.WorkspaceDoesNotExistError:
        raise HTTPException(status_code=404, detail="Workspace does not exist")
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

    try:
        prompts_outputs = await dbreader.get_prompts_with_output(ws.id)
        return await dashboard.parse_messages_in_conversations(prompts_outputs)
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")


@v1.get(
    "/workspaces/{workspace_name}/system-prompt",
    tags=["Workspaces"],
    generate_unique_id_function=uniq_name,
)
async def get_workspace_system_prompt(workspace_name: str) -> v1_models.SystemPrompt:
    """Get the system prompt for a workspace."""
    try:
        ws = await wscrud.get_workspace_by_name(workspace_name)
    except crud.WorkspaceDoesNotExistError:
        raise HTTPException(status_code=404, detail="Workspace does not exist")
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

    if ws.system_prompt is None:
        return v1_models.SystemPrompt(prompt="")

    return v1_models.SystemPrompt(prompt=ws.system_prompt)


@v1.put(
    "/workspaces/{workspace_name}/system-prompt",
    tags=["Workspaces"],
    generate_unique_id_function=uniq_name,
    status_code=204,
)
async def set_workspace_system_prompt(workspace_name: str, request: v1_models.SystemPrompt):
    try:
        # This already checks if the workspace exists
        await wscrud.update_workspace_system_prompt(workspace_name, [request.prompt])
    except crud.WorkspaceDoesNotExistError:
        raise HTTPException(status_code=404, detail="Workspace does not exist")
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

    return Response(status_code=204)


@v1.delete(
    "/workspaces/{workspace_name}/system-prompt",
    tags=["Workspaces"],
    generate_unique_id_function=uniq_name,
    status_code=204,
)
async def delete_workspace_system_prompt(workspace_name: str):
    try:
        # This already checks if the workspace exists
        await wscrud.update_workspace_system_prompt(workspace_name, [])
    except crud.WorkspaceDoesNotExistError:
        raise HTTPException(status_code=404, detail="Workspace does not exist")
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

    return Response(status_code=204)
