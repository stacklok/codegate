from typing import List, Optional

import requests
import structlog
from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute
from pydantic import ValidationError

from codegate import __version__
from codegate.api import v1_models, v1_processing
from codegate.db.connection import AlreadyExistsError, DbReader
from codegate.providers import crud as provendcrud
from codegate.workspaces import crud

logger = structlog.get_logger("codegate")

v1 = APIRouter()
wscrud = crud.WorkspaceCrud()

# This is a singleton object
dbreader = DbReader()


def uniq_name(route: APIRoute):
    return f"v1_{route.name}"


@v1.get("/provider-endpoints", tags=["Providers"], generate_unique_id_function=uniq_name)
async def list_provider_endpoints(
    name: Optional[str] = None,
    pcrud: provendcrud.ProviderCrud = Depends(provendcrud.ProviderCrud),
) -> List[v1_models.ProviderEndpoint]:
    """List all provider endpoints."""
    try:
        return pcrud.list_endpoints()
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")


@v1.get(
    "/provider-endpoints/{provider_id}", tags=["Providers"], generate_unique_id_function=uniq_name
)
async def get_provider_endpoint(
    provider_id: int,
    pcrud: provendcrud.ProviderCrud = Depends(provendcrud.ProviderCrud),
) -> v1_models.ProviderEndpoint:
    """Get a provider endpoint by ID."""
    # NOTE: This is a dummy implementation. In the future, we should have a proper
    # implementation that fetches the provider endpoint from the database.
    return v1_models.ProviderEndpoint(
        id=provider_id,
        name="dummy",
        description="Dummy provider endpoint",
        endpoint="http://example.com",
        provider_type=v1_models.ProviderType.openai,
        auth_type=v1_models.ProviderAuthType.none,
    )


@v1.post(
    "/provider-endpoints",
    tags=["Providers"],
    generate_unique_id_function=uniq_name,
    status_code=201,
)
async def add_provider_endpoint(
    request: v1_models.ProviderEndpoint,
    pcrud: provendcrud.ProviderCrud = Depends(provendcrud.ProviderCrud),
) -> v1_models.ProviderEndpoint:
    """Add a provider endpoint."""
    # NOTE: This is a dummy implementation. In the future, we should have a proper
    # implementation that adds the provider endpoint to the database.
    return request


@v1.put(
    "/provider-endpoints/{provider_id}", tags=["Providers"], generate_unique_id_function=uniq_name
)
async def update_provider_endpoint(
    provider_id: int,
    request: v1_models.ProviderEndpoint,
    pcrud: provendcrud.ProviderCrud = Depends(provendcrud.ProviderCrud),
) -> v1_models.ProviderEndpoint:
    """Update a provider endpoint by ID."""
    # NOTE: This is a dummy implementation. In the future, we should have a proper
    # implementation that updates the provider endpoint in the database.
    return request


@v1.delete(
    "/provider-endpoints/{provider_id}", tags=["Providers"], generate_unique_id_function=uniq_name
)
async def delete_provider_endpoint(
    provider_id: int,
    pcrud: provendcrud.ProviderCrud = Depends(provendcrud.ProviderCrud),
):
    """Delete a provider endpoint by id."""
    # NOTE: This is a dummy implementation. In the future, we should have a proper
    # implementation that deletes the provider endpoint from the database.
    return Response(status_code=204)


@v1.get(
    "/provider-endpoints/{provider_name}/models",
    tags=["Providers"],
    generate_unique_id_function=uniq_name,
)
async def list_models_by_provider(
    provider_name: str,
    pcrud: provendcrud.ProviderCrud = Depends(provendcrud.ProviderCrud),
) -> List[v1_models.ModelByProvider]:
    """List models by provider."""
    # NOTE: This is a dummy implementation. In the future, we should have a proper
    # implementation that fetches the models by provider from the database.
    return [v1_models.ModelByProvider(name="dummy", provider="dummy")]


@v1.get(
    "/provider-endpoints/models",
    tags=["Providers"],
    generate_unique_id_function=uniq_name,
)
async def list_all_models_for_all_providers(
    pcrud: provendcrud.ProviderCrud = Depends(provendcrud.ProviderCrud),
) -> List[v1_models.ModelByProvider]:
    """List all models for all providers."""
    # NOTE: This is a dummy implementation. In the future, we should have a proper
    # implementation that fetches all the models for all providers from the database.
    return [v1_models.ModelByProvider(name="dummy", provider="dummy")]


@v1.get("/workspaces", tags=["Workspaces"], generate_unique_id_function=uniq_name)
async def list_workspaces() -> v1_models.ListWorkspacesResponse:
    """List all workspaces."""
    wslist = await wscrud.get_workspaces()

    resp = v1_models.ListWorkspacesResponse.from_db_workspaces_with_sessioninfo(wslist)

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
async def create_workspace(
    request: v1_models.CreateOrRenameWorkspaceRequest,
) -> v1_models.Workspace:
    """Create a new workspace."""
    if request.rename_to is not None:
        return await rename_workspace(request)
    return await create_new_workspace(request)


async def create_new_workspace(
    request: v1_models.CreateOrRenameWorkspaceRequest,
) -> v1_models.Workspace:
    # Input validation is done in the model
    try:
        _ = await wscrud.add_workspace(request.name)
    except AlreadyExistsError:
        raise HTTPException(status_code=409, detail="Workspace already exists")
    except ValidationError:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid workspace name. "
                "Please use only alphanumeric characters, hyphens, or underscores."
            ),
        )
    except crud.WorkspaceCrudError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

    return v1_models.Workspace(name=request.name, is_active=False)


async def rename_workspace(
    request: v1_models.CreateOrRenameWorkspaceRequest,
) -> v1_models.Workspace:
    try:
        _ = await wscrud.rename_workspace(request.name, request.rename_to)
    except crud.WorkspaceDoesNotExistError:
        raise HTTPException(status_code=404, detail="Workspace does not exist")
    except AlreadyExistsError:
        raise HTTPException(status_code=409, detail="Workspace already exists")
    except ValidationError:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid workspace name. "
                "Please use only alphanumeric characters, hyphens, or underscores."
            ),
        )
    except crud.WorkspaceCrudError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

    return v1_models.Workspace(name=request.rename_to, is_active=False)


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


@v1.get("/workspaces/archive", tags=["Workspaces"], generate_unique_id_function=uniq_name)
async def list_archived_workspaces() -> v1_models.ListWorkspacesResponse:
    """List all archived workspaces."""
    wslist = await wscrud.get_archived_workspaces()

    resp = v1_models.ListWorkspacesResponse.from_db_workspaces(wslist)

    return resp


@v1.post(
    "/workspaces/archive/{workspace_name}/recover",
    tags=["Workspaces"],
    generate_unique_id_function=uniq_name,
    status_code=204,
)
async def recover_workspace(workspace_name: str):
    """Recover an archived workspace by name."""
    try:
        _ = await wscrud.recover_workspace(workspace_name)
    except crud.WorkspaceDoesNotExistError:
        raise HTTPException(status_code=404, detail="Workspace does not exist")
    except crud.WorkspaceCrudError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

    return Response(status_code=204)


@v1.delete(
    "/workspaces/archive/{workspace_name}",
    tags=["Workspaces"],
    generate_unique_id_function=uniq_name,
)
async def hard_delete_workspace(workspace_name: str):
    """Hard delete an archived workspace by name."""
    try:
        _ = await wscrud.hard_delete_workspace(workspace_name)
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
async def get_workspace_alerts(workspace_name: str) -> List[Optional[v1_models.AlertConversation]]:
    """Get alerts for a workspace."""
    try:
        ws = await wscrud.get_workspace_by_name(workspace_name)
    except crud.WorkspaceDoesNotExistError:
        raise HTTPException(status_code=404, detail="Workspace does not exist")
    except Exception:
        logger.exception("Error while getting workspace")
        raise HTTPException(status_code=500, detail="Internal server error")

    try:
        alerts = await dbreader.get_alerts_by_workspace(ws.id)
        prompts_outputs = await dbreader.get_prompts_with_output(ws.id)
        return await v1_processing.parse_get_alert_conversation(alerts, prompts_outputs)
    except Exception:
        logger.exception("Error while getting alerts and messages")
        raise HTTPException(status_code=500, detail="Internal server error")


@v1.get(
    "/workspaces/{workspace_name}/messages",
    tags=["Workspaces"],
    generate_unique_id_function=uniq_name,
)
async def get_workspace_messages(workspace_name: str) -> List[v1_models.Conversation]:
    """Get messages for a workspace."""
    try:
        ws = await wscrud.get_workspace_by_name(workspace_name)
    except crud.WorkspaceDoesNotExistError:
        raise HTTPException(status_code=404, detail="Workspace does not exist")
    except Exception:
        logger.exception("Error while getting workspace")
        raise HTTPException(status_code=500, detail="Internal server error")

    try:
        prompts_outputs = await dbreader.get_prompts_with_output(ws.id)
        conversations, _ = await v1_processing.parse_messages_in_conversations(prompts_outputs)
        return conversations
    except Exception:
        logger.exception("Error while getting messages")
        raise HTTPException(status_code=500, detail="Internal server error")


@v1.get(
    "/workspaces/{workspace_name}/custom-instructions",
    tags=["Workspaces"],
    generate_unique_id_function=uniq_name,
)
async def get_workspace_custom_instructions(workspace_name: str) -> v1_models.CustomInstructions:
    """Get the custom instructions of a workspace."""
    try:
        ws = await wscrud.get_workspace_by_name(workspace_name)
    except crud.WorkspaceDoesNotExistError:
        raise HTTPException(status_code=404, detail="Workspace does not exist")
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

    if ws.custom_instructions is None:
        return v1_models.CustomInstructions(prompt="")

    return v1_models.CustomInstructions(prompt=ws.custom_instructions)


@v1.put(
    "/workspaces/{workspace_name}/custom-instructions",
    tags=["Workspaces"],
    generate_unique_id_function=uniq_name,
    status_code=204,
)
async def set_workspace_custom_instructions(
    workspace_name: str, request: v1_models.CustomInstructions
):
    try:
        # This already checks if the workspace exists
        await wscrud.update_workspace_custom_instructions(workspace_name, [request.prompt])
    except crud.WorkspaceDoesNotExistError:
        raise HTTPException(status_code=404, detail="Workspace does not exist")
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

    return Response(status_code=204)


@v1.delete(
    "/workspaces/{workspace_name}/custom-instructions",
    tags=["Workspaces"],
    generate_unique_id_function=uniq_name,
    status_code=204,
)
async def delete_workspace_custom_instructions(workspace_name: str):
    try:
        # This already checks if the workspace exists
        await wscrud.update_workspace_custom_instructions(workspace_name, [])
    except crud.WorkspaceDoesNotExistError:
        raise HTTPException(status_code=404, detail="Workspace does not exist")
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

    return Response(status_code=204)


@v1.get(
    "/workspaces/{workspace_name}/muxes",
    tags=["Workspaces", "Muxes"],
    generate_unique_id_function=uniq_name,
)
async def get_workspace_muxes(
    workspace_name: str,
    pcrud: provendcrud.ProviderCrud = Depends(provendcrud.ProviderCrud),
) -> List[v1_models.MuxRule]:
    """Get the mux rules of a workspace.

    The list is ordered in order of priority. That is, the first rule in the list
    has the highest priority."""
    # TODO: This is a dummy implementation. In the future, we should have a proper
    # implementation that fetches the mux rules from the database.
    return [
        v1_models.MuxRule(
            provider="openai",
            model="gpt-3.5-turbo",
            matcher_type=v1_models.MuxMatcherType.file_regex,
            matcher=".*\\.txt",
        ),
        v1_models.MuxRule(
            provider="anthropic",
            model="davinci",
            matcher_type=v1_models.MuxMatcherType.catch_all,
        ),
    ]


@v1.put(
    "/workspaces/{workspace_name}/muxes",
    tags=["Workspaces", "Muxes"],
    generate_unique_id_function=uniq_name,
    status_code=204,
)
async def set_workspace_muxes(
    workspace_name: str,
    request: List[v1_models.MuxRule],
    pcrud: provendcrud.ProviderCrud = Depends(provendcrud.ProviderCrud),
):
    """Set the mux rules of a workspace."""
    # TODO: This is a dummy implementation. In the future, we should have a proper
    # implementation that sets the mux rules in the database.
    return Response(status_code=204)


@v1.get("/alerts_notification", tags=["Dashboard"], generate_unique_id_function=uniq_name)
async def stream_sse():
    """
    Send alerts event
    """
    return StreamingResponse(v1_processing.generate_sse_events(), media_type="text/event-stream")


@v1.get("/version", tags=["Dashboard"], generate_unique_id_function=uniq_name)
def version_check():
    try:
        latest_version = v1_processing.fetch_latest_version()

        # normalize the versions as github will return them with a 'v' prefix
        current_version = __version__.lstrip("v")
        latest_version_stripped = latest_version.lstrip("v")

        is_latest: bool = latest_version_stripped == current_version

        return {
            "current_version": current_version,
            "latest_version": latest_version_stripped,
            "is_latest": is_latest,
            "error": None,
        }
    except requests.RequestException as e:
        logger.error(f"RequestException: {str(e)}")
        return {
            "current_version": __version__,
            "latest_version": "unknown",
            "is_latest": None,
            "error": "An error occurred while fetching the latest version",
        }
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {
            "current_version": __version__,
            "latest_version": "unknown",
            "is_latest": None,
            "error": "An unexpected error occurred",
        }
