from typing import List, Optional

import cachetools.func
import requests
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ValidationError

import codegate.muxing.models as mux_models
from codegate import __version__
from codegate.api import v1_models, v1_processing
from codegate.config import API_DEFAULT_PAGE_SIZE, API_MAX_PAGE_SIZE
from codegate.db.connection import AlreadyExistsError, DbReader
from codegate.db.models import AlertSeverity, AlertTriggerType, Persona
from codegate.muxing.persona import (
    PersonaDoesNotExistError,
    PersonaManager,
    PersonaSimilarDescriptionError,
)
from codegate.providers import crud as provendcrud
from codegate.updates.client import Origin, get_update_client_singleton
from codegate.workspaces import crud

logger = structlog.get_logger("codegate")

v1 = APIRouter()
wscrud = crud.WorkspaceCrud()
pcrud = provendcrud.ProviderCrud()
persona_manager = PersonaManager()

# This is a singleton object
dbreader = DbReader()


def uniq_name(route: APIRoute):
    return f"v1_{route.name}"


class FilterByNameParams(BaseModel):
    name: Optional[str] = None


@v1.get("/provider-endpoints", tags=["Providers"], generate_unique_id_function=uniq_name)
async def list_provider_endpoints(
    filter_query: FilterByNameParams = Depends(),
) -> List[v1_models.ProviderEndpoint]:
    """List all provider endpoints."""
    if filter_query.name is None:
        try:
            return await pcrud.list_endpoints()
        except Exception:
            raise HTTPException(status_code=500, detail="Internal server error")

    try:
        provend = await pcrud.get_endpoint_by_name(filter_query.name)
        return [provend]
    except provendcrud.ProviderNotFoundError:
        raise HTTPException(status_code=404, detail="Provider endpoint not found")
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")


# This needs to be above /provider-endpoints/{provider_name} to avoid conflict
@v1.get(
    "/provider-endpoints/models",
    tags=["Providers"],
    generate_unique_id_function=uniq_name,
)
async def list_all_models_for_all_providers() -> List[v1_models.ModelByProvider]:
    """List all models for all providers."""
    try:
        return await pcrud.get_all_models()
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")


@v1.get(
    "/provider-endpoints/{provider_name}/models",
    tags=["Providers"],
    generate_unique_id_function=uniq_name,
)
async def list_models_by_provider(
    provider_name: str,
) -> List[v1_models.ModelByProvider]:
    """List models by provider."""

    try:
        provider = await pcrud.get_endpoint_by_name(provider_name)
        return await pcrud.models_by_provider(provider.id)
    except provendcrud.ProviderNotFoundError:
        raise HTTPException(status_code=404, detail="Provider not found")
    except Exception as e:
        logger.exception("Error while listing models by provider")
        raise HTTPException(status_code=500, detail=str(e))


@v1.get(
    "/provider-endpoints/{provider_name}", tags=["Providers"], generate_unique_id_function=uniq_name
)
async def get_provider_endpoint(
    provider_name: str,
) -> v1_models.ProviderEndpoint:
    """Get a provider endpoint by name."""
    try:
        provend = await pcrud.get_endpoint_by_name(provider_name)
    except provendcrud.ProviderNotFoundError:
        raise HTTPException(status_code=404, detail="Provider endpoint not found")
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")
    return provend


@v1.post(
    "/provider-endpoints",
    tags=["Providers"],
    generate_unique_id_function=uniq_name,
    status_code=201,
)
async def add_provider_endpoint(
    request: v1_models.AddProviderEndpointRequest,
) -> v1_models.ProviderEndpoint:
    """Add a provider endpoint."""
    try:
        provend = await pcrud.add_endpoint(request)
    except AlreadyExistsError:
        raise HTTPException(status_code=409, detail="Provider endpoint already exists")
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )
    except provendcrud.ProviderModelsNotFoundError:
        raise HTTPException(status_code=401, detail="Provider models could not be found")
    except provendcrud.ProviderInvalidAuthConfigError:
        raise HTTPException(status_code=400, detail="Invalid auth configuration")
    except ValidationError as e:
        # TODO: This should be more specific
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )
    except Exception:
        logger.exception("Error while adding provider endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")

    return provend


@v1.put(
    "/provider-endpoints/{provider_name}/auth-material",
    tags=["Providers"],
    generate_unique_id_function=uniq_name,
    status_code=204,
)
async def configure_auth_material(
    provider_name: str,
    request: v1_models.ConfigureAuthMaterial,
):
    """Configure auth material for a provider."""
    try:
        provider = await pcrud.get_endpoint_by_name(provider_name)
        await pcrud.configure_auth_material(provider.id, request)
    except provendcrud.ProviderNotFoundError:
        raise HTTPException(status_code=404, detail="Provider endpoint not found")
    except provendcrud.ProviderModelsNotFoundError:
        raise HTTPException(status_code=401, detail="Provider models could not be found")
    except provendcrud.ProviderInvalidAuthConfigError:
        raise HTTPException(status_code=400, detail="Invalid auth configuration")
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

    return Response(status_code=204)


@v1.put(
    "/provider-endpoints/{provider_name}", tags=["Providers"], generate_unique_id_function=uniq_name
)
async def update_provider_endpoint(
    provider_name: str,
    request: v1_models.ProviderEndpoint,
) -> v1_models.ProviderEndpoint:
    """Update a provider endpoint by name."""
    try:
        provider = await pcrud.get_endpoint_by_name(provider_name)
        request.id = str(provider.id)
        provend = await pcrud.update_endpoint(request)
    except provendcrud.ProviderNotFoundError:
        raise HTTPException(status_code=404, detail="Provider endpoint not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValidationError as e:
        # TODO: This should be more specific
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("Error while updating provider endpoint")
        raise HTTPException(status_code=500, detail=str(e))

    return provend


@v1.delete(
    "/provider-endpoints/{provider_name}", tags=["Providers"], generate_unique_id_function=uniq_name
)
async def delete_provider_endpoint(
    provider_name: str,
):
    """Delete a provider endpoint by name."""
    try:
        provider = await pcrud.get_endpoint_by_name(provider_name)
        await pcrud.delete_endpoint(provider.id)
    except provendcrud.ProviderNotFoundError:
        raise HTTPException(status_code=404, detail="Provider endpoint not found")
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")
    return Response(status_code=204)


@v1.get("/workspaces", tags=["Workspaces"], generate_unique_id_function=uniq_name)
async def list_workspaces(
    provider_name: Optional[str] = Query(None),
) -> v1_models.ListWorkspacesResponse:
    """
    List all workspaces.

    Args:
        provider_name (Optional[str]): Filter workspaces by provider name. If provided,
        will return workspaces where models from the specified provider (e.g., OpenAI,
        Anthropic) have been used in workspace muxing rules.

    Returns:
        ListWorkspacesResponse: A response object containing the list of workspaces.
    """
    try:
        if provider_name:
            provider = await pcrud.get_endpoint_by_name(provider_name)
            wslist = await wscrud.workspaces_by_provider(provider.id)
            resp = v1_models.ListWorkspacesResponse.from_db_workspaces(wslist)
            return resp
        else:
            wslist = await wscrud.get_workspaces()
            resp = v1_models.ListWorkspacesResponse.from_db_workspaces_with_sessioninfo(wslist)
            return resp
    except provendcrud.ProviderNotFoundError:
        return v1_models.ListWorkspacesResponse(workspaces=[])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    request: v1_models.FullWorkspace,
) -> v1_models.FullWorkspace:
    """Create a new workspace."""
    try:
        custom_instructions = request.config.custom_instructions if request.config else None
        mux_rules = []
        if request.config and request.config.muxing_rules:
            mux_rules = await pcrud.add_provider_ids_to_mux_rule_list(request.config.muxing_rules)

        workspace_row, created_mux_rules = await wscrud.add_workspace(
            request.name, custom_instructions, mux_rules
        )

        created_muxes_with_name_type = [
            mux_models.MuxRule.from_db_models(
                mux_rule, await pcrud.get_endpoint_by_id(mux_rule.provider_endpoint_id)
            )
            for mux_rule in created_mux_rules
        ]
    except crud.WorkspaceNameAlreadyInUseError:
        raise HTTPException(status_code=409, detail="Workspace name already in use")
    except ValidationError:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid workspace name. "
                "Please use only alphanumeric characters, hyphens, or underscores."
            ),
        )
    except provendcrud.ProviderNotFoundError as e:
        logger.exception("Error matching a provider for a mux rule while creating a workspace")
        raise HTTPException(status_code=400, detail=str(e))
    except crud.WorkspaceCrudError as e:
        logger.exception("Error while create a workspace")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Error while creating workspace")
        raise HTTPException(status_code=500, detail="Internal server error")

    return v1_models.FullWorkspace(
        name=workspace_row.name,
        config=v1_models.WorkspaceConfig(
            custom_instructions=workspace_row.custom_instructions or "",
            muxing_rules=created_muxes_with_name_type,
        ),
    )


@v1.put(
    "/workspaces/{workspace_name}",
    tags=["Workspaces"],
    generate_unique_id_function=uniq_name,
    status_code=200,
)
async def update_workspace(
    workspace_name: str,
    request: v1_models.FullWorkspace,
) -> v1_models.FullWorkspace:
    """Update a workspace."""
    try:
        custom_instructions = request.config.custom_instructions if request.config else None
        mux_rules = []
        if request.config and request.config.muxing_rules:
            mux_rules = await pcrud.add_provider_ids_to_mux_rule_list(request.config.muxing_rules)

        workspace_row, updated_muxes = await wscrud.update_workspace(
            workspace_name,
            request.name,
            custom_instructions,
            mux_rules,
        )

        updated_muxes_with_name_type = [
            mux_models.MuxRule.from_db_models(
                mux_rule, await pcrud.get_endpoint_by_id(mux_rule.provider_endpoint_id)
            )
            for mux_rule in updated_muxes
        ]

    except provendcrud.ProviderNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except crud.WorkspaceDoesNotExistError:
        raise HTTPException(status_code=404, detail="Workspace does not exist")
    except crud.WorkspaceNameAlreadyInUseError:
        raise HTTPException(status_code=409, detail="Workspace name already in use")
    except ValidationError:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid workspace name. "
                "Please use only alphanumeric characters, hyphens, or underscores."
            ),
        )
    except crud.WorkspaceCrudError as e:
        logger.exception("Error while updating workspace")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

    return v1_models.FullWorkspace(
        name=workspace_row.name,
        config=v1_models.WorkspaceConfig(
            custom_instructions=workspace_row.custom_instructions or "",
            muxing_rules=updated_muxes_with_name_type,
        ),
    )


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
    except crud.DeleteMuxesFromRegistryError:
        logger.exception("Error deleting muxes while deleting workspace")
        raise HTTPException(status_code=500, detail="Internal server error")
    except Exception:
        logger.exception("Error while deleting workspace")
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
        alerts = await dbreader.get_alerts_by_workspace_or_prompt_id(
            workspace_id=ws.id, trigger_category=AlertSeverity.CRITICAL.value
        )
        prompts_outputs = await dbreader.get_prompts_with_output(ws.id)
        return await v1_processing.parse_get_alert_conversation(alerts, prompts_outputs)
    except Exception:
        logger.exception("Error while getting alerts and messages")
        raise HTTPException(status_code=500, detail="Internal server error")


@v1.get(
    "/workspaces/{workspace_name}/alerts-summary",
    tags=["Workspaces"],
    generate_unique_id_function=uniq_name,
)
async def get_workspace_alerts_summary(workspace_name: str) -> v1_models.AlertSummary:
    """Get alert summary for a workspace."""
    try:
        ws = await wscrud.get_workspace_by_name(workspace_name)
    except crud.WorkspaceDoesNotExistError:
        raise HTTPException(status_code=404, detail="Workspace does not exist")
    except Exception:
        logger.exception("Error while getting workspace")
        raise HTTPException(status_code=500, detail="Internal server error")

    try:
        summary = await dbreader.get_alerts_summary(workspace_id=ws.id)
        return v1_models.AlertSummary(
            malicious_packages=summary.total_packages_count,
            pii=summary.total_pii_count,
            secrets=summary.total_secrets_count,
            total_alerts=summary.total_alerts,
        )
    except Exception:
        logger.exception("Error while getting alerts summary")
        raise HTTPException(status_code=500, detail="Internal server error")


@v1.get(
    "/workspaces/{workspace_name}/messages",
    tags=["Workspaces"],
    generate_unique_id_function=uniq_name,
)
async def get_workspace_messages(
    workspace_name: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(API_DEFAULT_PAGE_SIZE, ge=1, le=API_MAX_PAGE_SIZE),
    filter_by_ids: Optional[List[str]] = Query(None),
    filter_by_alert_trigger_types: Optional[List[AlertTriggerType]] = Query(None),
) -> v1_models.PaginatedMessagesResponse:
    """Get messages for a workspace."""
    try:
        ws = await wscrud.get_workspace_by_name(workspace_name)
    except crud.WorkspaceDoesNotExistError:
        raise HTTPException(status_code=404, detail="Workspace does not exist")
    except Exception:
        logger.exception("Error while getting workspace")
        raise HTTPException(status_code=500, detail="Internal server error")

    offset = (page - 1) * page_size
    valid_conversations: List[v1_models.ConversationSummary] = []
    fetched_prompts = 0

    while len(valid_conversations) < page_size:
        batch_size = page_size * 2  # Fetch more prompts to compensate for potential skips

        prompts = await dbreader.get_prompts(
            ws.id,
            offset + fetched_prompts,
            batch_size,
            filter_by_ids,
            list([AlertSeverity.CRITICAL.value]),
            filter_by_alert_trigger_types,
        )

        if not prompts or len(prompts) == 0:
            break

        # iterate for all prompts to compose the conversation summary
        for prompt in prompts:
            fetched_prompts += 1
            if not prompt.request:
                logger.warning(f"Skipping prompt {prompt.id}. Empty request field")
                continue

            messages, _ = await v1_processing.parse_request(prompt.request)
            if not messages or len(messages) == 0:
                logger.warning(f"Skipping prompt {prompt.id}. No messages found")
                continue

            # message is just the first entry in the request, cleaned properly
            message = v1_processing.parse_question_answer(messages[0])
            message_obj = v1_models.ChatMessage(
                message=message, timestamp=prompt.timestamp, message_id=prompt.id
            )

            # count total alerts for the prompt
            total_alerts_row = await dbreader.get_alerts_summary(prompt_id=prompt.id)

            # get token usage for the prompt
            prompts_outputs = await dbreader.get_prompts_with_output(prompt_id=prompt.id)
            ws_token_usage = await v1_processing.parse_workspace_token_usage(prompts_outputs)

            conversation_summary = v1_models.ConversationSummary(
                chat_id=prompt.id,
                prompt=message_obj,
                provider=prompt.provider,
                type=prompt.type,
                conversation_timestamp=prompt.timestamp,
                alerts_summary=v1_models.AlertSummary(
                    malicious_packages=total_alerts_row.total_packages_count,
                    pii=total_alerts_row.total_pii_count,
                    secrets=total_alerts_row.total_secrets_count,
                    total_alerts=total_alerts_row.total_alerts,
                ),
                total_alerts=total_alerts_row.total_alerts,
                token_usage_agg=ws_token_usage,
            )

            valid_conversations.append(conversation_summary)
            if len(valid_conversations) >= page_size:
                break

    # Fetch total message count
    total_count = await dbreader.get_total_messages_count_by_workspace_id(
        ws.id,
        filter_by_ids,
        list([AlertSeverity.CRITICAL.value]),
        filter_by_alert_trigger_types,
    )

    return v1_models.PaginatedMessagesResponse(
        data=valid_conversations,
        limit=page_size,
        offset=offset,
        total=total_count,
    )


@v1.get(
    "/workspaces/{workspace_name}/messages/{prompt_id}",
    tags=["Workspaces"],
    generate_unique_id_function=uniq_name,
)
async def get_messages_by_prompt_id(
    workspace_name: str,
    prompt_id: str,
) -> v1_models.Conversation:
    """Get messages for a workspace."""
    try:
        ws = await wscrud.get_workspace_by_name(workspace_name)
    except crud.WorkspaceDoesNotExistError:
        raise HTTPException(status_code=404, detail="Workspace does not exist")
    except Exception:
        logger.exception("Error while getting workspace")
        raise HTTPException(status_code=500, detail="Internal server error")
    prompts_outputs = await dbreader.get_prompts_with_output(
        workspace_id=ws.id, prompt_id=prompt_id
    )

    # get all alerts for the prompt
    alerts = await dbreader.get_alerts_by_workspace_or_prompt_id(
        workspace_id=ws.id, prompt_id=prompt_id, trigger_category=AlertSeverity.CRITICAL.value
    )
    deduped_alerts = await v1_processing.remove_duplicate_alerts(alerts)
    conversations, _ = await v1_processing.parse_messages_in_conversations(prompts_outputs)
    if not conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conversation = conversations[0]
    conversation.alerts = deduped_alerts
    return conversation


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
) -> List[mux_models.MuxRule]:
    """Get the mux rules of a workspace.

    The list is ordered in order of priority. That is, the first rule in the list
    has the highest priority."""
    try:
        db_muxes = await wscrud.get_muxes(workspace_name)

        muxes = []
        for db_mux in db_muxes:
            db_endpoint = await pcrud.get_endpoint_by_id(db_mux.provider_endpoint_id)
            mux_rule = mux_models.MuxRule.from_db_models(db_mux, db_endpoint)
            muxes.append(mux_rule)
    except crud.WorkspaceDoesNotExistError:
        raise HTTPException(status_code=404, detail="Workspace does not exist")
    except Exception:
        logger.exception("Error while getting workspace")
        raise HTTPException(status_code=500, detail="Internal server error")

    return [mux_models.MuxRule.from_mux_rule_with_provider_id(mux) for mux in muxes]


@v1.put(
    "/workspaces/{workspace_name}/muxes",
    tags=["Workspaces", "Muxes"],
    generate_unique_id_function=uniq_name,
    status_code=204,
)
async def set_workspace_muxes(
    workspace_name: str,
    request: List[mux_models.MuxRule],
):
    """Set the mux rules of a workspace."""
    try:
        mux_rules = await pcrud.add_provider_ids_to_mux_rule_list(request)
        await wscrud.set_muxes(workspace_name, mux_rules)
    except provendcrud.ProviderNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except crud.WorkspaceDoesNotExistError:
        raise HTTPException(status_code=404, detail="Workspace does not exist")
    except crud.WorkspaceCrudError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Error while setting muxes {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

    return Response(status_code=204)


@v1.get(
    "/workspaces/{workspace_name}",
    tags=["Workspaces"],
    generate_unique_id_function=uniq_name,
)
async def get_workspace_by_name(
    workspace_name: str,
) -> v1_models.FullWorkspace:
    """List workspaces by provider ID."""
    try:
        ws = await wscrud.get_workspace_by_name(workspace_name)
        db_muxes = await wscrud.get_muxes(workspace_name)

        muxes = []
        for db_mux in db_muxes:
            db_endpoint = await pcrud.get_endpoint_by_id(db_mux.provider_endpoint_id)
            mux_rule = mux_models.MuxRule.from_db_models(db_mux, db_endpoint)
            muxes.append(mux_rule)

        return v1_models.FullWorkspace(
            name=ws.name,
            config=v1_models.WorkspaceConfig(
                custom_instructions=ws.custom_instructions or "",
                muxing_rules=muxes,
            ),
        )

    except crud.WorkspaceDoesNotExistError:
        raise HTTPException(status_code=404, detail="Workspace does not exist")
    except Exception as e:
        logger.exception(f"Error while getting workspace {e}")
        raise HTTPException(status_code=500, detail=str(e))


@v1.get("/alerts_notification", tags=["Dashboard"], generate_unique_id_function=uniq_name)
async def stream_sse():
    """
    Send alerts event
    """
    return StreamingResponse(v1_processing.generate_sse_events(), media_type="text/event-stream")


@v1.get("/version", tags=["Dashboard"], generate_unique_id_function=uniq_name)
async def version_check():
    try:
        latest_version = _get_latest_version()
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


@v1.get(
    "/workspaces/{workspace_name}/token-usage",
    tags=["Workspaces", "Token Usage"],
    generate_unique_id_function=uniq_name,
)
async def get_workspace_token_usage(workspace_name: str) -> v1_models.TokenUsageAggregate:
    """Get the token usage of a workspace."""

    try:
        ws = await wscrud.get_workspace_by_name(workspace_name)
    except crud.WorkspaceDoesNotExistError:
        raise HTTPException(status_code=404, detail="Workspace does not exist")
    except Exception:
        logger.exception("Error while getting workspace")
        raise HTTPException(status_code=500, detail="Internal server error")

    try:
        prompts_outputs = await dbreader.get_prompts_with_output(workspace_id=ws.id)
        ws_token_usage = await v1_processing.parse_workspace_token_usage(prompts_outputs)
        return ws_token_usage
    except Exception:
        logger.exception("Error while getting messages")
        raise HTTPException(status_code=500, detail="Internal server error")


@v1.get("/personas", tags=["Personas"], generate_unique_id_function=uniq_name)
async def list_personas() -> List[Persona]:
    """List all personas."""
    try:
        personas = await persona_manager.get_all_personas()
        return personas
    except Exception:
        logger.exception("Error while getting personas")
        raise HTTPException(status_code=500, detail="Internal server error")


@v1.get("/personas/{persona_name}", tags=["Personas"], generate_unique_id_function=uniq_name)
async def get_persona(persona_name: str) -> Persona:
    """Get a persona by name."""
    try:
        persona = await persona_manager.get_persona(persona_name)
        return persona
    except PersonaDoesNotExistError:
        logger.exception("Error while getting persona")
        raise HTTPException(status_code=404, detail="Persona does not exist")


@v1.post("/personas", tags=["Personas"], generate_unique_id_function=uniq_name, status_code=201)
async def create_persona(request: v1_models.PersonaRequest) -> Persona:
    """Create a new persona."""
    try:
        await persona_manager.add_persona(request.name, request.description)
        persona = await dbreader.get_persona_by_name(request.name)
        return persona
    except PersonaSimilarDescriptionError:
        logger.exception("Error while creating persona")
        raise HTTPException(status_code=409, detail="Persona has a similar description to another")
    except AlreadyExistsError:
        logger.exception("Error while creating persona")
        raise HTTPException(status_code=409, detail="Persona already exists")
    except ValidationError:
        logger.exception("Error while creating persona")
        raise HTTPException(
            status_code=400,
            detail=(
                "Persona has invalid name, check is alphanumeric "
                "and only contains dashes and underscores"
            ),
        )
    except Exception:
        logger.exception("Error while creating persona")
        raise HTTPException(status_code=500, detail="Internal server error")


@v1.put("/personas/{persona_name}", tags=["Personas"], generate_unique_id_function=uniq_name)
async def update_persona(persona_name: str, request: v1_models.PersonaUpdateRequest) -> Persona:
    """Update an existing persona."""
    try:
        await persona_manager.update_persona(
            persona_name, request.new_name, request.new_description
        )
        persona = await dbreader.get_persona_by_name(request.new_name)
        return persona
    except PersonaSimilarDescriptionError:
        logger.exception("Error while updating persona")
        raise HTTPException(status_code=409, detail="Persona has a similar description to another")
    except PersonaDoesNotExistError:
        logger.exception("Error while updating persona")
        raise HTTPException(status_code=404, detail="Persona does not exist")
    except AlreadyExistsError:
        logger.exception("Error while updating persona")
        raise HTTPException(status_code=409, detail="Persona already exists")
    except ValidationError:
        logger.exception("Error while creating persona")
        raise HTTPException(
            status_code=400,
            detail=(
                "Persona has invalid name, check is alphanumeric "
                "and only contains dashes and underscores"
            ),
        )
    except Exception:
        logger.exception("Error while updating persona")
        raise HTTPException(status_code=500, detail="Internal server error")


@v1.delete(
    "/personas/{persona_name}",
    tags=["Personas"],
    generate_unique_id_function=uniq_name,
    status_code=204,
)
async def delete_persona(persona_name: str):
    """Delete a persona."""
    try:
        await persona_manager.delete_persona(persona_name)
        return Response(status_code=204)
    except PersonaDoesNotExistError:
        logger.exception("Error while updating persona")
        raise HTTPException(status_code=404, detail="Persona does not exist")
    except Exception:
        logger.exception("Error while deleting persona")
        raise HTTPException(status_code=500, detail="Internal server error")


@cachetools.func.ttl_cache(maxsize=128, ttl=20 * 60)
def _get_latest_version():
    update_client = get_update_client_singleton()
    return update_client.get_latest_version(Origin.FrontEnd)
