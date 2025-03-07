import json
from typing import Optional

import structlog
from fastapi import APIRouter, HTTPException, Request

from codegate.clients.detector import DetectClient
from codegate.muxing import models as mux_models
from codegate.muxing import rulematcher
from codegate.muxing.adapter import BodyAdapter, ResponseAdapter
from codegate.providers.fim_analyzer import FIMAnalyzer
from codegate.providers.registry import ProviderRegistry
from codegate.workspaces.crud import WorkspaceCrud, WorkspaceDoesNotExistError

logger = structlog.get_logger("codegate")


class MuxRouter:
    """
    MuxRouter is a class that handles the muxing requests and routes them to
    the correct destination provider.
    """

    def __init__(self, provider_registry: ProviderRegistry):
        self._ws_crud = WorkspaceCrud()
        self._body_adapter = BodyAdapter()
        self.router = APIRouter()
        self._setup_routes()
        self._provider_registry = provider_registry
        self._response_adapter = ResponseAdapter()

    @property
    def route_name(self) -> str:
        return "v1/mux"

    def get_routes(self) -> APIRouter:
        return self.router

    def _ensure_path_starts_with_slash(self, path: str) -> str:
        return path if path.startswith("/") else f"/{path}"

    async def _get_model_route(
        self, thing_to_match: mux_models.ThingToMatchMux, workspace_name: Optional[str] = None
    ) -> Optional[rulematcher.ModelRoute]:
        """
        Get the model route for the given things_to_match.

        If workspace_name is provided and exists, use that workspace.
        Otherwise, use the active workspace.
        """
        try:
            mux_registry = await rulematcher.get_muxing_rules_registry()
            relevant_workspace = await self._get_relevant_workspace_name(
                mux_registry, workspace_name
            )
            return await mux_registry.get_match_for_workspace(relevant_workspace, thing_to_match)
        except rulematcher.MuxMatchingError as e:
            logger.exception(f"Error matching rule and getting model route: {e}")
            raise HTTPException(detail=str(e), status_code=404)
        except Exception as e:
            logger.exception(f"Error getting workspace muxes: {e}")
            raise HTTPException(detail=str(e), status_code=404)

    async def _get_relevant_workspace_name(
        self, mreg: rulematcher.MuxingRulesinWorkspaces, workspace_name: Optional[str]
    ) -> str:
        if not workspace_name:
            # No workspace specified, use active workspace
            return mreg.get_active_workspace()

        try:
            # Verify the requested workspace exists
            # TODO: We should have an in-memory cache of the workspaces
            await self._ws_crud.get_workspace_by_name(workspace_name)
            logger.debug(f"Using workspace from X-CodeGate-Workspace header: {workspace_name}")
            return workspace_name
        except WorkspaceDoesNotExistError:
            # Workspace doesn't exist, fall back to active workspace
            logger.warning(
                f"Workspace {workspace_name} does not exist, falling back to active workspace"
            )
            return mreg.get_active_workspace()

    def _setup_routes(self):

        @self.router.post(f"/{self.route_name}/{{rest_of_path:path}}")
        @DetectClient()
        async def route_to_dest_provider(
            request: Request,
            rest_of_path: str = "",
        ):
            """
            Route the request to the correct destination provider.

            1. Get destination provider from DB and workspace (from header or active).
            2. Map the request body to the destination provider format.
            3. Run pipeline. Selecting the correct destination provider.
            4. Transmit the response back to the client in OpenAI format.
            """

            body = await request.body()
            data = json.loads(body)
            is_fim_request = FIMAnalyzer.is_fim_request(rest_of_path, data)

            # Check if X-CodeGate-Workspace header is present
            workspace_header = request.headers.get("X-CodeGate-Workspace")

            # 1. Get destination provider from DB and workspace (from header or active).
            thing_to_match = mux_models.ThingToMatchMux(
                body=data,
                url_request_path=rest_of_path,
                is_fim_request=is_fim_request,
                client_type=request.state.detected_client,
            )
            model_route = await self._get_model_route(thing_to_match, workspace_header)
            if not model_route:
                raise HTTPException(
                    detail="No matching rule found for the active workspace", status_code=404
                )

            logger.info(
                "Muxing request routed to destination provider",
                model=model_route.model.name,
                provider_type=model_route.endpoint.provider_type,
                provider_name=model_route.endpoint.name,
                is_fim_request=is_fim_request,
            )

            # 2. Map the request body to the destination provider format.
            rest_of_path = self._ensure_path_starts_with_slash(rest_of_path)
            new_data = self._body_adapter.set_destination_info(model_route, data)

            # 3. Run pipeline. Selecting the correct destination provider.
            provider = self._provider_registry.get_provider(model_route.endpoint.provider_type)
            api_key = model_route.auth_material.auth_blob
            response = await provider.process_request(
                new_data, api_key, is_fim_request, request.state.detected_client
            )

            # 4. Transmit the response back to the client in OpenAI format.
            return self._response_adapter.format_response_to_client(
                response, model_route.endpoint.provider_type, is_fim_request=is_fim_request
            )
