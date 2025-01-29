import json

import structlog
from fastapi import APIRouter, HTTPException, Request

from codegate.db.models import MuxRuleProviderEndpoint
from codegate.mux.adapter import BodyAdapter
from codegate.providers.registry import ProviderRegistry
from codegate.workspaces.crud import WorkspaceCrud

logger = structlog.get_logger("codegate")


class MuxRouter:
    """
    MuxRouter is a class that handles the muxing requests and routes them to
    the correct destination provider.
    """

    def __init__(self, provider_registry: ProviderRegistry):
        self._ws_crud = WorkspaceCrud()
        self._adapter = BodyAdapter()
        self.router = APIRouter()
        self._setup_routes()
        self._provider_registry = provider_registry

    @property
    def route_name(self) -> str:
        return "v1/mux"

    def get_routes(self) -> APIRouter:
        return self.router

    def _set_destination_info(self, data: dict, mux_and_provider: MuxRuleProviderEndpoint) -> dict:
        data["model"] = mux_and_provider.provider_model_name
        data["base_url"] = mux_and_provider.endpoint
        return data

    def _ensure_path_starts_with_slash(self, path: str) -> str:
        return path if path.startswith("/") else f"/{path}"

    def _setup_routes(self):

        @self.router.post(f"/{self.route_name}/{{rest_of_path:path}}")
        async def route_to_dest_provider(
            request: Request,
            rest_of_path: str = "",
        ):
            """
            Route the request to the correct destination provider.

            1. Get destination provider from DB and active workspace
            2. Map the request body to the destination provider format
            3. Run pipeline. Selecting the correct destination provider
            4. Transmit the response back to the client
            """

            body = await request.body()
            data = json.loads(body)

            try:
                active_ws_muxes = await self._ws_crud.get_active_workspace_muxes()
            except Exception as e:
                logger.error(f"Error getting active workspace muxes: {e}")
                raise HTTPException(str(e))

            # TODO: Implement the muxing logic here. For the moment we will assume
            # that we have a single mux, i.e. a single destination provider.
            if len(active_ws_muxes) == 0:
                raise HTTPException(status_code=404, detail="No active workspace muxes found")
            mux_and_provider = active_ws_muxes[0]

            rest_of_path = self._ensure_path_starts_with_slash(rest_of_path)
            new_data = self._adapter.map_body_to_dest(mux_and_provider.provider_type, data)
            new_data = self._set_destination_info(new_data, mux_and_provider)
            provider = self._provider_registry.get_provider(mux_and_provider.provider_type)

            # TODO: Implement the logic to get the api_key from the DB
            api_key = mux_and_provider.auth_blob

            return await provider.process_request(new_data, api_key, rest_of_path)
