import json

import structlog
from fastapi import APIRouter, HTTPException, Request

from codegate.clients.clients import ClientType
from codegate.clients.detector import DetectClient
from codegate.extract_snippets.body_extractor import BodyCodeSnippetExtractorError
from codegate.extract_snippets.factory import BodyCodeExtractorFactory
from codegate.muxing import rulematcher
from codegate.muxing.adapter import BodyAdapter, ResponseAdapter
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

    def _extract_request_filenames(self, detected_client: ClientType, data: dict) -> set[str]:
        """
        Extract filenames from the request data.
        """
        try:
            body_extractor = BodyCodeExtractorFactory.create_snippet_extractor(detected_client)
            return body_extractor.extract_unique_filenames(data)
        except BodyCodeSnippetExtractorError as e:
            logger.error(f"Error extracting filenames from request: {e}")
            return set()

    async def _get_model_routes(self, filenames: set[str]) -> list[rulematcher.ModelRoute]:
        """
        Get the model routes for the given filenames.
        """
        model_routes = []
        mux_registry = await rulematcher.get_muxing_rules_registry()
        try:
            # Try to get a catch_all route
            single_model_route = await mux_registry.get_match_for_active_workspace(
                thing_to_match=None
            )
            model_routes.append(single_model_route)

            # Get the model routes for each filename
            for filename in filenames:
                model_route = await mux_registry.get_match_for_active_workspace(
                    thing_to_match=filename
                )
                model_routes.append(model_route)
        except Exception as e:
            logger.error(f"Error getting active workspace muxes: {e}")
            raise HTTPException(str(e), status_code=404)
        return model_routes

    def _setup_routes(self):

        @self.router.post(f"/{self.route_name}/{{rest_of_path:path}}")
        @DetectClient()
        async def route_to_dest_provider(
            request: Request,
            rest_of_path: str = "",
        ):
            """
            Route the request to the correct destination provider.

            1. Get destination provider from DB and active workspace.
            2. Map the request body to the destination provider format.
            3. Run pipeline. Selecting the correct destination provider.
            4. Transmit the response back to the client in the correct format.
            """

            body = await request.body()
            data = json.loads(body)

            filenames_in_data = self._extract_request_filenames(request.state.detected_client, data)
            logger.info(f"Extracted filenames from request: {filenames_in_data}")

            model_routes = await self._get_model_routes(filenames_in_data)
            if not model_routes:
                raise HTTPException("No rule found for the active workspace", status_code=404)

            # We still need some logic here to handle the case where we have multiple model routes.
            # For the moment since we match all only pick the first.
            model_route = model_routes[0]

            # Parse the input data and map it to the destination provider format
            rest_of_path = self._ensure_path_starts_with_slash(rest_of_path)
            new_data = self._body_adapter.map_body_to_dest(model_route, data)
            provider = self._provider_registry.get_provider(model_route.endpoint.provider_type)
            api_key = model_route.auth_material.auth_blob

            # Send the request to the destination provider. It will run the pipeline
            response = await provider.process_request(
                new_data, api_key, rest_of_path, request.state.detected_client
            )
            # Format the response to the client always using the OpenAI format
            return self._response_adapter.format_response_to_client(
                response, model_route.endpoint.provider_type
            )
