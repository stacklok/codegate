from typing import Callable, Optional

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

import codegate.providers.llamacpp.completion_handler as llamacpp
from codegate.clients.detector import DetectClient
from codegate.db.models import ProviderType
from codegate.muxing import models as mux_models
from codegate.muxing import rulematcher
from codegate.muxing.adapter import BodyAdapter, ResponseAdapter
from codegate.providers.fim_analyzer import FIMAnalyzer
from codegate.providers.registry import ProviderRegistry
from codegate.types import anthropic, ollama, openai
from codegate.workspaces.crud import WorkspaceCrud

from .anthropic_mappers import (
    anthropic_from_legacy_openai,
    anthropic_from_openai,
    anthropic_to_legacy_openai,
    anthropic_to_openai,
)
from .ollama_mappers import (
    ollama_chat_from_openai,
    ollama_chat_stream_to_openai_stream,
    ollama_generate_from_openai,
    ollama_generate_stream_to_openai_stream,
)

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
        self, thing_to_match: mux_models.ThingToMatchMux
    ) -> Optional[rulematcher.ModelRoute]:
        """
        Get the model route for the given things_to_match.
        """
        mux_registry = await rulematcher.get_muxing_rules_registry()
        try:
            # Try to get a model route for the active workspace
            model_route = await mux_registry.get_match_for_active_workspace(thing_to_match)
            return model_route
        except rulematcher.MuxMatchingError as e:
            logger.exception(f"Error matching rule and getting model route: {e}")
            raise HTTPException(detail=str(e), status_code=404)
        except Exception as e:
            logger.exception(f"Error getting active workspace muxes: {e}")
            raise HTTPException(detail=str(e), status_code=404)

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
            4. Transmit the response back to the client in OpenAI format.
            """

            body = await request.body()
            parsed = None
            match rest_of_path:
                case "chat/completions":
                    parsed = openai.ChatCompletionRequest.model_validate_json(body)
                case "api/v1/chat/completions":
                    parsed = openai.ChatCompletionRequest.model_validate_json(body)
                case "completions":
                    parsed = openai.LegacyCompletionRequest.model_validate_json(body)
                case "api/v1/completions":
                    parsed = openai.LegacyCompletionRequest.model_validate_json(body)
                case _:
                    raise ValueError(f"unknown rest of path: {rest_of_path}")
            is_fim_request = FIMAnalyzer.is_fim_request(rest_of_path, parsed)

            # 1. Get destination provider from DB and active workspace.
            thing_to_match = mux_models.ThingToMatchMux(
                body=parsed,
                url_request_path=rest_of_path,
                is_fim_request=is_fim_request,
                client_type=request.state.detected_client,
            )
            model_route = await self._get_model_route(thing_to_match)
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
            model, base_url = self._body_adapter.get_destination_info(model_route)

            # 3. Run pipeline. Selecting the correct destination provider.
            provider = self._provider_registry.get_provider(model_route.endpoint.provider_type)
            api_key = model_route.auth_material.auth_blob

            completion_function = default_completion_function
            from_openai = default_from_openai
            to_openai = default_to_openai
            # TODO this should be improved
            match model_route.endpoint.provider_type:
                case ProviderType.anthropic:
                    if is_fim_request:
                        completion_function = anthropic.acompletion
                        from_openai = anthropic_from_legacy_openai
                        to_openai = anthropic_to_legacy_openai
                    else:
                        completion_function = anthropic.acompletion
                        from_openai = anthropic_from_openai
                        to_openai = anthropic_to_openai
                case ProviderType.llamacpp:
                    if is_fim_request:
                        completion_function = llamacpp.complete
                        from_openai = identity
                        to_openai = identity
                    else:
                        completion_function = llamacpp.chat
                        from_openai = identity
                        to_openai = identity
                case ProviderType.ollama:
                    if is_fim_request:
                        completion_function = ollama.generate_streaming
                        from_openai = ollama_generate_from_openai
                        to_openai = ollama_generate_stream_to_openai_stream
                    else:
                        completion_function = ollama.chat_streaming
                        from_openai = ollama_chat_from_openai
                        to_openai = ollama_chat_stream_to_openai_stream
                case ProviderType.openai:
                    completion_function = openai.completions_streaming
                    from_openai = identity
                    to_openai = identity
                case ProviderType.openrouter:
                    completion_function = openai.completions_streaming
                    from_openai = identity
                    to_openai = identity
                case ProviderType.vllm:
                    completion_function = openai.completions_streaming
                    from_openai = identity
                    to_openai = identity

            response = await provider.process_request(
                parsed,
                api_key,
                base_url,
                is_fim_request,
                request.state.detected_client,
                completion_handler=inout_transformer(
                    from_openai,
                    to_openai,
                    completion_function,
                    model,
                ),
                stream_generator=openai.stream_generator,
            )

            # 4. Transmit the response back to the client in OpenAI format.
            return StreamingResponse(
                response.body_iterator,
                status_code=response.status_code,
                headers=response.headers,
                background=response.background,
                media_type=response.media_type,
            )


def default_completion_function(*args, **kwargs):
    raise NotImplementedError


def default_from_openai(*args, **kwargs):
    raise NotImplementedError


def default_to_openai(*args, **kwargs):
    raise NotImplementedError


def identity(x):
    return x


def inout_transformer(
    from_openai: Callable,
    to_openai: Callable,
    completion_handler: Callable,
    model: str,
):
    async def _inner(
        request,
        base_url,
        api_key,
        stream=None,
        is_fim_request=None,
    ):
        # Map request from OpenAI
        new_request = from_openai(request)
        new_request.model = model

        # Execute e.g. acompletion from Anthropic types
        response = completion_handler(
            new_request,
            api_key,
            base_url,
        )

        # Wrap with an async generator that maps from
        # e.g. Anthropic types to OpenAI's.
        return to_openai(response)

    return _inner
