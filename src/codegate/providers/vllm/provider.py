import json
from typing import Callable, List
from urllib.parse import urljoin

import httpx
import structlog
from fastapi import Header, HTTPException, Request

from codegate.clients.clients import ClientType
from codegate.clients.detector import DetectClient
from codegate.pipeline.factory import PipelineFactory
from codegate.providers.base import BaseProvider, ModelFetchError
from codegate.providers.fim_analyzer import FIMAnalyzer
from codegate.providers.litellmshim import LiteLLmShim
from codegate.types.vllm import (
    completions_streaming,
    stream_generator,
    ChatCompletionRequest,
    LegacyCompletionRequest,
)


logger = structlog.get_logger("codegate")


class VLLMProvider(BaseProvider):
    def __init__(
        self,
        pipeline_factory: PipelineFactory,
    ):
        if self._get_base_url() != "":
            self.base_url = self._get_base_url()
        else:
            self.base_url = "http://localhost:8000"
        completion_handler = LiteLLmShim(
            completion_func=completions_streaming,
            stream_generator=stream_generator,
        )
        super().__init__(
            None,
            None,
            completion_handler,
            pipeline_factory,
        )

    @property
    def provider_route_name(self) -> str:
        return "vllm"

    def _get_base_url(self) -> str:
        """
        Get the base URL from config with proper formatting
        """
        base_url = super()._get_base_url()
        if base_url:
            base_url = base_url.rstrip("/")
        return base_url

    def models(self, endpoint: str = None, api_key: str = None) -> List[str]:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if not endpoint:
            endpoint = self._get_base_url()

        resp = httpx.get(
            f"{endpoint}/v1/models",
            headers=headers,
        )

        if resp.status_code != 200:
            raise ModelFetchError(f"Failed to fetch models from vLLM API: {resp.text}")

        jsonresp = resp.json()

        return [model["id"] for model in jsonresp.get("data", [])]

    async def process_request(
        self,
        data: dict,
        api_key: str,
        base_url: str,
        is_fim_request: bool,
        client_type: ClientType,
        completion_handler: Callable | None = None,
        stream_generator: Callable | None = None,
    ):
        try:
            # Pass the potentially None api_key to complete
            stream = await self.complete(
                data,
                api_key,
                base_url,
                is_fim_request=is_fim_request,
                client_type=client_type,
                completion_handler=completion_handler,
            )
        except Exception as e:
            # Check if we have a status code there
            if hasattr(e, "status_code"):
                logger = structlog.get_logger("codegate")
                logger.error("Error in VLLMProvider completion", error=str(e))
                raise HTTPException(status_code=e.status_code, detail=str(e))
            raise e
        return self._completion_handler.create_response(stream, client_type, stream_generator=stream_generator)

    def _setup_routes(self):
        """
        Sets up the /chat/completions route for the provider as expected by the
        OpenAI API. Makes the API key optional in the "Authorization" header.
        """

        @self.router.get(f"/{self.provider_route_name}/models")
        async def get_models(
            authorization: str | None = Header(None, description="Optional Bearer token")
        ):
            base_url = self._get_base_url()
            headers = {}

            if authorization:
                if not authorization.startswith("Bearer "):
                    raise HTTPException(
                        status_code=401, detail="Invalid authorization header format"
                    )
                token = authorization.split(" ")[1]
                headers["Authorization"] = f"Bearer {token}"

            try:
                models_url = urljoin(base_url, "v1/models")
                async with httpx.AsyncClient() as client:
                    response = await client.get(models_url, headers=headers)
                    response.raise_for_status()
                    return response.json()
            except httpx.HTTPError as e:
                logger.error("Error fetching vLLM models", error=str(e))
                raise HTTPException(
                    status_code=e.response.status_code if hasattr(e, "response") else 500,
                    detail=str(e),
                )

        @self.router.post(f"/{self.provider_route_name}/completions")
        @DetectClient()
        async def create_completion(
            request: Request,
            authorization: str | None = Header(None, description="Optional Bearer token"),
        ):
            api_key = None
            if authorization:
                if not authorization.startswith("Bearer "):
                    raise HTTPException(
                        status_code=401, detail="Invalid authorization header format"
                    )
                api_key = authorization.split(" ")[1]

            body = await request.body()
            req = LegacyCompletionRequest.model_validate_json(body)
            is_fim_request = FIMAnalyzer.is_fim_request(request.url.path, req)

            if not req.stream:
                logger.warn("We got a non-streaming request, forcing to a streaming one")
                req.stream = True

            return await self.process_request(
                req,
                api_key,
                self.base_url,
                is_fim_request,
                request.state.detected_client,
            )

        @self.router.post(f"/{self.provider_route_name}/chat/completions")
        @DetectClient()
        async def create_completion(
            request: Request,
            authorization: str | None = Header(None, description="Optional Bearer token"),
        ):
            api_key = None
            if authorization:
                if not authorization.startswith("Bearer "):
                    raise HTTPException(
                        status_code=401, detail="Invalid authorization header format"
                    )
                api_key = authorization.split(" ")[1]

            body = await request.body()
            req = ChatCompletionRequest.model_validate_json(body)
            is_fim_request = FIMAnalyzer.is_fim_request(request.url.path, req)

            if not req.stream:
                logger.warn("We got a non-streaming request, forcing to a streaming one")
                req.stream = True

            return await self.process_request(
                req,
                api_key,
                self.base_url,
                is_fim_request,
                request.state.detected_client,
            )
