import json
from typing import Dict

from fastapi import Header, HTTPException, Request

from codegate.clients.clients import ClientType
from codegate.clients.detector import DetectClient
from codegate.pipeline.factory import PipelineFactory
from codegate.providers.fim_analyzer import FIMAnalyzer
from codegate.providers.openai import OpenAIProvider
from codegate.types.openai import (
        ChatCompletionRequest,
)


class OpenRouterProvider(OpenAIProvider):
    def __init__(self, pipeline_factory: PipelineFactory):
        super().__init__(pipeline_factory)
        if self._get_base_url() != "":
            self.base_url = self._get_base_url()
        else:
            self.base_url = "https://openrouter.ai/api"

    @property
    def provider_route_name(self) -> str:
        return "openrouter"

    async def process_request(
        self,
        data: dict,
        api_key: str,
        base_url: str,
        is_fim_request: bool,
        client_type: ClientType,
    ):
        return await super().process_request(data, api_key, base_url, is_fim_request, client_type)

    def _setup_routes(self):
        @self.router.post(f"/{self.provider_route_name}/api/v1/chat/completions")
        @self.router.post(f"/{self.provider_route_name}/chat/completions")
        @self.router.post(f"/{self.provider_route_name}/completions")
        @DetectClient()
        async def create_completion(
            request: Request,
            authorization: str = Header(..., description="Bearer token"),
        ):
            if not authorization.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Invalid authorization header")

            api_key = authorization.split(" ")[1]
            body = await request.body()

            req = ChatCompletionRequest.model_validate_json(body)
            is_fim_request = FIMAnalyzer.is_fim_request(request.url.path, req)

            return await self.process_request(
                req,
                api_key,
                self.base_url,
                is_fim_request,
                request.state.detected_client,
            )
