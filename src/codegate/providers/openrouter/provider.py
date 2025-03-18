from typing import Callable

from fastapi import Header, HTTPException, Request

from codegate.clients.clients import ClientType
from codegate.clients.detector import DetectClient
from codegate.pipeline.factory import PipelineFactory
from codegate.providers.fim_analyzer import FIMAnalyzer
from codegate.providers.litellmshim import LiteLLmShim
from codegate.providers.openai import OpenAIProvider
from codegate.types.openai import (
    ChatCompletion,
    ChatCompletionRequest,
    LegacyCompletion,
    LegacyCompletionRequest,
    completions_streaming,
    stream_generator,
    streaming,
)


async def generate_streaming(request, api_key, base_url):
    if base_url is None:
        base_url = "https://api.openai.com"

    url = f"{base_url}/v1/chat/completions"
    cls = ChatCompletion
    if isinstance(request, LegacyCompletionRequest):
        cls = LegacyCompletion

    async for item in streaming(request, api_key, url, cls):
        yield item


class OpenRouterProvider(OpenAIProvider):
    def __init__(self, pipeline_factory: PipelineFactory):
        completion_handler = LiteLLmShim(
            completion_func=completions_streaming,
            fim_completion_func=generate_streaming,
            stream_generator=stream_generator,
        )
        super().__init__(pipeline_factory, completion_handler)
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
        completion_handler: Callable | None = None,
        stream_generator: Callable | None = None,
    ):
        return await super().process_request(
            data,
            api_key,
            base_url,
            is_fim_request,
            client_type,
            completion_handler=completion_handler,
            stream_generator=stream_generator,
        )

    def _setup_routes(self):
        @self.router.post(f"/{self.provider_route_name}/completions")
        @DetectClient()
        async def completions(
            request: Request,
            authorization: str = Header(..., description="Bearer token"),
        ):
            if not authorization.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Invalid authorization header")

            api_key = authorization.split(" ")[1]
            body = await request.body()

            req = LegacyCompletionRequest.model_validate_json(body)
            is_fim_request = FIMAnalyzer.is_fim_request(request.url.path, req)

            return await self.process_request(
                req,
                api_key,
                self.base_url,
                is_fim_request,
                request.state.detected_client,
            )

        @self.router.post(f"/{self.provider_route_name}/api/v1/chat/completions")
        @self.router.post(f"/{self.provider_route_name}/chat/completions")
        @DetectClient()
        async def chat_completion(
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
