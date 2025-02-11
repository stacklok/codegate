import json
from typing import List, Optional

import httpx
import structlog
from fastapi import Header, HTTPException, Request

from codegate.clients.clients import ClientType
from codegate.clients.detector import DetectClient
from codegate.config import Config
from codegate.pipeline.factory import PipelineFactory
from codegate.providers.base import BaseProvider, ModelFetchError
from codegate.providers.fim_analyzer import FIMAnalyzer
from codegate.providers.ollama.adapter import OllamaInputNormalizer, OllamaOutputNormalizer
from codegate.providers.ollama.completion_handler import OllamaShim
from codegate.types.openai import ChatCompletionRequest
from codegate.types.ollama import ChatRequest, GenerateRequest


logger = structlog.get_logger("codegate")


class OllamaProvider(BaseProvider):
    def __init__(
        self,
        pipeline_factory: PipelineFactory,
    ):
        config = Config.get_config()
        if config is None:
            provided_urls = {}
        else:
            provided_urls = config.provider_urls
        self.base_url = provided_urls.get("ollama", "http://localhost:11434/")
        completion_handler = OllamaShim()
        super().__init__(
            None,
            None,
            completion_handler,
            pipeline_factory,
        )

    @property
    def provider_route_name(self) -> str:
        return "ollama"

    def models(self, endpoint: str = None, api_key: str = None) -> List[str]:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if not endpoint:
            endpoint = self.base_url
        resp = httpx.get(
            f"{endpoint}/api/tags",
            headers=headers,
        )

        if resp.status_code != 200:
            raise ModelFetchError(f"Failed to fetch models from Ollama API: {resp.text}")

        jsonresp = resp.json()

        return [model["name"] for model in jsonresp.get("models", [])]

    async def process_request(
        self,
        data: dict,
        api_key: str,
        is_fim_request: bool,
        client_type: ClientType,
    ):
        try:
            stream = await self.complete(
                data,
                api_key=api_key,
                is_fim_request=is_fim_request,
                client_type=client_type,
            )
        except httpx.ConnectError as e:
            logger.error("Error in OllamaProvider completion", error=str(e))
            raise HTTPException(status_code=503, detail="Ollama service is unavailable")
        except Exception as e:
            #  check if we have an status code there
            if hasattr(e, "status_code"):
                # log the exception
                logger.error("Error in OllamaProvider completion", error=str(e))
                raise HTTPException(status_code=e.status_code, detail=str(e))  # type: ignore
            else:
                # just continue raising the exception
                raise e
        return self._completion_handler.create_response(stream, client_type)

    def _setup_routes(self):
        """
        Sets up Ollama API routes.
        """

        @self.router.get(f"/{self.provider_route_name}/api/tags")
        async def get_tags(request: Request):
            """
            Special route for /api/tags that responds outside of the pipeline
            Tags are used to get the list of models
            https://github.com/ollama/ollama/blob/main/docs/api.md#list-local-models
            """
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.json()

        @self.router.post(f"/{self.provider_route_name}/api/show")
        async def show_model(
            request: Request,
            authorization: str | None = Header(None, description="Bearer token"),
        ):
            """
            route for /api/show that responds outside of the pipeline
            /api/show displays model is used to get the model information
            https://github.com/ollama/ollama/blob/main/docs/api.md#show-model-information
            """
            api_key = _api_key_from_optional_header_value(authorization)
            body = await request.body()
            body_json = json.loads(body)
            if "name" not in body_json:
                raise HTTPException(status_code=400, detail="model is required in the request body")
            async with httpx.AsyncClient() as client:
                headers = {"Content-Type": "application/json; charset=utf-8"}
                if api_key:
                    headers["Authorization"] = api_key
                response = await client.post(
                    f"{self.base_url}/api/show",
                    content=body,
                    headers=headers,
                )
                return response.json()

        # Native Ollama API routes
        @self.router.post(f"/{self.provider_route_name}/api/generate")
        @DetectClient()
        async def generate(request: Request):
            body = await request.body()
            req = GenerateRequest.model_validate_json(body)
            is_fim_request = FIMAnalyzer.is_fim_request(request.url.path, req)
            return await self.process_request(
                req,
                None,
                is_fim_request,
                request.state.detected_client,
            )

        # Native Ollama API routes
        @self.router.post(f"/{self.provider_route_name}/api/chat")
        @DetectClient()
        async def chat(request: Request):
            body = await request.body()
            req = ChatRequest.model_validate_json(body)
            is_fim_request = FIMAnalyzer.is_fim_request(request.url.path, req)
            return await self.process_request(
                req,
                None,
                is_fim_request,
                request.state.detected_client,
            )

        # OpenAI-compatible routes for backward compatibility
        @self.router.post(f"/{self.provider_route_name}/chat/completions")
        @self.router.post(f"/{self.provider_route_name}/completions")
        # Cline API routes
        @self.router.post(f"/{self.provider_route_name}/v1/chat/completions")
        @self.router.post(f"/{self.provider_route_name}/v1/generate")
        @DetectClient()
        async def create_completion(
            request: Request,
            authorization: str | None = Header(None, description="Bearer token"),
        ):
            api_key = _api_key_from_optional_header_value(authorization)
            body = await request.body()
            # data = json.loads(body)

            # `base_url` is used in the providers pipeline to do the packages lookup.
            # Force it to be the one that comes in the configuration.
            # data["base_url"] = self.base_url
            req = ChatCompletionRequest.model_validate_json(body)
            is_fim_request = FIMAnalyzer.is_fim_request(request.url.path, req)
            return await self.process_request(
                req,
                api_key,
                is_fim_request,
                request.state.detected_client,
            )


def _api_key_from_optional_header_value(val: str) -> str:
    # The header is optional, so if we don't
    # have it, let's just return None
    if not val:
        return None

    # The header value should be "Beaerer <key>"
    if not val.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    vals = val.split(" ")
    if len(vals) != 2:
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    return vals[1]
