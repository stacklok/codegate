import json
from typing import List

import httpx
import structlog
from fastapi import HTTPException, Request

from codegate.config import Config
from codegate.pipeline.factory import PipelineFactory
from codegate.providers.base import BaseProvider, ModelFetchError
from codegate.providers.ollama.adapter import OllamaInputNormalizer, OllamaOutputNormalizer
from codegate.providers.ollama.completion_handler import OllamaShim

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
        completion_handler = OllamaShim(self.base_url)
        super().__init__(
            OllamaInputNormalizer(),
            OllamaOutputNormalizer(),
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

    async def process_request(self, data: dict, api_key: str, request_url_path: str):
        is_fim_request = self._is_fim_request(request_url_path, data)
        try:
            stream = await self.complete(data, api_key=None, is_fim_request=is_fim_request)
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
        return self._completion_handler.create_response(stream)

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
        async def show_model(request: Request):
            """
            route for /api/show that responds outside of the pipeline
            /api/show displays model is used to get the model information
            https://github.com/ollama/ollama/blob/main/docs/api.md#show-model-information
            """
            body = await request.body()
            body_json = json.loads(body)
            if "name" not in body_json:
                raise HTTPException(status_code=400, detail="model is required in the request body")
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/show",
                    content=body,
                    headers={"Content-Type": "application/json; charset=utf-8"},
                )
                return response.json()

        # Native Ollama API routes
        @self.router.post(f"/{self.provider_route_name}/api/chat")
        @self.router.post(f"/{self.provider_route_name}/api/generate")
        # OpenAI-compatible routes for backward compatibility
        @self.router.post(f"/{self.provider_route_name}/chat/completions")
        @self.router.post(f"/{self.provider_route_name}/completions")
        # Cline API routes
        @self.router.post(f"/{self.provider_route_name}/v1/chat/completions")
        @self.router.post(f"/{self.provider_route_name}/v1/generate")
        async def create_completion(request: Request):
            body = await request.body()
            data = json.loads(body)

            # `base_url` is used in the providers pipeline to do the packages lookup.
            # Force it to be the one that comes in the configuration.
            data["base_url"] = self.base_url

            return await self.process_request(data, None, request.url.path)
