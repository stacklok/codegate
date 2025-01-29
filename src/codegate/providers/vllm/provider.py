import json
from urllib.parse import urljoin
from typing import List

import httpx
import structlog
from fastapi import Header, HTTPException, Request
from litellm import atext_completion

from codegate.config import Config
from codegate.pipeline.factory import PipelineFactory
from codegate.providers.base import BaseProvider, ModelFetchError
from codegate.providers.litellmshim import LiteLLmShim, sse_stream_generator
from codegate.providers.vllm.adapter import VLLMInputNormalizer, VLLMOutputNormalizer


class VLLMProvider(BaseProvider):
    def __init__(
        self,
        pipeline_factory: PipelineFactory,
    ):
        completion_handler = LiteLLmShim(
            stream_generator=sse_stream_generator, fim_completion_func=atext_completion
        )
        super().__init__(
            VLLMInputNormalizer(),
            VLLMOutputNormalizer(),
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
        config = Config.get_config()
        base_url = config.provider_urls.get("vllm") if config else ""
        if base_url:
            base_url = base_url.rstrip("/")
            # Add /v1 if not present
            if not base_url.endswith("/v1"):
                base_url = f"{base_url}/v1"
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
                logger = structlog.get_logger("codegate")
                logger.error("Error fetching vLLM models", error=str(e))
                raise HTTPException(
                    status_code=e.response.status_code if hasattr(e, "response") else 500,
                    detail=str(e),
                )

        @self.router.post(f"/{self.provider_route_name}/chat/completions")
        @self.router.post(f"/{self.provider_route_name}/completions")
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
            data = json.loads(body)

            # Add the vLLM base URL to the request
            base_url = self._get_base_url()
            data["base_url"] = base_url

            is_fim_request = self._is_fim_request(request, data)
            try:
                # Pass the potentially None api_key to complete
                stream = await self.complete(data, api_key, is_fim_request=is_fim_request)
            except Exception as e:
                # Check if we have a status code there
                if hasattr(e, "status_code"):
                    logger = structlog.get_logger("codegate")
                    logger.error("Error in VLLMProvider completion", error=str(e))
                    raise HTTPException(status_code=e.status_code, detail=str(e))
                raise e

            return self._completion_handler.create_response(stream)
