import json

import httpx
import structlog
from fastapi import Header, HTTPException, Request
from litellm import atext_completion

from codegate.config import Config
from codegate.pipeline.factory import PipelineFactory
from codegate.providers.base import BaseProvider
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

    def models(self):
        resp = httpx.get(f"{self.base_url}/v1/models")
        jsonresp = resp.json()

        return [model["id"] for model in jsonresp.get("data", [])]

    def _setup_routes(self):
        """
        Sets up the /chat/completions route for the provider as expected by the
        OpenAI API. Extracts the API key from the "Authorization" header and
        passes it to the completion handler.
        """

        @self.router.get(f"/{self.provider_route_name}/models")
        async def get_models(authorization: str = Header(..., description="Bearer token")):
            if not authorization.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Invalid authorization header")

            token = authorization.split(" ")[1]
            config = Config.get_config()
            if config:
                base_url = config.provider_urls.get("vllm")
            else:
                base_url = ""

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/v1/models", headers={"Authorization": f"Bearer {token}"}
                )
                response.raise_for_status()
                return response.json()

        @self.router.post(f"/{self.provider_route_name}/chat/completions")
        @self.router.post(f"/{self.provider_route_name}/completions")
        async def create_completion(
            request: Request,
            authorization: str = Header(..., description="Bearer token"),
        ):
            if not authorization.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Invalid authorization header")

            api_key = authorization.split(" ")[1]
            body = await request.body()
            data = json.loads(body)

            # Add the vLLM base URL to the request
            config = Config.get_config()
            if config:
                data["base_url"] = config.provider_urls.get("vllm")
            else:
                data["base_url"] = ""

            is_fim_request = self._is_fim_request(request, data)
            try:
                stream = await self.complete(data, api_key, is_fim_request=is_fim_request)
            except Exception as e:
                #  check if we have an status code there
                if hasattr(e, "status_code"):
                    logger = structlog.get_logger("codegate")
                    logger.error("Error in VLLMProvider completion", error=str(e))

                    raise HTTPException(status_code=e.status_code, detail=str(e))  # type: ignore
                else:
                    # just continue raising the exception
                    raise e
            return self._completion_handler.create_response(stream)
