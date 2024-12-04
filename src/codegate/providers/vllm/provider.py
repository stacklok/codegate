import json
from typing import Optional

import httpx
from fastapi import Header, HTTPException, Request
from litellm import atext_completion

from codegate.config import Config
from codegate.pipeline.output import OutputPipelineProcessor
from codegate.pipeline.secrets.manager import SecretsManager
from codegate.providers.base import BaseProvider, SequentialPipelineProcessor
from codegate.providers.litellmshim import LiteLLmShim, sse_stream_generator
from codegate.providers.vllm.adapter import VLLMInputNormalizer, VLLMOutputNormalizer


class VLLMProvider(BaseProvider):
    def __init__(
        self,
        secrets_manager: SecretsManager,
        pipeline_processor: Optional[SequentialPipelineProcessor] = None,
        fim_pipeline_processor: Optional[SequentialPipelineProcessor] = None,
        output_pipeline_processor: Optional[OutputPipelineProcessor] = None,
    ):
        completion_handler = LiteLLmShim(
            stream_generator=sse_stream_generator, fim_completion_func=atext_completion
        )
        super().__init__(
            secrets_manager,
            VLLMInputNormalizer(),
            VLLMOutputNormalizer(),
            completion_handler,
            pipeline_processor,
            fim_pipeline_processor,
            output_pipeline_processor,
        )

    @property
    def provider_route_name(self) -> str:
        return "vllm"

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
            base_url = config.provider_urls.get("vllm")

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
            data["base_url"] = config.provider_urls.get("vllm")

            is_fim_request = self._is_fim_request(request, data)
            stream = await self.complete(data, api_key, is_fim_request=is_fim_request)
            return self._completion_handler.create_response(stream)
