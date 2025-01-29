import json
from typing import List

import httpx
import structlog
from fastapi import Header, HTTPException, Request

from codegate.pipeline.factory import PipelineFactory
from codegate.providers.anthropic.adapter import AnthropicInputNormalizer, AnthropicOutputNormalizer
from codegate.providers.anthropic.completion_handler import AnthropicCompletion
from codegate.providers.base import BaseProvider, ModelFetchError
from codegate.providers.litellmshim import anthropic_stream_generator


class AnthropicProvider(BaseProvider):
    def __init__(
        self,
        pipeline_factory: PipelineFactory,
    ):
        completion_handler = AnthropicCompletion(stream_generator=anthropic_stream_generator)
        super().__init__(
            AnthropicInputNormalizer(),
            AnthropicOutputNormalizer(),
            completion_handler,
            pipeline_factory,
        )

    @property
    def provider_route_name(self) -> str:
        return "anthropic"

    def models(self, endpoint: str = None, api_key: str = None) -> List[str]:
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if api_key:
            headers["x-api-key"] = api_key
        if not endpoint:
            endpoint = "https://api.anthropic.com"

        resp = httpx.get(
            f"{endpoint}/v1/models",
            headers=headers,
        )

        if resp.status_code != 200:
            raise ModelFetchError(f"Failed to fetch models from Anthropic API: {resp.text}")

        respjson = resp.json()

        return [model["id"] for model in respjson.get("data", [])]

    async def process_request(self, data: dict, api_key: str, request_url_path: str):
        is_fim_request = self._is_fim_request(request_url_path, data)
        try:
            stream = await self.complete(data, api_key, is_fim_request)
        except Exception as e:
            #  check if we have an status code there
            if hasattr(e, "status_code"):
                # log the exception
                logger = structlog.get_logger("codegate")
                logger.error("Error in AnthropicProvider completion", error=str(e))
                raise HTTPException(status_code=e.status_code, detail=str(e))  # type: ignore
            else:
                # just continue raising the exception
                raise e
        return self._completion_handler.create_response(stream)

    def _setup_routes(self):
        """
        Sets up the /messages route for the provider as expected by the Anthropic
        API. Extracts the API key from the "x-api-key" header and passes it to the
        completion handler.

        There are two routes:
        - /messages: This is the route that is used by the Anthropic API with Continue.dev
        - /v1/messages: This is the route that is used by the Anthropic API with Cline
        """

        @self.router.post(f"/{self.provider_route_name}/messages")
        @self.router.post(f"/{self.provider_route_name}/v1/messages")
        async def create_message(
            request: Request,
            x_api_key: str = Header(None),
        ):
            if x_api_key == "":
                raise HTTPException(status_code=401, detail="No API key provided")

            body = await request.body()
            data = json.loads(body)

            return await self.process_request(data, x_api_key, request.url.path)
