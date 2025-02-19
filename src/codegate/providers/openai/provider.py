import json
from typing import List

import httpx
import structlog
from fastapi import Header, HTTPException, Request

from codegate.clients.clients import ClientType
from codegate.clients.detector import DetectClient
from codegate.pipeline.factory import PipelineFactory
from codegate.providers.base import BaseProvider, ModelFetchError
from codegate.providers.fim_analyzer import FIMAnalyzer
from codegate.providers.litellmshim import LiteLLmShim
from codegate.types.openai import (
    completions_streaming,
    stream_generator,
    ChatCompletionRequest,
)


logger = structlog.get_logger("codegate")


class OpenAIProvider(BaseProvider):
    def __init__(
        self,
        pipeline_factory: PipelineFactory,
    ):
        if self._get_base_url() != "":
            self.base_url = self._get_base_url()
        else:
            self.base_url = "https://api.openai.com/api/v1"

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
        return "openai"

    def models(self, endpoint: str = None, api_key: str = None) -> List[str]:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        resp = httpx.get(f"{endpoint}/v1/models", headers=headers)

        if resp.status_code != 200:
            raise ModelFetchError(f"Failed to fetch models from OpenAI API: {resp.text}")

        jsonresp = resp.json()

        return [model["id"] for model in jsonresp.get("data", [])]

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
                api_key,
                is_fim_request=is_fim_request,
                client_type=client_type,
            )
        except Exception as e:
            # Check if we have an status code there
            if hasattr(e, "status_code"):
                logger.error("Error in OpenAIProvider completion", error=str(e))

                raise HTTPException(status_code=e.status_code, detail=str(e))  # type: ignore
            else:
                # just continue raising the exception
                raise e
        return self._completion_handler.create_response(stream, client_type)

    def _setup_routes(self):
        """
        Sets up the /chat/completions route for the provider as expected by the
        OpenAI API. Extracts the API key from the "Authorization" header and
        passes it to the completion handler.
        """

        @self.router.post(f"/{self.provider_route_name}/chat/completions")
        @self.router.post(f"/{self.provider_route_name}/completions")
        @self.router.post(f"/{self.provider_route_name}/v1/chat/completions")
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

            if not req.stream:
                logger.warn("We got a non-streaming request, forcing to a streaming one")
                req.stream = True

            return await self.process_request(
                req,
                api_key,
                is_fim_request,
                request.state.detected_client,
            )
