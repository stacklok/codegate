import json
import os
from typing import List

import httpx
import structlog
from fastapi import Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from codegate.clients.clients import ClientType
from codegate.clients.detector import DetectClient
from codegate.pipeline.factory import PipelineFactory
from codegate.providers.anthropic.completion_handler import AnthropicCompletion
from codegate.providers.base import BaseProvider, ModelFetchError
from codegate.providers.fim_analyzer import FIMAnalyzer
from codegate.types.anthropic import stream_generator
from codegate.types.anthropic import ChatCompletionRequest


logger = structlog.get_logger("codegate")


class AnthropicProvider(BaseProvider):
    def __init__(
        self,
        pipeline_factory: PipelineFactory,
    ):
        if self._get_base_url() != "":
            self.base_url = self._get_base_url()
        else:
            self.base_url = "https://api.anthropic.com/v1"

        completion_handler = AnthropicCompletion(stream_generator=stream_generator)
        super().__init__(
            None,
            None,
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

    async def process_request(
        self,
        data: dict,
        api_key: str,
        base_url: str,
        is_fim_request: bool,
        client_type: ClientType,
    ):
        try:
            stream = await self.complete(data, api_key, base_url, is_fim_request, client_type)
        except Exception as e:
            # check if we have an status code there
            if hasattr(e, "status_code"):
                # log the exception
                logger.exception("Error in AnthropicProvider completion")
                raise HTTPException(status_code=e.status_code, detail=str(e))  # type: ignore
            else:
                # just continue raising the exception
                raise e
        return self._completion_handler.create_response(stream, client_type)

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
        @DetectClient()
        async def create_message(
            request: Request,
            x_api_key: str = Header(None),
        ):
            if x_api_key == "":
                raise HTTPException(status_code=401, detail="No API key provided")

            body = await request.body()

            if os.getenv("CODEGATE_DEBUG_ANTHROPIC") is not None:
                print(f"{create_message.__name__}: {body}")

            req = ChatCompletionRequest.model_validate_json(body)
            is_fim_request = FIMAnalyzer.is_fim_request(request.url.path, req)

            return await self.process_request(
                req,
                x_api_key,
                base_url,
                is_fim_request,
                request.state.detected_client,
            )


async def dumper(stream):
    print("==========")
    async for event in stream:
        res = f"event: {event.type}\ndata: {event.json(exclude_defaults=True, exclude_unset=True)}\n"
        print(res)
        yield res
    print("==========")
