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
from codegate.providers.gemini.adapter import GeminiInputNormalizer, GeminiOutputNormalizer
from codegate.providers.gemini.completion_handler import GeminiCompletion
from codegate.providers.litellmshim import sse_stream_generator

logger = structlog.get_logger("codegate")


class GeminiProvider(BaseProvider):
    """
    Gemini provider for CodeGate.

    This provider implements the Google Gemini API interface.
    """

    def __init__(
        self,
        pipeline_factory: PipelineFactory,
    ):
        completion_handler = GeminiCompletion(stream_generator=sse_stream_generator)
        super().__init__(
            GeminiInputNormalizer(),
            GeminiOutputNormalizer(),
            completion_handler,
            pipeline_factory,
        )

    @property
    def provider_route_name(self) -> str:
        return "gemini"

    def models(self, endpoint: str = None, api_key: str = None) -> List[str]:
        """
        Fetch available models from the Gemini API.

        The Gemini API uses a different endpoint structure than OpenAI.
        """
        headers = {
            "Content-Type": "application/json",
        }

        if api_key:
            headers["x-goog-api-key"] = api_key

        if not endpoint:
            endpoint = "https://generativelanguage.googleapis.com"

        try:
            resp = httpx.get(
                f"{endpoint}/v1/models",
                headers=headers,
            )

            if resp.status_code != 200:
                raise ModelFetchError(f"Failed to fetch models from Gemini API: {resp.text}")

            respjson = resp.json()

            # Filter for only generative models
            return [
                model["name"].split("/")[-1]
                for model in respjson.get("models", [])
                if "generateContent" in model.get("supportedGenerationMethods", [])
            ]
        except Exception as e:
            logger.error(f"Error fetching Gemini models: {str(e)}")
            raise ModelFetchError(f"Failed to fetch models from Gemini API: {str(e)}")

    async def process_request(
        self,
        data: dict,
        api_key: str,
        is_fim_request: bool,
        client_type: ClientType,
    ):
        """
        Process a request to the Gemini API.
        """
        try:
            stream = await self.complete(data, api_key, is_fim_request, client_type)
        except Exception as e:
            # Check if we have a status code there
            if hasattr(e, "status_code"):
                logger.exception("Error in GeminiProvider completion")
                raise HTTPException(status_code=e.status_code, detail=str(e))
            else:
                # Just continue raising the exception
                raise e
        return self._completion_handler.create_response(stream, client_type)

    def _setup_routes(self):
        """
        Set up the routes for the Gemini API.

        Gemini API has two main endpoints:
        - /generateContent: For generating content
        - /chat/completions: For compatibility with OpenAI-style clients
        """

        @self.router.post(f"/{self.provider_route_name}/models/{{model}}:generateContent")
        @self.router.post(f"/{self.provider_route_name}/models/{{model}}:generateContent")
        @self.router.post(f"/{self.provider_route_name}/models/{{model}}:streamGenerateContent")
        @self.router.post(f"/{self.provider_route_name}/models/{{model}}:streamGenerateContent")
        @self.router.post(f"/{self.provider_route_name}/v1/models/{{model}}:generateContent")
        @self.router.post(f"/{self.provider_route_name}/v1beta/models/{{model}}:generateContent")
        @self.router.post(f"/{self.provider_route_name}/v1/models/{{model}}:streamGenerateContent")
        @self.router.post(
            f"/{self.provider_route_name}/v1beta/models/{{model}}:streamGenerateContent"
        )
        @DetectClient()
        async def generate_content(
            request: Request,
            model: str,
            x_goog_api_key: str = Header(None),
        ):
            """
            Handle requests to the native Gemini API endpoint.
            """
            api_key = x_goog_api_key

            # If no header is provided, check for key query parameter
            if not api_key:
                key_param = request.query_params.get("key")
                if key_param:
                    api_key = key_param

            if not api_key:
                raise HTTPException(
                    status_code=401, detail="No API key provided in header or query parameter"
                )

            body = await request.body()
            data = json.loads(body)

            # Add the model to the data
            data["model"] = model

            # Add a custom field to indicate this is NOT the OpenAI-compatible endpoint
            data["_is_openai_compatible"] = False

            is_fim_request = FIMAnalyzer.is_fim_request(request.url.path, data)

            return await self.process_request(
                data,
                api_key,
                is_fim_request,
                request.state.detected_client,
            )

        @self.router.post(f"/{self.provider_route_name}/chat/completions")
        @self.router.post(f"/{self.provider_route_name}/v1/chat/completions")
        @self.router.post(f"/{self.provider_route_name}/v1beta/openai/chat/completions")
        @DetectClient()
        async def chat_completions(
            request: Request,
            authorization: str = Header(None),
        ):
            """
            Handle requests to the OpenAI-compatible endpoint.

            This allows clients that use the OpenAI API to work with Gemini.
            """
            api_key = None

            # Check for authorization header
            if authorization:
                if not authorization.startswith("Bearer "):
                    raise HTTPException(status_code=401, detail="Invalid authorization header")
                api_key = authorization.split(" ")[1]

            # If no authorization header, check for key query parameter
            if not api_key:
                key_param = request.query_params.get("key")
                if key_param:
                    api_key = key_param

            # If still no API key, raise an error
            if not api_key:
                raise HTTPException(
                    status_code=401, detail="No API key provided in header or query parameter"
                )

            body = await request.body()
            data = json.loads(body)

            # Add a custom field to indicate this is the OpenAI-compatible endpoint
            data["_is_openai_compatible"] = "v1beta/openai" in request.url.path

            is_fim_request = FIMAnalyzer.is_fim_request(request.url.path, data)

            return await self.process_request(
                data,
                api_key,
                is_fim_request,
                request.state.detected_client,
            )
