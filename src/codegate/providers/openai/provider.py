import json

import structlog
from fastapi import Header, HTTPException, Request
from fastapi.responses import JSONResponse

from codegate.config import Config
from codegate.pipeline.factory import PipelineFactory
from codegate.providers.base import BaseProvider
from codegate.providers.litellmshim import LiteLLmShim, sse_stream_generator
from codegate.providers.openai.adapter import OpenAIInputNormalizer, OpenAIOutputNormalizer


class OpenAIProvider(BaseProvider):
    def __init__(
        self,
        pipeline_factory: PipelineFactory,
    ):
        completion_handler = LiteLLmShim(stream_generator=sse_stream_generator)
        config = Config.get_config()
        if config is not None:
            provided_urls = config.provider_urls
            self.lm_studio_url = provided_urls.get("lm_studio", "http://localhost:11434/")

        super().__init__(
            OpenAIInputNormalizer(),
            OpenAIOutputNormalizer(),
            completion_handler,
            pipeline_factory,
        )

    @property
    def provider_route_name(self) -> str:
        return "openai"

    def _setup_routes(self):
        """
        Sets up the /chat/completions route for the provider as expected by the
        OpenAI API. Extracts the API key from the "Authorization" header and
        passes it to the completion handler.
        """

        @self.router.get(f"/{self.provider_route_name}/models")
        @self.router.get(f"/{self.provider_route_name}/v1/models")
        async def get_models():
            # dummy method for lm studio
            return JSONResponse(status_code=200, content=[])

        @self.router.post(f"/{self.provider_route_name}/chat/completions")
        @self.router.post(f"/{self.provider_route_name}/completions")
        @self.router.post(f"/{self.provider_route_name}/v1/chat/completions")
        async def create_completion(
            request: Request,
            authorization: str = Header(..., description="Bearer token"),
        ):
            if not authorization.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Invalid authorization header")

            api_key = authorization.split(" ")[1]
            body = await request.body()
            data = json.loads(body)

            # if model starts with lm_studio, propagate it
            if data.get("model", "").startswith("lm_studio"):
                data["base_url"] = self.lm_studio_url + "/v1/"
            is_fim_request = self._is_fim_request(request, data)
            try:
                stream = await self.complete(data, api_key, is_fim_request=is_fim_request)
            except Exception as e:
                #  check if we have an status code there
                if hasattr(e, "status_code"):
                    logger = structlog.get_logger("codegate")
                    logger.error("Error in OpenAIProvider completion", error=str(e))

                    raise HTTPException(status_code=e.status_code, detail=str(e))  # type: ignore
                else:
                    # just continue raising the exception
                    raise e
            return self._completion_handler.create_response(stream)
