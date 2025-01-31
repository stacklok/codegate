import json

from fastapi import Header, HTTPException, Request
from fastapi.responses import JSONResponse

from codegate.config import Config
from codegate.pipeline.factory import PipelineFactory
from codegate.providers.openai.provider import OpenAIProvider


class LmStudioProvider(OpenAIProvider):
    def __init__(
        self,
        pipeline_factory: PipelineFactory,
    ):
        config = Config.get_config()
        if config is not None:
            provided_urls = config.provider_urls
            self.lm_studio_url = provided_urls.get("lm_studio", "http://localhost:11434/")

        super().__init__(pipeline_factory)

    @property
    def provider_route_name(self) -> str:
        return "lm_studio"

    def _setup_routes(self):
        """
        Sets up the /chat/completions route for the provider as expected by the
        LM Studio API. Extracts the API key from the "Authorization" header and
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

            data["base_url"] = self.lm_studio_url + "/v1/"

            return await self.process_request(data, api_key, request.url.path)
