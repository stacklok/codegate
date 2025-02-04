import json

from fastapi import Header, HTTPException, Request

from codegate.clients.detector import DetectClient
from codegate.pipeline.factory import PipelineFactory
from codegate.providers.openai import OpenAIProvider


class OpenRouterProvider(OpenAIProvider):
    def __init__(self, pipeline_factory: PipelineFactory):
        super().__init__(pipeline_factory)

    @property
    def provider_route_name(self) -> str:
        return "openrouter"

    def _setup_routes(self):
        @self.router.post(f"/{self.provider_route_name}/api/v1/chat/completions")
        @self.router.post(f"/{self.provider_route_name}/chat/completions")
        @DetectClient()
        async def create_completion(
            request: Request,
            authorization: str = Header(..., description="Bearer token"),
        ):
            if not authorization.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Invalid authorization header")

            api_key = authorization.split(" ")[1]
            body = await request.body()
            data = json.loads(body)

            base_url = self._get_base_url()
            data["base_url"] = base_url

            # litellm workaround - add openrouter/ prefix to model name to make it openai-compatible
            # once we get rid of litellm, this can simply be removed
            original_model = data.get("model", "")
            if not original_model.startswith("openrouter/"):
                data["model"] = f"openrouter/{original_model}"

            return await self.process_request(
                data,
                api_key,
                request.url.path,
                request.state.detected_client,
            )
