import json
from typing import Dict

from fastapi import Header, HTTPException, Request
from litellm import atext_completion
from litellm.types.llms.openai import ChatCompletionRequest

from codegate.clients.clients import ClientType
from codegate.clients.detector import DetectClient
from codegate.pipeline.factory import PipelineFactory
from codegate.providers.fim_analyzer import FIMAnalyzer
from codegate.providers.litellmshim import LiteLLmShim, sse_stream_generator
from codegate.providers.normalizer.completion import CompletionNormalizer
from codegate.providers.openai import OpenAIProvider


class OpenRouterNormalizer(CompletionNormalizer):
    def __init__(self):
        super().__init__()

    def normalize(self, data: Dict) -> ChatCompletionRequest:
        return super().normalize(data)

    def denormalize(self, data: ChatCompletionRequest) -> Dict:
        """
        Denormalize a FIM OpenRouter request. Force it to be an accepted atext_completion format.
        """
        denormalized_data = super().denormalize(data)
        # We are forcing atext_completion which expects to have a "prompt" key in the data
        # Forcing it in case is not present
        if "prompt" in data:
            return denormalized_data
        custom_prompt = ""
        for msg_dict in denormalized_data.get("messages", []):
            content_obj = msg_dict.get("content")
            if not content_obj:
                continue
            if isinstance(content_obj, list):
                for content_dict in content_obj:
                    custom_prompt += (
                        content_dict.get("text", "") if isinstance(content_dict, dict) else ""
                    )
            elif isinstance(content_obj, str):
                custom_prompt += content_obj

        # Erase the original "messages" key. Replace it by "prompt"
        del denormalized_data["messages"]
        denormalized_data["prompt"] = custom_prompt

        return denormalized_data


class OpenRouterProvider(OpenAIProvider):
    def __init__(self, pipeline_factory: PipelineFactory):
        super().__init__(
            pipeline_factory,
            # We get FIM requests in /completions. LiteLLM is forcing /chat/completions
            # which returns "choices":[{"delta":{"content":"some text"}}]
            # instead of "choices":[{"text":"some text"}] expected by the client (Continue)
            completion_handler=LiteLLmShim(
                stream_generator=sse_stream_generator, fim_completion_func=atext_completion
            ),
        )
        self._fim_normalizer = OpenRouterNormalizer()

    @property
    def provider_route_name(self) -> str:
        return "openrouter"

    async def process_request(
        self,
        data: dict,
        api_key: str,
        is_fim_request: bool,
        client_type: ClientType,
    ):
        # litellm workaround - add openrouter/ prefix to model name to make it openai-compatible
        # once we get rid of litellm, this can simply be removed
        original_model = data.get("model", "")
        if not original_model.startswith("openrouter/"):
            data["model"] = f"openrouter/{original_model}"

        return await super().process_request(data, api_key, is_fim_request, client_type)

    def _setup_routes(self):
        @self.router.post(f"/{self.provider_route_name}/api/v1/chat/completions")
        @self.router.post(f"/{self.provider_route_name}/chat/completions")
        @self.router.post(f"/{self.provider_route_name}/completions")
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
            is_fim_request = FIMAnalyzer.is_fim_request(request.url.path, data)

            return await self.process_request(
                data,
                api_key,
                is_fim_request,
                request.state.detected_client,
            )
