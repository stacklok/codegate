import json
from typing import Optional

from fastapi import Request

from codegate.config import Config
from codegate.pipeline.base import SequentialPipelineProcessor
from codegate.pipeline.output import OutputPipelineProcessor
from codegate.providers.base import BaseProvider
from codegate.providers.ollama.adapter import OllamaInputNormalizer, OllamaOutputNormalizer
from codegate.providers.ollama.completion_handler import OllamaCompletionHandler


class OllamaProvider(BaseProvider):
    def __init__(
        self,
        pipeline_processor: Optional[SequentialPipelineProcessor] = None,
        fim_pipeline_processor: Optional[SequentialPipelineProcessor] = None,
        output_pipeline_processor: Optional[OutputPipelineProcessor] = None,
        fim_output_pipeline_processor: Optional[OutputPipelineProcessor] = None,
    ):
        completion_handler = OllamaCompletionHandler()
        super().__init__(
            OllamaInputNormalizer(),
            OllamaOutputNormalizer(),
            completion_handler,
            pipeline_processor,
            fim_pipeline_processor,
            output_pipeline_processor,
        )
        # Get the Ollama base URL
        config = Config.get_config()
        if config is None:
            provided_urls = {}
        else:
            provided_urls = config.provider_urls
        self.base_url = provided_urls.get("ollama", "http://localhost:11434/api")

    @property
    def provider_route_name(self) -> str:
        return "ollama"

    def _setup_routes(self):
        """
        Sets up Ollama API routes.
        """

        # Native Ollama API routes
        @self.router.post(f"/{self.provider_route_name}/api/chat")
        @self.router.post(f"/{self.provider_route_name}/api/generate")
        # OpenAI-compatible routes for backward compatibility
        @self.router.post(f"/{self.provider_route_name}/chat/completions")
        @self.router.post(f"/{self.provider_route_name}/completions")
        async def create_completion(request: Request):
            body = await request.body()
            data = json.loads(body)
            if "base_url" not in data or not data["base_url"]:
                data["base_url"] = self.base_url

            is_fim_request = self._is_fim_request(request, data)
            stream = await self.complete(data, None, is_fim_request=is_fim_request)
            return self._completion_handler.create_response(stream)
