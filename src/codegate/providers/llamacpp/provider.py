import json
from typing import Optional

from fastapi import Request

from codegate.pipeline.base import SequentialPipelineProcessor
from codegate.pipeline.output import OutputPipelineProcessor
from codegate.providers.base import BaseProvider
from codegate.providers.llamacpp.completion_handler import LlamaCppCompletionHandler
from codegate.providers.llamacpp.normalizer import LLamaCppInputNormalizer, LLamaCppOutputNormalizer


class LlamaCppProvider(BaseProvider):
    def __init__(
        self,
        pipeline_processor: Optional[SequentialPipelineProcessor] = None,
        fim_pipeline_processor: Optional[SequentialPipelineProcessor] = None,
        output_pipeline_processor: Optional[OutputPipelineProcessor] = None,
    ):
        completion_handler = LlamaCppCompletionHandler()
        super().__init__(
            LLamaCppInputNormalizer(),
            LLamaCppOutputNormalizer(),
            completion_handler,
            pipeline_processor,
            fim_pipeline_processor,
            output_pipeline_processor,
        )

    @property
    def provider_route_name(self) -> str:
        return "llamacpp"

    def _setup_routes(self):
        """
        Sets up the /completions and /chat/completions routes for the
        provider as expected by the Llama API.
        """

        @self.router.post(f"/{self.provider_route_name}/completions")
        @self.router.post(f"/{self.provider_route_name}/chat/completions")
        async def create_completion(
            request: Request,
        ):
            body = await request.body()
            data = json.loads(body)

            is_fim_request = self._is_fim_request(request, data)
            stream = await self.complete(data, None, is_fim_request=is_fim_request)
            return self._completion_handler.create_response(stream)
