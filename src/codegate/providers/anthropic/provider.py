import json
from typing import Optional

from fastapi import Header, HTTPException, Request

from codegate.pipeline.base import SequentialPipelineProcessor
from codegate.pipeline.output import OutputPipelineProcessor
from codegate.providers.anthropic.adapter import AnthropicInputNormalizer, AnthropicOutputNormalizer
from codegate.providers.anthropic.completion_handler import AnthropicCompletion
from codegate.providers.base import BaseProvider
from codegate.providers.litellmshim import anthropic_stream_generator


class AnthropicProvider(BaseProvider):
    def __init__(
        self,
        pipeline_processor: Optional[SequentialPipelineProcessor] = None,
        fim_pipeline_processor: Optional[SequentialPipelineProcessor] = None,
        output_pipeline_processor: Optional[OutputPipelineProcessor] = None,
    ):
        completion_handler = AnthropicCompletion(stream_generator=anthropic_stream_generator)
        super().__init__(
            AnthropicInputNormalizer(),
            AnthropicOutputNormalizer(),
            completion_handler,
            pipeline_processor,
            fim_pipeline_processor,
            output_pipeline_processor,
        )

    @property
    def provider_route_name(self) -> str:
        return "anthropic"

    def _setup_routes(self):
        """
        Sets up the /messages route for the provider as expected by the Anthropic
        API. Extracts the API key from the "x-api-key" header and passes it to the
        completion handler.
        """

        @self.router.post(f"/{self.provider_route_name}/messages")
        async def create_message(
            request: Request,
            x_api_key: str = Header(None),
        ):
            if x_api_key == "":
                raise HTTPException(status_code=401, detail="No API key provided")

            body = await request.body()
            data = json.loads(body)

            is_fim_request = self._is_fim_request(request, data)
            stream = await self.complete(data, x_api_key, is_fim_request)
            return self._completion_handler.create_response(stream)
