import json
from typing import Optional

from fastapi import Header, HTTPException, Request

from codegate.pipeline.base import SequentialPipelineProcessor
from codegate.pipeline.output import OutputPipelineProcessor
from codegate.providers.base import BaseProvider
from codegate.providers.litellmshim import LiteLLmShim, sse_stream_generator
from codegate.providers.openai.adapter import OpenAIInputNormalizer, OpenAIOutputNormalizer


class OpenAIProvider(BaseProvider):
    def __init__(
        self,
        pipeline_processor: Optional[SequentialPipelineProcessor] = None,
        fim_pipeline_processor: Optional[SequentialPipelineProcessor] = None,
        output_pipeline_processor: Optional[OutputPipelineProcessor] = None,
    ):
        completion_handler = LiteLLmShim(stream_generator=sse_stream_generator)
        super().__init__(
            OpenAIInputNormalizer(),
            OpenAIOutputNormalizer(),
            completion_handler,
            pipeline_processor,
            fim_pipeline_processor,
            output_pipeline_processor,
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

        @self.router.post(f"/{self.provider_route_name}/chat/completions")
        @self.router.post(f"/{self.provider_route_name}/completions")
        async def create_completion(
            request: Request,
            authorization: str = Header(..., description="Bearer token"),
        ):
            if not authorization.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Invalid authorization header")

            api_key = authorization.split(" ")[1]
            body = await request.body()
            data = json.loads(body)

            is_fim_request = self._is_fim_request(request, data)
            stream = await self.complete(data, api_key, is_fim_request=is_fim_request)
            return self._completion_handler.create_response(stream)
