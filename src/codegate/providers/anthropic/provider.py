import json

from fastapi import Header, HTTPException, Request

from codegate.providers.anthropic.adapter import AnthropicInputNormalizer, AnthropicOutputNormalizer
from codegate.providers.base import BaseProvider
from codegate.providers.litellmshim import LiteLLmShim, anthropic_stream_generator


class AnthropicProvider(BaseProvider):
    def __init__(self, pipeline_processor=None):
        completion_handler = LiteLLmShim(stream_generator=anthropic_stream_generator)
        super().__init__(
            AnthropicInputNormalizer(),
            AnthropicOutputNormalizer(),
            completion_handler,
            pipeline_processor,
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

            stream = await self.complete(data, x_api_key)
            return self._completion_handler.create_streaming_response(stream)
