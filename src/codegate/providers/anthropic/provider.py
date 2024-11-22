import json

from fastapi import Header, HTTPException, Request

from ..base import BaseProvider
from ..litellmshim.litellmshim import LiteLLmShim
from .adapter import AnthropicAdapter


class AnthropicProvider(BaseProvider):
    def __init__(self):
        adapter = AnthropicAdapter()
        completion_handler = LiteLLmShim(adapter)
        super().__init__(completion_handler)

    def _setup_routes(self):
        """
        Sets up the /messages route for the provider as expected by the Anthropic
        API. Extracts the API key from the "x-api-key" header and passes it to the
        completion handler.
        """
        @self.router.post("/messages")
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
