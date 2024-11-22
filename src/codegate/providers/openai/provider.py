import json

from fastapi import Header, HTTPException, Request

from ..base import BaseProvider
from ..litellmshim.litellmshim import LiteLLmShim
from .adapter import OpenAIAdapter


class OpenAIProvider(BaseProvider):
    def __init__(self):
        adapter = OpenAIAdapter()
        completion_handler = LiteLLmShim(adapter)
        super().__init__(completion_handler)

    def _setup_routes(self):
        """
        Sets up the /chat/completions route for the provider as expected by the
        OpenAI API. Extracts the API key from the "Authorization" header and
        passes it to the completion handler.
        """
        @self.router.post("/chat/completions")
        async def create_completion(
            request: Request,
            authorization: str = Header(..., description="Bearer token"),
        ):
            if not authorization.startswith("Bearer "):
                raise HTTPException(
                    status_code=401, detail="Invalid authorization header"
                )

            api_key = authorization.split(" ")[1]
            body = await request.body()
            data = json.loads(body)

            stream = await self.complete(data, api_key)
            return self._completion_handler.create_streaming_response(stream)
