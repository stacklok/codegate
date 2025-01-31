import json
from typing import List

import structlog
from fastapi import HTTPException, Request

from codegate.pipeline.factory import PipelineFactory
from codegate.providers.base import BaseProvider
from codegate.providers.llamacpp.completion_handler import LlamaCppCompletionHandler
from codegate.providers.llamacpp.normalizer import LLamaCppInputNormalizer, LLamaCppOutputNormalizer

logger = structlog.get_logger("codegate")


class LlamaCppProvider(BaseProvider):
    def __init__(
        self,
        pipeline_factory: PipelineFactory,
    ):
        completion_handler = LlamaCppCompletionHandler()
        super().__init__(
            LLamaCppInputNormalizer(),
            LLamaCppOutputNormalizer(),
            completion_handler,
            pipeline_factory,
        )

    @property
    def provider_route_name(self) -> str:
        return "llamacpp"

    def models(self, endpoint: str = None, api_key: str = None) -> List[str]:
        # TODO: Implement file fetching
        return []

    async def process_request(self, data: dict, api_key: str, request_url_path: str):
        is_fim_request = self._is_fim_request(request_url_path, data)
        try:
            stream = await self.complete(data, None, is_fim_request=is_fim_request)
        except RuntimeError as e:
            # propagate as error 500
            logger.error("Error in LlamaCppProvider completion", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
        except ValueError as e:
            # capture well known exceptions
            logger.error("Error in LlamaCppProvider completion", error=str(e))
            if str(e).startswith("Model path does not exist") or str(e).startswith("No file found"):
                raise HTTPException(status_code=404, detail=str(e))
            elif "exceed" in str(e):
                raise HTTPException(status_code=429, detail=str(e))
            else:
                # just continue raising the exception
                raise e
        return self._completion_handler.create_response(stream)

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

            return await self.process_request(data, None, request.url.path)
