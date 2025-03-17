from pathlib import Path
from typing import Callable, List

import structlog
from fastapi import HTTPException, Request

from codegate.clients.clients import ClientType
from codegate.clients.detector import DetectClient
from codegate.config import Config
from codegate.pipeline.factory import PipelineFactory
from codegate.providers.base import BaseProvider, ModelFetchError
from codegate.providers.fim_analyzer import FIMAnalyzer
from codegate.providers.llamacpp.completion_handler import LlamaCppCompletionHandler
from codegate.types.openai import (
    ChatCompletionRequest,
    LegacyCompletionRequest,
)

logger = structlog.get_logger("codegate")


class LlamaCppProvider(BaseProvider):
    def __init__(
        self,
        pipeline_factory: PipelineFactory,
    ):
        if self._get_base_url() != "":
            self.base_url = self._get_base_url()
        else:
            self.base_url = "./codegate_volume/models"
        completion_handler = LlamaCppCompletionHandler()
        super().__init__(
            None,
            None,
            completion_handler,
            pipeline_factory,
        )

    @property
    def provider_route_name(self) -> str:
        return "llamacpp"

    def models(self, endpoint: str = None, api_key: str = None) -> List[str]:
        models_path = Path(Config.get_config().model_base_path)
        if not models_path.is_dir():
            raise ModelFetchError(f"llamacpp model path does not exist: {models_path}")

        # get all models except the all-minilm-L6-v2-q5_k_m model which we use for embeddings
        found_models = [
            model.stem
            for model in models_path.glob("*.gguf")
            if model.is_file() and model.stem != "all-minilm-L6-v2-q5_k_m"
        ]
        if len(found_models) == 0:
            raise ModelFetchError("Not found models for llamacpp provider")

        return found_models

    async def process_request(
        self,
        data: dict,
        api_key: str,
        base_url: str,
        is_fim_request: bool,
        client_type: ClientType,
        completion_handler: Callable | None = None,
        stream_generator: Callable | None = None,
    ):
        try:
            stream = await self.complete(
                data,
                None,
                base_url,
                is_fim_request=is_fim_request,
                client_type=client_type,
                completion_handler=completion_handler,
            )
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
        return self._completion_handler.create_response(
            stream,
            client_type,
            stream_generator=stream_generator,
        )

    def _setup_routes(self):
        """
        Sets up the /completions and /chat/completions routes for the
        provider as expected by the Llama API.
        """

        @self.router.post(f"/{self.provider_route_name}/completions")
        @DetectClient()
        async def completions(
            request: Request,
        ):
            body = await request.body()
            req = LegacyCompletionRequest.model_validate_json(body)
            is_fim_request = FIMAnalyzer.is_fim_request(request.url.path, req)
            return await self.process_request(
                req,
                None,
                self.base_url,
                is_fim_request,
                request.state.detected_client,
            )

        @self.router.post(f"/{self.provider_route_name}/chat/completions")
        @DetectClient()
        async def chat_completion(
            request: Request,
        ):
            body = await request.body()
            req = ChatCompletionRequest.model_validate_json(body)
            is_fim_request = FIMAnalyzer.is_fim_request(request.url.path, req)
            return await self.process_request(
                req,
                None,
                self.base_url,
                is_fim_request,
                request.state.detected_client,
            )
