import datetime
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

import structlog
from fastapi import APIRouter
from litellm import ModelResponse
from litellm.types.llms.openai import ChatCompletionRequest

from codegate.clients.clients import ClientType
from codegate.codegate_logging import setup_logging
from codegate.config import Config
from codegate.db.connection import DbRecorder
from codegate.pipeline.base import (
    PipelineContext,
    PipelineResult,
)
from codegate.pipeline.factory import PipelineFactory
from codegate.pipeline.output import OutputPipelineInstance
from codegate.providers.completion.base import BaseCompletionHandler
from codegate.providers.formatting.input_pipeline import PipelineResponseFormatter
from codegate.providers.normalizer.base import ModelInputNormalizer, ModelOutputNormalizer
from codegate.providers.normalizer.completion import CompletionNormalizer

setup_logging()
logger = structlog.get_logger("codegate")

TEMPDIR = None
if os.getenv("CODEGATE_DUMP_DIR"):
    basedir = os.getenv("CODEGATE_DUMP_DIR")
    TEMPDIR = tempfile.TemporaryDirectory(prefix="codegate-", dir=basedir, delete=False)

StreamGenerator = Callable[[AsyncIterator[Any]], AsyncIterator[str]]


class ModelFetchError(Exception):
    pass


class BaseProvider(ABC):
    """
    The provider class is responsible for defining the API routes and
    calling the completion method using the completion handler.
    """

    def __init__(
        self,
        input_normalizer: ModelInputNormalizer,
        output_normalizer: ModelOutputNormalizer,
        completion_handler: BaseCompletionHandler,
        pipeline_factory: PipelineFactory,
    ):
        self.router = APIRouter()
        self._completion_handler = completion_handler
        self._input_normalizer = input_normalizer
        self._output_normalizer = output_normalizer
        self._pipeline_factory = pipeline_factory
        self._db_recorder = DbRecorder()
        self._pipeline_response_formatter = PipelineResponseFormatter(
            output_normalizer, self._db_recorder
        )
        self._fim_normalizer = CompletionNormalizer()

        self._setup_routes()

    @abstractmethod
    def _setup_routes(self) -> None:
        pass

    @abstractmethod
    def models(self, endpoint, str=None, api_key: str = None) -> List[str]:
        pass

    @abstractmethod
    async def process_request(
        self,
        data: dict,
        api_key: str,
        request_url_path: str,
        client_type: ClientType,
    ):
        pass

    @property
    @abstractmethod
    def provider_route_name(self) -> str:
        pass

    def _get_base_url(self) -> str:
        """
        Get the base URL from config with proper formatting
        """
        config = Config.get_config()
        return config.provider_urls.get(self.provider_route_name) if config else ""

    async def _run_output_stream_pipeline(
        self,
        input_context: PipelineContext,
        model_stream: AsyncIterator[ModelResponse],
        is_fim_request: bool,
    ) -> AsyncIterator[ModelResponse]:
        # Decide which pipeline processor to use
        out_pipeline_processor = None
        if is_fim_request:
            out_pipeline_processor = self._pipeline_factory.create_fim_output_pipeline()
            logger.info("FIM pipeline selected for output.")
        else:
            out_pipeline_processor = self._pipeline_factory.create_output_pipeline()
            logger.info("Chat completion pipeline selected for output.")
        if out_pipeline_processor is None:
            logger.info("No output pipeline processor found, passing through")
            return model_stream

        # HACK! for anthropic we always need to run the output FIM pipeline even
        # if empty to run the normalizers
        if (
            len(out_pipeline_processor.pipeline_steps) == 0
            and self.provider_route_name != "anthropic"
        ):
            logger.info("No output pipeline steps configured, passing through")
            return model_stream

        normalized_stream = self._output_normalizer.normalize_streaming(model_stream)

        output_pipeline_instance = OutputPipelineInstance(
            pipeline_steps=out_pipeline_processor.pipeline_steps,
            input_context=input_context,
        )
        pipeline_output_stream = output_pipeline_instance.process_stream(normalized_stream)
        denormalized_stream = self._output_normalizer.denormalize_streaming(pipeline_output_stream)
        return denormalized_stream

    def _run_output_pipeline(
        self,
        normalized_response: ModelResponse,
    ) -> ModelResponse:
        # we don't have a pipeline for non-streamed output yet
        return normalized_response

    async def _run_input_pipeline(
        self,
        normalized_request: ChatCompletionRequest,
        api_key: Optional[str],
        api_base: Optional[str],
        client_type: ClientType,
        is_fim_request: bool,
    ) -> PipelineResult:
        # Decide which pipeline processor to use
        if is_fim_request:
            pipeline_processor = self._pipeline_factory.create_fim_pipeline(client_type)
            logger.info("FIM pipeline selected for execution.")
            normalized_request = self._fim_normalizer.normalize(normalized_request)
        else:
            pipeline_processor = self._pipeline_factory.create_input_pipeline(client_type)
            logger.info("Chat completion pipeline selected for execution.")
        if pipeline_processor is None:
            return PipelineResult(request=normalized_request)

        result = await pipeline_processor.process_request(
            request=normalized_request,
            provider=self.provider_route_name,
            model=normalized_request.get("model"),
            api_key=api_key,
            api_base=api_base,
        )

        # TODO(jakub): handle this by returning a message to the client
        if result.error_message:
            raise Exception(result.error_message)

        return result

    def _is_fim_request_url(self, request_url_path: str) -> bool:
        """
        Checks the request URL to determine if a request is FIM or chat completion.
        Used by: llama.cpp
        """
        # Evaluate first a larger substring.
        if request_url_path.endswith("/chat/completions"):
            return False

        # /completions is for OpenAI standard. /api/generate is for ollama.
        if request_url_path.endswith("/completions") or request_url_path.endswith("/api/generate"):
            return True

        return False

    def _is_fim_request_body(self, data: Dict) -> bool:
        """
        Determine from the raw incoming data if it's a FIM request.
        Used by: OpenAI and Anthropic
        """
        messages = data.get("messages", [])
        if not messages:
            return False

        first_message_content = messages[0].get("content")
        if first_message_content is None:
            return False

        fim_stop_sequences = ["</COMPLETION>", "<COMPLETION>", "</QUERY>", "<QUERY>"]
        if isinstance(first_message_content, str):
            msg_prompt = first_message_content
        elif isinstance(first_message_content, list):
            msg_prompt = first_message_content[0].get("text", "")
        else:
            logger.warning(f"Could not determine if message was FIM from data: {data}")
            return False
        return all([stop_sequence in msg_prompt for stop_sequence in fim_stop_sequences])

    def _is_fim_request(self, request_url_path: str, data: Dict) -> bool:
        """
        Determine if the request is FIM by the URL or the data of the request.
        """
        # first check if we are in specific tools to discard FIM
        prompt = data.get("prompt", "")
        tools = ["cline", "kodu", "open interpreter"]
        for tool in tools:
            if tool in prompt.lower():
                #  those tools can never be FIM
                return False
        # Avoid more expensive inspection of body by just checking the URL.
        if self._is_fim_request_url(request_url_path):
            return True

        return self._is_fim_request_body(data)

    async def _cleanup_after_streaming(
        self, stream: AsyncIterator[ModelResponse], context: PipelineContext
    ) -> AsyncIterator[ModelResponse]:
        """Wraps the stream to ensure cleanup after consumption"""
        try:
            async for item in stream:
                yield item
        finally:
            if context:
                # Ensure sensitive data is cleaned up after the stream is consumed
                if context.sensitive:
                    context.sensitive.secure_cleanup()

    def _dump_request_response(self, prefix: str, data: Any) -> None:
        """Dump request or response data to a file if CODEGATE_DUMP_DIR is set"""
        if not TEMPDIR:
            return

        ts = datetime.datetime.now()
        fname = (
            Path(TEMPDIR.name)
            / f"{prefix}-{self.provider_route_name}-{ts.strftime('%Y%m%dT%H%M%S%f')}.json"
        )

        if isinstance(data, (dict, list)):
            import json

            with open(fname, "w") as f:
                json.dump(data, f, indent=2)
        else:
            with open(fname, "w") as f:
                f.write(str(data))

    async def complete(
        self,
        data: Dict,
        api_key: Optional[str],
        is_fim_request: bool,
        client_type: ClientType,
    ) -> Union[ModelResponse, AsyncIterator[ModelResponse]]:
        """
        Main completion flow with pipeline integration

        The flow has three main steps:
        - Translate the request to the OpenAI API format used internally
        - Process the request with the pipeline processor. This can modify the request
          or yield a response. The response can either be returned or streamed back to
          the client
        - Execute the completion and translate the response back to the
          provider-specific format
        """
        # Dump the incoming request
        self._dump_request_response("request", data)
        normalized_request = self._input_normalizer.normalize(data)
        # Dump the normalized request
        self._dump_request_response("normalized-request", normalized_request)
        streaming = normalized_request.get("stream", False)

        # Get detected client if available
        input_pipeline_result = await self._run_input_pipeline(
            normalized_request,
            api_key,
            data.get("base_url"),
            client_type,
            is_fim_request,
        )

        if input_pipeline_result.response and input_pipeline_result.context:
            return await self._pipeline_response_formatter.handle_pipeline_response(
                input_pipeline_result.response, streaming, context=input_pipeline_result.context
            )

        if input_pipeline_result.request:
            provider_request = self._input_normalizer.denormalize(input_pipeline_result.request)
        if is_fim_request:
            provider_request = self._fim_normalizer.denormalize(provider_request)  # type: ignore

        self._dump_request_response("provider-request", provider_request)

        # Execute the completion and translate the response
        # This gives us either a single response or a stream of responses
        # based on the streaming flag
        model_response = await self._completion_handler.execute_completion(
            provider_request,
            api_key=api_key,
            stream=streaming,
            is_fim_request=is_fim_request,
        )
        if not streaming:
            normalized_response = self._output_normalizer.normalize(model_response)
            pipeline_output = self._run_output_pipeline(normalized_response)
            await self._db_recorder.record_context(input_pipeline_result.context)
            return self._output_normalizer.denormalize(pipeline_output)

        pipeline_output_stream = await self._run_output_stream_pipeline(
            input_pipeline_result.context, model_response, is_fim_request=is_fim_request  # type: ignore
        )
        return self._cleanup_after_streaming(pipeline_output_stream, input_pipeline_result.context)  # type: ignore

    def get_routes(self) -> APIRouter:
        return self.router
