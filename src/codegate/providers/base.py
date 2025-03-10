import datetime
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

import structlog
from fastapi import APIRouter

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


class PassThroughNormalizer:

    def normalize(self, arg):
        return arg

    def denormalize(self, arg):
        return arg

    def normalize_streaming(self, arg):
        return arg

    def denormalize_streaming(self, arg):
        return arg


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
        self._input_normalizer = input_normalizer if input_normalizer else PassThroughNormalizer()
        self._output_normalizer = (
            output_normalizer if output_normalizer else PassThroughNormalizer()
        )
        self._pipeline_factory = pipeline_factory
        self._db_recorder = DbRecorder()
        self._pipeline_response_formatter = PipelineResponseFormatter(
            output_normalizer, self._db_recorder
        )
        self._fim_normalizer = PassThroughNormalizer()  # CompletionNormalizer()

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
        base_url: str,
        is_fim_request: bool,
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

    async def process_stream_no_pipeline(
        self, stream: AsyncIterator[Any], context: PipelineContext
    ) -> AsyncIterator[Any]:
        """
        Process a stream when there is no pipeline.
        This is needed to record the output stream chunks for FIM.
        """
        try:
            async for chunk in stream:
                context.add_output(chunk)
                yield chunk
        except Exception as e:
            # Log exception and stop processing
            logger.error(f"Error processing stream: {e}")
            raise e
        finally:
            await self._db_recorder.record_context(context)

    async def _run_output_stream_pipeline(
        self,
        input_context: PipelineContext,
        model_stream: AsyncIterator[Any],
        is_fim_request: bool,
    ) -> AsyncIterator[Any]:
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
            return self.process_stream_no_pipeline(model_stream, input_context)

        normalized_stream = self._output_normalizer.normalize_streaming(model_stream)

        output_pipeline_instance = OutputPipelineInstance(
            pipeline_steps=out_pipeline_processor.pipeline_steps,
            input_context=input_context,
        )
        pipeline_output_stream = output_pipeline_instance.process_stream(normalized_stream)
        denormalized_stream = self._output_normalizer.denormalize_streaming(pipeline_output_stream)
        return denormalized_stream

    async def _run_output_pipeline(
        self,
        input_context: PipelineContext,
        model_response: Any,
    ) -> Any:
        """
        Run the output pipeline for a single response.

        For the moment we don't have a pipeline for non-streamed output, so we
        just normalize the response and record the context. It is done here to match
        the behaviour of the streaming pipeline.
        """
        normalized_response = self._output_normalizer.normalize(model_response)
        input_context.add_output(normalized_response)
        await self._db_recorder.record_context(input_context)
        output_result = self._output_normalizer.denormalize(normalized_response)
        return output_result

    async def _run_input_pipeline(
        self,
        normalized_request: Any,
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
            model=normalized_request.get_model(),
            api_key=api_key,
            api_base=api_base,
        )

        # TODO(jakub): handle this by returning a message to the client
        if result.error_message:
            raise Exception(result.error_message)

        return result

    async def _cleanup_after_streaming(
        self, stream: AsyncIterator[Any], context: PipelineContext
    ) -> AsyncIterator[Any]:
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
        elif hasattr(data, "json"):
            # The new format
            with open(fname, "w") as f:
                f.write(data.json())
        else:
            with open(fname, "w") as f:
                f.write(str(data))

    async def complete(
        self,
        data: Dict,
        api_key: Optional[str],
        base_url: Optional[str],
        is_fim_request: bool,
        client_type: ClientType,
        completion_handler: Callable | None = None,
    ) -> Union[Any, AsyncIterator[Any]]:
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
        streaming = normalized_request.get_stream()

        # Pass the request through the input pipeline.
        input_pipeline_result = await self._run_input_pipeline(
            normalized_request,
            api_key,
            base_url,
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
        #
        # With "executing the completion" we actually mean "calling
        # upstream LLM", e.g. sending the HTTP request to OpenAI or
        # Anthropic.
        model_response = None
        if completion_handler != None:
            model_response = await completion_handler(
                provider_request,
                base_url,
                api_key,
                stream=streaming,
                is_fim_request=is_fim_request,
            )
        else:
            model_response = await self._completion_handler.execute_completion(
                provider_request,
                base_url,
                api_key,
                stream=streaming,
                is_fim_request=is_fim_request,
            )

        # Pass the request through the output pipeline
        if not streaming:
            return await self._run_output_pipeline(input_pipeline_result.context, model_response)

        pipeline_output_stream = await self._run_output_stream_pipeline(
            input_pipeline_result.context, model_response, is_fim_request=is_fim_request  # type: ignore
        )
        return self._cleanup_after_streaming(pipeline_output_stream, input_pipeline_result.context)  # type: ignore

    def get_routes(self) -> APIRouter:
        return self.router
