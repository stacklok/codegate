from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Callable, Dict, Optional, Union

from fastapi import APIRouter, Request
from litellm import ModelResponse
from litellm.types.llms.openai import ChatCompletionRequest

from codegate.codegate_logging import setup_logging
from codegate.pipeline.base import PipelineResult, SequentialPipelineProcessor
from codegate.providers.completion.base import BaseCompletionHandler
from codegate.providers.formatting.input_pipeline import PipelineResponseFormatter
from codegate.providers.normalizer.base import ModelInputNormalizer, ModelOutputNormalizer

logger = setup_logging()
StreamGenerator = Callable[[AsyncIterator[Any]], AsyncIterator[str]]


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
        pipeline_processor: Optional[SequentialPipelineProcessor] = None,
        fim_pipeline_processor: Optional[SequentialPipelineProcessor] = None,
    ):
        self.router = APIRouter()
        self._completion_handler = completion_handler
        self._input_normalizer = input_normalizer
        self._output_normalizer = output_normalizer
        self._pipeline_processor = pipeline_processor
        self._fim_pipelin_processor = fim_pipeline_processor

        self._pipeline_response_formatter = PipelineResponseFormatter(output_normalizer)

        self._setup_routes()

    @abstractmethod
    def _setup_routes(self) -> None:
        pass

    @property
    @abstractmethod
    def provider_route_name(self) -> str:
        pass

    async def _run_output_stream_pipeline(
        self,
        normalized_stream: AsyncIterator[ModelResponse],
    ) -> AsyncIterator[ModelResponse]:
        # we don't have a pipeline for output stream yet
        return normalized_stream

    def _run_output_pipeline(
        self,
        normalized_response: ModelResponse,
    ) -> ModelResponse:
        # we don't have a pipeline for output yet
        return normalized_response

    async def _run_input_pipeline(
        self, normalized_request: ChatCompletionRequest, is_fim_request: bool
    ) -> PipelineResult:
        # Decide which pipeline processor to use
        if is_fim_request:
            pipeline_processor = self._fim_pipelin_processor
            logger.info("FIM pipeline selected for execution.")
        else:
            pipeline_processor = self._pipeline_processor
            logger.info("Chat completion pipeline selected for execution.")
        if pipeline_processor is None:
            return PipelineResult(request=normalized_request)

        result = await pipeline_processor.process_request(normalized_request)

        # TODO(jakub): handle this by returning a message to the client
        if result.error_message:
            raise Exception(result.error_message)

        return result

    def _is_fim_request_url(self, request: Request) -> bool:
        """
        Checks the request URL to determine if a request is FIM or chat completion.
        Used by: llama.cpp
        """
        request_path = request.url.path
        # Evaluate first a larger substring.
        if request_path.endswith("/chat/completions"):
            return False

        if request_path.endswith("/completions"):
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

    def _is_fim_request(self, request: Request, data: Dict) -> bool:
        """
        Determin if the request is FIM by the URL or the data of the request.
        """
        # Avoid more expensive inspection of body by just checking the URL.
        if self._is_fim_request_url(request):
            return True

        return self._is_fim_request_body(data)

    async def complete(
        self, data: Dict, api_key: Optional[str], is_fim_request: bool
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
        normalized_request = self._input_normalizer.normalize(data)
        streaming = data.get("stream", False)
        input_pipeline_result = await self._run_input_pipeline(normalized_request, is_fim_request)
        if input_pipeline_result.response:
            return self._pipeline_response_formatter.handle_pipeline_response(
                input_pipeline_result.response, streaming
            )

        provider_request = self._input_normalizer.denormalize(input_pipeline_result.request)

        # Execute the completion and translate the response
        # This gives us either a single response or a stream of responses
        # based on the streaming flag
        model_response = await self._completion_handler.execute_completion(
            provider_request, api_key=api_key, stream=streaming
        )
        if not streaming:
            normalized_response = self._output_normalizer.normalize(model_response)
            pipeline_output = self._run_output_pipeline(normalized_response)
            return self._output_normalizer.denormalize(pipeline_output)

        normalized_stream = self._output_normalizer.normalize_streaming(model_response)
        pipeline_output_stream = await self._run_output_stream_pipeline(normalized_stream)
        return self._output_normalizer.denormalize_streaming(pipeline_output_stream)

    def get_routes(self) -> APIRouter:
        return self.router
