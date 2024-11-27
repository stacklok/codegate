from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Callable, Dict, Optional, Union

from fastapi import APIRouter
from litellm import ModelResponse
from litellm.types.llms.openai import ChatCompletionRequest

from codegate.pipeline.base import PipelineResult, SequentialPipelineProcessor
from codegate.providers.completion.base import BaseCompletionHandler
from codegate.providers.formatting.input_pipeline import PipelineResponseFormatter
from codegate.providers.normalizer.base import ModelInputNormalizer, ModelOutputNormalizer

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
    ):
        self.router = APIRouter()
        self._completion_handler = completion_handler
        self._input_normalizer = input_normalizer
        self._output_normalizer = output_normalizer
        self._pipeline_processor = pipeline_processor

        self._pipeline_response_formatter = PipelineResponseFormatter(output_normalizer)

        self._setup_routes()

    @abstractmethod
    def _setup_routes(self) -> None:
        pass

    @property
    @abstractmethod
    def provider_route_name(self) -> str:
        pass

    async def _run_input_pipeline(
        self,
        normalized_request: ChatCompletionRequest,
    ) -> PipelineResult:
        if self._pipeline_processor is None:
            return PipelineResult(request=normalized_request)

        result = await self._pipeline_processor.process_request(normalized_request)

        # TODO(jakub): handle this by returning a message to the client
        if result.error_message:
            raise Exception(result.error_message)

        return result

    async def complete(
        self,
        data: Dict,
        api_key: Optional[str],
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

        input_pipeline_result = await self._run_input_pipeline(normalized_request)
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
            return self._output_normalizer.denormalize(model_response)
        return self._output_normalizer.denormalize_streaming(model_response)

    def get_routes(self) -> APIRouter:
        return self.router
