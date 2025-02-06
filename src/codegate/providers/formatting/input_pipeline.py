import time
from typing import AsyncIterator, Union

from litellm import ModelResponse
from litellm.types.utils import Delta, StreamingChoices

from codegate.db.connection import DbRecorder
from codegate.pipeline.base import PipelineContext, PipelineResponse
from codegate.providers.normalizer.base import ModelOutputNormalizer


def _create_stream_end_response(original_response: ModelResponse) -> ModelResponse:
    """Create the final chunk of a stream with finish_reason=stop"""
    return ModelResponse(
        id=original_response.id,
        choices=[
            StreamingChoices(
                finish_reason="stop", index=0, delta=Delta(content="", role=None), logprobs=None
            )
        ],
        created=original_response.created,
        model=original_response.model,
        object="chat.completion.chunk",
    )


def _create_model_response(
    content: str,
    step_name: str,
    model: str,
    streaming: bool,
) -> ModelResponse:
    """
    Create a ModelResponse in either streaming or non-streaming format
    This is required because the ModelResponse format is different for streaming
    and non-streaming responses (see StreamingChoices vs. Dict)
    """
    response_id = f"pipeline-{step_name}-{int(time.time())}"
    created = int(time.time())

    if streaming:
        return ModelResponse(
            id=response_id,
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(content=content, role="assistant"),
                    logprobs=None,
                )
            ],
            created=created,
            model=model,
            object="chat.completion.chunk",
        )
    else:
        return ModelResponse(
            id=response_id,
            choices=[{"text": content, "index": 0, "finish_reason": None}],
            created=created,
            model=model,
        )

async def _convert_to_stream(
    content: str,
    step_name: str,
    model: str,
    context: PipelineContext,
) -> AsyncIterator[ModelResponse]:
    """
    Converts a single completion response, provided by our pipeline as a shortcut
    to a streaming response. The streaming response has two chunks: the first
    one contains the actual content, and the second one contains the finish_reason.
    """
    # First chunk with content
    first_response = _create_model_response(content, step_name, model, streaming=True)
    yield first_response
    # Final chunk with finish_reason
    yield _create_stream_end_response(first_response)


class PipelineResponseFormatter:
    def __init__(
        self,
        output_normalizer: ModelOutputNormalizer,
        db_recorder: DbRecorder,
    ):
        self._output_normalizer = output_normalizer
        self._db_recorder = db_recorder

    async def _cleanup_after_streaming(
        self, stream: AsyncIterator[ModelResponse], context: PipelineContext
    ) -> AsyncIterator[ModelResponse]:
        """Wraps the stream to ensure cleanup after consumption"""
        try:
            async for item in stream:
                context.add_output(item)
                yield item
        finally:
            if context:
                # Record to DB the objects captured during the stream
                await self._db_recorder.record_context(context)

    async def handle_pipeline_response(
        self, pipeline_response: PipelineResponse, streaming: bool, context: PipelineContext
    ) -> Union[ModelResponse, AsyncIterator[ModelResponse]]:
        """
        Convert pipeline response to appropriate format based on streaming flag
        The response is either a ModelResponse or an AsyncIterator[ModelResponse]
        based on the streaming flag
        """
        # First, get the ModelResponse from the pipeline response. The pipeline
        # response itself it just a string (pipeline_response.content) so we turn
        # it into a ModelResponse
        model_response = _create_model_response(
            pipeline_response.content,
            pipeline_response.step_name,
            pipeline_response.model,
            streaming=streaming,
        )
        if not streaming:
            # If we're not streaming, we just return the response translated
            # to the provider-specific format
            context.add_output(model_response)
            await self._db_recorder.record_context(context)
            return self._output_normalizer.denormalize(model_response)

        # If we're streaming, we need to convert the response to a stream first
        # then feed the stream into the completion handler's conversion method
        model_response_stream = _convert_to_stream(
            pipeline_response.content, pipeline_response.step_name, pipeline_response.model, context
        )
        model_response_stream = self._cleanup_after_streaming(model_response_stream, context)
        return self._output_normalizer.denormalize_streaming(model_response_stream)
