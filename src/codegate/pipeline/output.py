import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, List, Optional

import structlog

from codegate.db.connection import DbRecorder
from codegate.extract_snippets.message_extractor import CodeSnippet
from codegate.pipeline.base import PipelineContext

logger = structlog.get_logger("codegate")


@dataclass
class OutputPipelineContext:
    """
    Context passed between output pipeline steps.

    Does not include the input context, that one is separate.
    """

    # We store the messages that are not yet sent to the client in the buffer.
    # One reason for this might be that the buffer contains a secret that we want to de-obfuscate
    buffer: list[str] = field(default_factory=list)
    # Store extracted code snippets
    snippets: List[CodeSnippet] = field(default_factory=list)
    # Store all content that has been processed by the pipeline
    processed_content: List[str] = field(default_factory=list)
    # partial buffer to store prefixes
    prefix_buffer: str = ""


class OutputPipelineStep(ABC):
    """
    Base class for output pipeline steps
    The process method should be implemented by subclasses and handles
    processing of a single chunk of the stream.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of this pipeline step"""
        pass

    @abstractmethod
    async def process_chunk(
        self,
        chunk: Any,
        context: OutputPipelineContext,
        input_context: Optional[PipelineContext] = None,
    ) -> List[Any]:
        """
        Process a single chunk of the stream.

        Args:
        - chunk: The input chunk to process, normalized to Any
        - context: The output pipeline context. Can be used to store state between steps, mainly
          the buffer.
        - input_context: The input context from processing the user's input. Can include the secrets
          obfuscated in the user message or code snippets in the user message.

        Return:
        - Empty list to pause the stream
        - List containing one or more Any objects to emit
        """
        pass


class OutputPipelineInstance:
    """
    Handles processing of a single stream
    Think of this class as steps + buffer
    """

    def __init__(
        self,
        pipeline_steps: list[OutputPipelineStep],
        input_context: Optional[PipelineContext] = None,
        db_recorder: Optional[DbRecorder] = None,
    ):
        self._input_context = input_context
        self._pipeline_steps = pipeline_steps
        self._context = OutputPipelineContext()
        # we won't actually buffer the chunk, but in case we need to pass
        # the remaining content in the buffer when the stream ends, we need
        # to store the parameters like model, timestamp, etc.
        self._buffered_chunk = None
        if not db_recorder:
            self._db_recorder = DbRecorder()
        else:
            self._db_recorder = db_recorder

    def _buffer_chunk(self, chunk: Any) -> None:
        """
        Add chunk content to buffer. This is used to store content that is not yet processed
        when a pipeline pauses streaming.
        """
        self._buffered_chunk = chunk
        for content in chunk.get_content():
            text = content.get_text()
            if text is not None:
                self._context.buffer.append(text)

    def _store_chunk_content(self, chunk: Any) -> None:
        """
        Store chunk content in processed content. This keeps track of the content that has been
        streamed through the pipeline.
        """
        for content in chunk.get_content():
            text = content.get_text()
            if text:
                self._context.processed_content.append(text)

    def _record_to_db(self) -> None:
        """
        Record the context to the database

        Important: We cannot use `await` in the finally statement. Otherwise, the stream
        will transmmitted properly. Hence we get the running loop and create a task to
        record the context.
        """
        loop = asyncio.get_running_loop()
        loop.create_task(self._db_recorder.record_context(self._input_context))

    async def process_stream(
        self,
        stream: AsyncIterator[Any],
        cleanup_sensitive: bool = True,
        finish_stream: bool = True,
    ) -> AsyncIterator[Any]:
        """
        Process a stream through all pipeline steps
        """
        try:
            async for chunk in stream:
                # Store chunk content in buffer
                self._buffer_chunk(chunk)
                self._input_context.add_output(chunk)

                # Process chunk through each step of the pipeline
                current_chunks = [chunk]
                for step in self._pipeline_steps:
                    if not current_chunks:
                        # Stop processing if a step returned empty list
                        break

                    processed_chunks = []
                    for c in current_chunks:
                        try:
                            step_result = await step.process_chunk(
                                c, self._context, self._input_context
                            )
                            if not step_result:
                                break
                        except Exception as e:
                            logger.error(f"Error processing step '{step.name}'", exc_info=e)
                            # Re-raise to maintain the current behaviour.
                            raise e

                        processed_chunks.extend(step_result)

                    current_chunks = processed_chunks

                # Yield all processed chunks
                for c in current_chunks:
                    self._store_chunk_content(c)
                    self._context.buffer.clear()
                    yield c

        except Exception as e:
            # Log exception and stop processing
            logger.error(f"Error processing stream: {e}", exc_info=e)
            raise e
        finally:
            # NOTE: Don't use await in finally block, it will break the stream
            # Don't flush the buffer if we assume we'll call the pipeline again
            if cleanup_sensitive is False:
                if finish_stream:
                    self._record_to_db()
                return

            # TODO figure out what's the logic here.
            # Process any remaining content in buffer when stream ends
            if self._context.buffer:
                final_content = "".join(self._context.buffer)
                logger.error(
                    "Context buffer was not empty, it should have been!",
                    content=final_content,
                    len=len(self._context.buffer),
                )

                # NOTE: this block ensured that buffered chunks were
                # flushed at the end of the pipeline. This was
                # possible as long as the current implementation
                # assumed that all messages were equivalent and
                # position was not relevant.
                #
                # This is not the case for Anthropic, whose protocol
                # is much more structured than that of the others.
                #
                # We're not there yet to ensure that such a protocol
                # is not broken in face of messages being arbitrarily
                # retained at each pipeline step, so we decided to
                # treat a clogged pipelines as a bug.
                self._context.buffer.clear()

            if finish_stream:
                self._record_to_db()
            # Cleanup sensitive data through the input context
            if cleanup_sensitive and self._input_context and self._input_context.sensitive:
                self._input_context.sensitive.secure_cleanup()


class OutputPipelineProcessor:
    """
    Since we want to provide each run of the pipeline with a fresh context,
    we need a factory to create new instances of the pipeline.
    """

    def __init__(self, pipeline_steps: list[OutputPipelineStep]):
        self.pipeline_steps = pipeline_steps

    def _create_instance(self) -> OutputPipelineInstance:
        """Create a new pipeline instance for processing a stream"""
        return OutputPipelineInstance(self.pipeline_steps)

    async def process_stream(self, stream: AsyncIterator[Any]) -> AsyncIterator[Any]:
        """Create a new pipeline instance and process the stream"""
        instance = self._create_instance()
        async for chunk in instance.process_stream(stream):
            yield chunk
