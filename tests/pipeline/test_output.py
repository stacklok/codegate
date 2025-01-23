from typing import List
from unittest.mock import AsyncMock

import pytest

from codegate.pipeline.base import PipelineContext
from codegate.pipeline.output import (
    OutputPipelineContext,
    OutputPipelineInstance,
    OutputPipelineStep,
)
from codegate.types.common import Delta, ModelResponse, StreamingChoices
from codegate.types.openai import (
    ChatCompletionRequest,
    ChoiceDelta,
    MessageDelta,
    StreamingChatCompletion,
)


class MockOutputPipelineStep(OutputPipelineStep):
    """Mock pipeline step for testing"""

    def __init__(self, name: str, should_pause: bool = False, modify_content: bool = False):
        self._name = name
        self._should_pause = should_pause
        self._modify_content = modify_content

    @property
    def name(self) -> str:
        return self._name

    async def process_chunk(
        self,
        chunk: StreamingChatCompletion,
        context: OutputPipelineContext,
        input_context: PipelineContext = None,
    ) -> list[StreamingChatCompletion]:
        if self._should_pause:
            return []

        if next(chunk.get_content(), None) is None:
            return [chunk] # short-circuit

        content = next(chunk.get_content())
        if content.get_text() is None or content.get_text() == "":
            return [chunk] # short-circuit

        if self._modify_content:
            # Append step name to content to track modifications
            modified_content = f"{content.get_text()}_{self.name}"
            content.set_text(modified_content)

        return [chunk]


def create_model_response(content: str, id: str = "test") -> StreamingChatCompletion:
    """Helper to create test StreamingChatCompletion objects"""
    return StreamingChatCompletion(
        id=id,
        choices=[
            ChoiceDelta(
                finish_reason=None,
                index=0,
                delta=MessageDelta(content=content, role="assistant"),
                logprobs=None,
            )
        ],
        created=0,
        model="test-model",
        object="chat.completion.chunk",
    )


class MockContext:

    def __init__(self):
        self.sensitive = False

    def add_output(self, chunk: StreamingChatCompletion):
        pass


class TestOutputPipelineContext:
    def test_buffer_initialization(self):
        """Test that buffer is properly initialized"""
        context = OutputPipelineContext()
        assert isinstance(context.buffer, list)
        assert len(context.buffer) == 0

    def test_buffer_operations(self):
        """Test adding and clearing buffer content"""
        context = OutputPipelineContext()
        context.buffer.append("test1")
        context.buffer.append("test2")

        assert len(context.buffer) == 2
        assert context.buffer == ["test1", "test2"]

        context.buffer.clear()
        assert len(context.buffer) == 0


class TestOutputPipelineInstance:
    @pytest.mark.asyncio
    async def test_single_step_processing(self):
        """Test processing a stream through a single step"""
        step = MockOutputPipelineStep("test_step", modify_content=True)
        context = MockContext()
        db_recorder = AsyncMock()
        instance = OutputPipelineInstance([step], context, db_recorder)

        async def mock_stream():
            yield create_model_response("Hello")
            yield create_model_response("World")

        chunks = []
        async for chunk in instance.process_stream(mock_stream()):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].choices[0].delta.content == "Hello_test_step"
        assert chunks[1].choices[0].delta.content == "World_test_step"
        # Buffer should be cleared after each successful chunk
        assert len(instance._context.buffer) == 0

    @pytest.mark.asyncio
    async def test_multiple_steps_processing(self):
        """Test processing a stream through multiple steps"""
        steps = [
            MockOutputPipelineStep("step1", modify_content=True),
            MockOutputPipelineStep("step2", modify_content=True),
        ]
        context = MockContext()
        db_recorder = AsyncMock()
        instance = OutputPipelineInstance(steps, context, db_recorder)

        async def mock_stream():
            yield create_model_response("Hello")

        chunks = []
        async for chunk in instance.process_stream(mock_stream()):
            chunks.append(chunk)

        assert len(chunks) == 1
        # Content should be modified by both steps
        assert chunks[0].choices[0].delta.content == "Hello_step1_step2"
        # Buffer should be cleared after successful processing
        assert len(instance._context.buffer) == 0

    @pytest.mark.asyncio
    async def test_step_pausing(self):
        """Test that a step can pause the stream and content is buffered until flushed"""
        steps = [
            MockOutputPipelineStep("step1", should_pause=True),
            MockOutputPipelineStep("step2", modify_content=True),
        ]
        context = MockContext()
        db_recorder = AsyncMock()
        instance = OutputPipelineInstance(steps, context, db_recorder)

        async def mock_stream():
            yield create_model_response("he")
            yield create_model_response("ll")
            yield create_model_response("o")
            yield create_model_response(" wo")
            yield create_model_response("rld")

        chunks = []
        async for chunk in instance.process_stream(mock_stream()):
            chunks.append(chunk)

        # NOTE: this test ensured that buffered chunks were flushed at
        # the end of the pipeline. This was possible as long as the
        # current implementation assumed that all messages were
        # equivalent and position was not relevant.
        #
        # This is not the case for Anthropic, whose protocol is much
        # more structured than that of the others.
        #
        # We're not there yet to ensure that such a protocol is not
        # broken in face of messages being arbitrarily retained at
        # each pipeline step, so we decided to treat a clogged
        # pipelines as a bug.

        # Should get one chunk at the end with all buffered content
        assert len(chunks) == 0
        # Content should be buffered and combined
        # assert chunks[0].choices[0].delta.content == "hello world"
        # Buffer should be cleared after flush
        assert len(instance._context.buffer) == 0

    @pytest.mark.asyncio
    async def test_step_pausing_with_replacement(self):
        """Test that a step can pause the stream and modify the buffered content before flushing"""

        class ReplacementStep(OutputPipelineStep):
            """Step that replaces 'world' with 'moon' when found in buffer"""

            def __init__(self, should_pause: bool = True):
                self._should_pause = should_pause

            @property
            def name(self) -> str:
                return "replacement"

            async def process_chunk(
                self,
                chunk: StreamingChatCompletion,
                context: OutputPipelineContext,
                input_context: PipelineContext = None,
            ) -> List[StreamingChatCompletion]:
                # Replace 'world' with 'moon' in buffered content
                content = "".join(context.buffer)
                if "world" in content:
                    content = content.replace("world", "moon")
                    chunk.choices = [
                        ChoiceDelta(
                            finish_reason=None,
                            index=0,
                            delta=MessageDelta(content=content, role="assistant"),
                            logprobs=None,
                        )
                    ]
                    return [chunk]
                return []

        context = MockContext()
        db_recorder = AsyncMock()
        instance = OutputPipelineInstance([ReplacementStep()], context, db_recorder)

        async def mock_stream():
            yield create_model_response("he")
            yield create_model_response("ll")
            yield create_model_response("o")
            yield create_model_response(" wo")
            yield create_model_response("rld")

        chunks = []
        async for chunk in instance.process_stream(mock_stream()):
            chunks.append(chunk)

        # Should get one chunk at the end with modified content
        assert len(chunks) == 1
        assert chunks[0].choices[0].delta.content == "hello moon"
        # Buffer should be cleared after flush
        assert len(instance._context.buffer) == 0

    @pytest.mark.asyncio
    async def test_buffer_processing(self):
        """Test that content is properly buffered and cleared"""
        step = MockOutputPipelineStep("test_step")
        context = MockContext()
        db_recorder = AsyncMock()
        instance = OutputPipelineInstance([step], context, db_recorder)

        async def mock_stream():
            yield create_model_response("Hello")
            yield create_model_response("World")

        chunks = []
        async for chunk in instance.process_stream(mock_stream()):
            chunks.append(chunk)
            # Buffer should be cleared after each successful chunk
            assert len(instance._context.buffer) == 0

        assert len(chunks) == 2
        assert chunks[0].choices[0].delta.content == "Hello"
        assert chunks[1].choices[0].delta.content == "World"

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        """Test handling of an empty stream"""
        step = MockOutputPipelineStep("test_step")
        context = MockContext()
        db_recorder = AsyncMock()
        instance = OutputPipelineInstance([step], context, db_recorder)

        async def mock_stream():
            if False:
                yield  # Empty stream

        chunks = []
        async for chunk in instance.process_stream(mock_stream()):
            chunks.append(chunk)

        assert len(chunks) == 0
        assert len(instance._context.buffer) == 0

    @pytest.mark.asyncio
    async def test_input_context_passing(self):
        """Test that input context is properly passed to steps"""
        input_context = PipelineContext()
        input_context.metadata["test"] = "value"

        class ContextCheckingStep(OutputPipelineStep):
            @property
            def name(self) -> str:
                return "context_checker"

            async def process_chunk(
                self,
                chunk: StreamingChatCompletion,
                context: OutputPipelineContext,
                input_context: PipelineContext = None,
            ) -> List[StreamingChatCompletion]:
                assert input_context.metadata["test"] == "value"
                return [chunk]

        db_recorder = AsyncMock()
        instance = OutputPipelineInstance(
            [ContextCheckingStep()], input_context=input_context, db_recorder=db_recorder
        )

        async def mock_stream():
            yield create_model_response("test")

        async for _ in instance.process_stream(mock_stream()):
            pass

    @pytest.mark.asyncio
    async def test_buffer_flush_on_stream_end(self):
        """Test that buffer is properly flushed when stream ends"""
        step = MockOutputPipelineStep("test_step", should_pause=True)
        context = MockContext()
        db_recorder = AsyncMock()
        instance = OutputPipelineInstance([step], context, db_recorder)

        async def mock_stream():
            yield create_model_response("Hello")
            yield create_model_response("World")

        chunks = []
        async for chunk in instance.process_stream(mock_stream()):
            chunks.append(chunk)

        # We do not flush messages anymore, this should be treated as
        # a bug of the pipeline rather than and edge case.
        assert len(chunks) == 0
