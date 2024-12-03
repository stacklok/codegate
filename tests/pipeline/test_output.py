from typing import Optional

import pytest
from litellm import ModelResponse
from litellm.types.utils import Delta, StreamingChoices

from codegate.pipeline.base import PipelineContext
from codegate.pipeline.output import (
    OutputPipelineContext,
    OutputPipelineInstance,
    OutputPipelineStep,
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
        chunk: ModelResponse,
        context: OutputPipelineContext,
        input_context: PipelineContext = None,
    ) -> ModelResponse:
        if self._should_pause:
            return None

        if self._modify_content and chunk.choices[0].delta.content:
            # Append step name to content to track modifications
            modified_content = f"{chunk.choices[0].delta.content}_{self.name}"
            chunk.choices[0].delta.content = modified_content

        return chunk


def create_model_response(content: str, id: str = "test") -> ModelResponse:
    """Helper to create test ModelResponse objects"""
    return ModelResponse(
        id=id,
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(content=content, role="assistant"),
                logprobs=None,
            )
        ],
        created=0,
        model="test-model",
        object="chat.completion.chunk",
    )


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
        instance = OutputPipelineInstance([step])

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
        instance = OutputPipelineInstance(steps)

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
        instance = OutputPipelineInstance(steps)

        async def mock_stream():
            yield create_model_response("he")
            yield create_model_response("ll")
            yield create_model_response("o")
            yield create_model_response(" wo")
            yield create_model_response("rld")

        chunks = []
        async for chunk in instance.process_stream(mock_stream()):
            chunks.append(chunk)

        # Should get one chunk at the end with all buffered content
        assert len(chunks) == 1
        # Content should be buffered and combined
        assert chunks[0].choices[0].delta.content == "hello world"
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
                chunk: ModelResponse,
                context: OutputPipelineContext,
                input_context: PipelineContext = None,
            ) -> Optional[ModelResponse]:
                # Replace 'world' with 'moon' in buffered content
                content = "".join(context.buffer)
                if "world" in content:
                    content = content.replace("world", "moon")
                    chunk.choices = [
                        StreamingChoices(
                            finish_reason=None,
                            index=0,
                            delta=Delta(content=content, role="assistant"),
                            logprobs=None,
                        )
                    ]
                    return chunk
                return None

        instance = OutputPipelineInstance([ReplacementStep()])

        async def mock_stream():
            yield create_model_response("he")
            yield create_model_response("ll")
            yield create_model_response("o")
            yield create_model_response("wo")
            yield create_model_response("rld")

        chunks = []
        async for chunk in instance.process_stream(mock_stream()):
            chunks.append(chunk)

        # Should get one chunk at the end with modified content
        assert len(chunks) == 1
        assert chunks[0].choices[0].delta.content == "hellomoon"
        # Buffer should be cleared after flush
        assert len(instance._context.buffer) == 0

    @pytest.mark.asyncio
    async def test_buffer_processing(self):
        """Test that content is properly buffered and cleared"""
        step = MockOutputPipelineStep("test_step")
        instance = OutputPipelineInstance([step])

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
        instance = OutputPipelineInstance([step])

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
                chunk: ModelResponse,
                context: OutputPipelineContext,
                input_context: PipelineContext = None,
            ) -> ModelResponse:
                assert input_context.metadata["test"] == "value"
                return chunk

        instance = OutputPipelineInstance([ContextCheckingStep()], input_context=input_context)

        async def mock_stream():
            yield create_model_response("test")

        async for _ in instance.process_stream(mock_stream()):
            pass

    @pytest.mark.asyncio
    async def test_buffer_flush_on_stream_end(self):
        """Test that buffer is properly flushed when stream ends"""
        step = MockOutputPipelineStep("test_step", should_pause=True)
        instance = OutputPipelineInstance([step])

        async def mock_stream():
            yield create_model_response("Hello")
            yield create_model_response("World")

        chunks = []
        async for chunk in instance.process_stream(mock_stream()):
            chunks.append(chunk)

        # Should get one chunk with combined buffer content
        assert len(chunks) == 1
        assert chunks[0].choices[0].delta.content == "HelloWorld"
        # Buffer should be cleared after flush
        assert len(instance._context.buffer) == 0