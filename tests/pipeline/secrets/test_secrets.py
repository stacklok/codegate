import pytest
from litellm import ModelResponse
from litellm.types.utils import Delta, StreamingChoices

from codegate.pipeline.base import PipelineContext, PipelineSensitiveData
from codegate.pipeline.output import OutputPipelineContext
from codegate.pipeline.secrets.manager import SecretsManager
from codegate.pipeline.secrets.secrets import SecretUnredactionStep


def create_model_response(content: str) -> ModelResponse:
    """Helper to create test ModelResponse objects"""
    return ModelResponse(
        id="test",
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


class TestSecretUnredactionStep:
    def setup_method(self):
        """Setup fresh instances for each test"""
        self.step = SecretUnredactionStep()
        self.context = OutputPipelineContext()
        self.secrets_manager = SecretsManager()
        self.session_id = "test_session"

        # Setup input context with secrets manager
        self.input_context = PipelineContext()
        self.input_context.sensitive = PipelineSensitiveData(
            manager=self.secrets_manager, session_id=self.session_id
        )

    @pytest.mark.asyncio
    async def test_complete_marker_processing(self):
        """Test processing of a complete REDACTED marker"""
        # Store a secret
        encrypted = self.secrets_manager.store_secret(
            "secret_value", "test_service", "api_key", self.session_id
        )

        # Add content with REDACTED marker to buffer
        self.context.buffer.append(f"Here is the REDACTED<${encrypted}> in text")

        # Process a chunk
        result = await self.step.process_chunk(
            create_model_response("more text"), self.context, self.input_context
        )

        # Verify unredaction
        assert len(result) == 1
        assert result[0].choices[0].delta.content == "Here is the secret_value in text"

    @pytest.mark.asyncio
    async def test_partial_marker_buffering(self):
        """Test handling of partial REDACTED markers"""
        # Add partial marker to buffer
        self.context.buffer.append("Here is REDACTED<$")

        # Process a chunk
        result = await self.step.process_chunk(
            create_model_response("partial"), self.context, self.input_context
        )

        # Should return empty list to continue buffering
        assert result == []

    @pytest.mark.asyncio
    async def test_invalid_encrypted_value(self):
        """Test handling of invalid encrypted values"""
        # Add content with invalid encrypted value
        self.context.buffer.append("Here is REDACTED<$invalid_value> in text")

        # Process chunk
        result = await self.step.process_chunk(
            create_model_response("text"), self.context, self.input_context
        )

        # Should keep the REDACTED marker for invalid values
        assert len(result) == 1
        assert result[0].choices[0].delta.content == "Here is REDACTED<$invalid_value> in text"

    @pytest.mark.asyncio
    async def test_missing_context(self):
        """Test handling of missing input context or secrets manager"""
        # Test with None input context
        with pytest.raises(ValueError, match="Input context not found"):
            await self.step.process_chunk(create_model_response("text"), self.context, None)

        # Test with missing secrets manager
        self.input_context.sensitive.manager = None
        with pytest.raises(ValueError, match="Secrets manager not found in input context"):
            await self.step.process_chunk(
                create_model_response("text"), self.context, self.input_context
            )

    @pytest.mark.asyncio
    async def test_empty_content(self):
        """Test handling of empty content chunks"""
        result = await self.step.process_chunk(
            create_model_response(""), self.context, self.input_context
        )

        # Should pass through empty chunks
        assert len(result) == 1
        assert result[0].choices[0].delta.content == ""

    @pytest.mark.asyncio
    async def test_no_markers(self):
        """Test processing of content without any REDACTED markers"""
        # Create chunk with content
        chunk = create_model_response("Regular text without any markers")

        # Process chunk
        result = await self.step.process_chunk(chunk, self.context, self.input_context)

        # Should pass through unchanged
        assert len(result) == 1
        assert result[0].choices[0].delta.content == "Regular text without any markers"

    @pytest.mark.asyncio
    async def test_wrong_session(self):
        """Test unredaction with wrong session ID"""
        # Store secret with one session
        encrypted = self.secrets_manager.store_secret(
            "secret_value", "test_service", "api_key", "different_session"
        )

        # Try to unredact with different session
        self.context.buffer.append(f"Here is the REDACTED<${encrypted}> in text")

        result = await self.step.process_chunk(
            create_model_response("text"), self.context, self.input_context
        )

        # Should keep REDACTED marker when session doesn't match
        assert len(result) == 1
        assert result[0].choices[0].delta.content == f"Here is the REDACTED<${encrypted}> in text"
