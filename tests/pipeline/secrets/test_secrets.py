import os
import tempfile

import pytest

from codegate.pipeline.base import PipelineContext, PipelineSensitiveData
from codegate.pipeline.output import OutputPipelineContext
from codegate.pipeline.secrets.manager import SecretsManager
from codegate.pipeline.secrets.secrets import (
    SecretsEncryptor,
    SecretsObfuscator,
    SecretUnredactionStep,
)
from codegate.pipeline.secrets.signatures import CodegateSignatures, Match
from codegate.types.common import Delta, ModelResponse, StreamingChoices


class TestSecretsModifier:
    def test_get_absolute_position(self):
        modifier = SecretsObfuscator()  # Using concrete implementation for testing
        text = "line1\nline2\nline3"

        # Test various positions
        assert modifier._get_absolute_position(1, 0, text) == 0  # Start of first line
        assert modifier._get_absolute_position(2, 0, text) == 6  # Start of second line
        assert modifier._get_absolute_position(1, 4, text) == 4  # Middle of first line

    def test_extend_match_boundaries(self):
        modifier = SecretsObfuscator()

        # Test extension with quotes
        text = 'config = "secret_value" # comment'
        secret = "secret_value"
        start = text.index(secret)
        end = start + len(secret)

        start, end = modifier._extend_match_boundaries(text, start, end)
        assert text[start:end] == secret

        # Test extension without quotes spaces
        text = "config = secret_value # comment"
        secret = "secret_value"
        start = text.index(secret)
        end = start + len(secret)

        start, end = modifier._extend_match_boundaries(text, start, end)
        assert text[start:end] == secret


@pytest.fixture
def valid_yaml_content():
    return """
- AWS:
    - Access Key: '[A-Z0-9]{20}'
"""


@pytest.fixture
def temp_yaml_file(valid_yaml_content):
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as f:
        f.write(valid_yaml_content)
    yield f.name
    os.unlink(f.name)


class TestSecretsEncryptor:
    @pytest.fixture(autouse=True)
    def setup(self, temp_yaml_file):
        CodegateSignatures.initialize(temp_yaml_file)
        self.context = PipelineContext()
        self.secrets_manager = SecretsManager()
        self.session_id = "test_session"
        self.encryptor = SecretsEncryptor(self.secrets_manager, self.context, self.session_id)

    def test_hide_secret(self):
        # Create a test match
        match = Match(
            service="AWS",
            type="Access Key",
            secret_key="API_KEY",
            value="AKIAIOSFODNN7EXAMPLE",
            line_number=1,
            start_index=0,
            end_index=9,
        )

        # Test secret hiding
        hidden = self.encryptor._hide_secret(match)
        assert hidden.startswith("REDACTED<$")
        assert hidden.endswith(">")

        # Verify the secret was stored
        encrypted_value = hidden[len("REDACTED<$") : -1]
        original = self.secrets_manager.get_original_value(encrypted_value, self.session_id)
        assert original == "AKIAIOSFODNN7EXAMPLE"

    def test_obfuscate(self):
        # Test text with a secret
        text = "API_KEY=AKIAIOSFODNN7EXAMPLE\nOther text"
        protected, matched_secrets = self.encryptor.obfuscate(text, None)

        assert len(matched_secrets) == 1
        assert "REDACTED<$" in protected
        assert "AKIAIOSFODNN7EXAMPLE" not in protected
        assert "Other text" in protected


class TestSecretsObfuscator:
    @pytest.fixture(autouse=True)
    def setup(self, temp_yaml_file):
        CodegateSignatures.initialize(temp_yaml_file)
        self.obfuscator = SecretsObfuscator()

    def test_hide_secret(self):
        match = Match(
            service="AWS",
            type="Access Key",
            secret_key="API_KEY",
            value="AKIAIOSFODNN7EXAMPLE",
            line_number=1,
            start_index=0,
            end_index=15,
        )

        hidden = self.obfuscator._hide_secret(match)
        assert hidden == "*" * 32
        assert len(hidden) == 32  # Consistent length regardless of input

    def test_obfuscate(self):
        # Test text with multiple secrets
        text = "API_KEY=AKIAIOSFODNN7EXAMPLE\nPASSWORD=AKIAIOSFODNN7EXAMPLE"
        protected, matched_secrets = self.obfuscator.obfuscate(text, None)

        assert len(matched_secrets) == 2
        assert "AKIAIOSFODNN7EXAMPLE" not in protected
        assert "*" * 32 in protected

        # Verify format
        expected_pattern = f"API_KEY={'*' * 32}\nPASSWORD={'*' * 32}"
        assert protected == expected_pattern

    def test_obfuscate_no_secrets(self):
        text = "Regular text without secrets"
        protected, matched_secrets = self.obfuscator.obfuscate(text, None)

        assert len(matched_secrets) == 0
        assert protected == text


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
