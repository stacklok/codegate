from unittest.mock import MagicMock, patch

import pytest
from litellm import ChatCompletionRequest, ModelResponse
from litellm.types.utils import Delta, StreamingChoices

from codegate.pipeline.base import PipelineContext, PipelineSensitiveData
from codegate.pipeline.output import OutputPipelineContext
from codegate.pipeline.pii.pii import CodegatePii, PiiRedactionNotifier, PiiUnRedactionStep
from codegate.pipeline.sensitive_data.manager import SensitiveDataManager


class TestCodegatePii:
    @pytest.fixture
    def mock_config(self):
        with patch("codegate.config.Config.get_config") as mock:
            mock_config = MagicMock()
            mock_config.prompts.pii_redacted = "PII has been redacted"
            mock.return_value = mock_config
            yield mock_config

    @pytest.fixture
    def pii_step(self):
        mock_sensitive_data_manager = MagicMock()
        return CodegatePii(mock_sensitive_data_manager)

    def test_name(self, pii_step):
        assert pii_step.name == "codegate-pii"

    def test_get_redacted_snippet_no_pii(self, pii_step):
        message = "Hello world"
        pii_details = []

        snippet = pii_step._get_redacted_snippet(message, pii_details)

        assert snippet == ""

    def test_get_redacted_snippet_with_pii(self, pii_step):
        message = "My email is <uuid> and phone is <uuid2>"
        pii_details = [{"start": 12, "end": 18}, {"start": 29, "end": 35}]

        snippet = pii_step._get_redacted_snippet(message, pii_details)

        assert snippet == message[12:36]

    @pytest.mark.asyncio
    async def test_process_no_messages(self, pii_step):
        request = ChatCompletionRequest(model="test-model")
        context = PipelineContext()

        result = await pii_step.process(request, context)

        assert result.request == request
        assert result.context == context


class TestPiiUnRedactionStep:
    @pytest.fixture
    def unredaction_step(self):
        return PiiUnRedactionStep()

    def test_name(self, unredaction_step):
        assert unredaction_step.name == "pii-unredaction-step"

    def test_is_complete_uuid_valid(self, unredaction_step):
        uuid = "12345678-1234-1234-1234-123456789012"
        assert unredaction_step._is_complete_uuid(uuid) is True

    def test_is_complete_uuid_invalid(self, unredaction_step):
        uuid = "invalid-uuid"
        assert unredaction_step._is_complete_uuid(uuid) is False

    @pytest.mark.asyncio
    async def test_process_chunk_no_content(self, unredaction_step):
        chunk = ModelResponse(
            id="test",
            choices=[
                StreamingChoices(
                    finish_reason=None, index=0, delta=Delta(content=None), logprobs=None
                )
            ],
            created=1234567890,
            model="test-model",
            object="chat.completion.chunk",
        )
        context = OutputPipelineContext()
        input_context = PipelineContext()

        result = await unredaction_step.process_chunk(chunk, context, input_context)

        assert result == [chunk]

    @pytest.mark.asyncio
    async def test_process_chunk_with_uuid(self, unredaction_step):
        uuid = "12345678-1234-1234-1234-123456789012"
        chunk = ModelResponse(
            id="test",
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(content=f"Text with #{uuid}#"),
                    logprobs=None,
                )
            ],
            created=1234567890,
            model="test-model",
            object="chat.completion.chunk",
        )
        context = OutputPipelineContext()
        manager = SensitiveDataManager()
        sensitive = PipelineSensitiveData(manager=manager, session_id="session-id")
        input_context = PipelineContext(sensitive=sensitive)

        # Mock PII manager in input context
        mock_sensitive_data_manager = MagicMock()
        mock_sensitive_data_manager.get_original_value = MagicMock(return_value="test@example.com")
        input_context.metadata["sensitive_data_manager"] = mock_sensitive_data_manager

        result = await unredaction_step.process_chunk(chunk, context, input_context)
        assert result[0].choices[0].delta.content == "Text with test@example.com"

    @pytest.mark.asyncio
    async def test_detect_not_an_uuid(self, unredaction_step):
        chunk1 = ModelResponse(
            id="test",
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(content="#"),
                    logprobs=None,
                )
            ],
            created=1234567890,
            model="test-model",
            object="chat.completion.chunk",
        )
        chunk2 = ModelResponse(
            id="test",
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(content=" filepath"),
                    logprobs=None,
                )
            ],
            created=1234567890,
            model="test-model",
            object="chat.completion.chunk",
        )

        context = OutputPipelineContext()
        manager = SensitiveDataManager()
        sensitive = PipelineSensitiveData(manager=manager, session_id="session-id")
        input_context = PipelineContext(sensitive=sensitive)

        # Mock PII manager in input context
        mock_sensitive_data_manager = MagicMock()
        mock_sensitive_data_manager.get_original_value = MagicMock(return_value="test@example.com")
        input_context.metadata["sensitive_data_manager"] = mock_sensitive_data_manager

        result = await unredaction_step.process_chunk(chunk1, context, input_context)
        assert not result
        result = await unredaction_step.process_chunk(chunk2, context, input_context)
        assert result[0].choices[0].delta.content == "# filepath"


class TestPiiRedactionNotifier:
    @pytest.fixture
    def notifier(self):
        return PiiRedactionNotifier()

    def test_name(self, notifier):
        assert notifier.name == "pii-redaction-notifier"

    def test_format_pii_summary_single(self, notifier):
        pii_details = [{"type": "EMAIL_ADDRESS", "value": "test@example.com"}]

        summary = notifier._format_pii_summary(pii_details)

        assert summary == "1 email address"

    def test_format_pii_summary_multiple(self, notifier):
        pii_details = [
            {"type": "EMAIL_ADDRESS", "value": "test1@example.com"},
            {"type": "EMAIL_ADDRESS", "value": "test2@example.com"},
            {"type": "PHONE_NUMBER", "value": "123-456-7890"},
        ]

        summary = notifier._format_pii_summary(pii_details)

        assert summary == "2 email addresss, 1 phone number"

    @pytest.mark.asyncio
    async def test_process_chunk_no_pii(self, notifier):
        chunk = ModelResponse(
            id="test",
            choices=[
                StreamingChoices(
                    finish_reason=None, index=0, delta=Delta(content="Hello"), logprobs=None
                )
            ],
            created=1234567890,
            model="test-model",
            object="chat.completion.chunk",
        )
        context = OutputPipelineContext()
        input_context = PipelineContext()

        result = await notifier.process_chunk(chunk, context, input_context)

        assert result == [chunk]

    @pytest.mark.asyncio
    async def test_process_chunk_with_pii(self, notifier):
        chunk = ModelResponse(
            id="test",
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(content="Hello", role="assistant"),
                    logprobs=None,
                )
            ],
            created=1234567890,
            model="test-model",
            object="chat.completion.chunk",
        )
        context = OutputPipelineContext()
        input_context = PipelineContext()
        input_context.metadata["redacted_pii_count"] = 1
        input_context.metadata["redacted_pii_details"] = [
            {"type": "EMAIL_ADDRESS", "value": "test@example.com"}
        ]
        input_context.metadata["redacted_text"] = "<test-uuid>"

        result = await notifier.process_chunk(chunk, context, input_context)

        assert len(result) == 2  # Notification chunk + original chunk
        notification_content = result[0].choices[0].delta.content
        assert "CodeGate protected" in notification_content
        assert "1 email address" in notification_content
        assert result[1] == chunk
