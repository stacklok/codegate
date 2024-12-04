import re
from typing import Optional

import structlog
from litellm import ChatCompletionRequest, ModelResponse
from litellm.types.utils import Delta, StreamingChoices

from codegate.pipeline.base import (
    AlertSeverity,
    PipelineContext,
    PipelineResult,
    PipelineStep,
)
from codegate.pipeline.output import OutputPipelineContext, OutputPipelineStep
from codegate.pipeline.secrets.manager import SecretsManager
from codegate.pipeline.secrets.signatures import CodegateSignatures

logger = structlog.get_logger("codegate")


class CodegateSecrets(PipelineStep):
    """Pipeline step that handles secret information requests."""

    def __init__(self):
        """Initialize the CodegateSecrets pipeline step."""
        super().__init__()

    @property
    def name(self) -> str:
        """
        Returns the name of this pipeline step.

        Returns:
            str: The identifier 'codegate-secrets'.
        """
        return "codegate-secrets"

    def _get_absolute_position(self, line_number: int, line_offset: int, text: str) -> int:
        """
        Convert line number and offset to absolute position in text.

        Args:
            line_number: The line number (1-based)
            line_offset: The offset within the line
            text: The full text

        Returns:
            Absolute position in text
        """
        lines = text.split("\n")
        position = sum(len(line) + 1 for line in lines[: line_number - 1])
        return position + line_offset

    def _extend_match_boundaries(self, text: str, start: int, end: int) -> tuple[int, int]:
        """
        Extend match boundaries to include full token if partially matched.

        Args:
            text: The full text
            start: Starting position of match
            end: Ending position of match

        Returns:
            Tuple of (new_start, new_end)
        """

        # Search backwards for opening quote or start of string
        while start > 0 and text[start - 1] not in ['"', "'", " ", "\n", "="]:
            start -= 1

        # Search forwards for closing quote or end of string
        while end < len(text) and text[end] not in ['"', "'", " ", "\n"]:
            end += 1

        return start, end

    def _redeact_text(
        self, text: str, secrets_manager: SecretsManager, session_id: str, context: PipelineContext
    ) -> str:
        """
        Find and encrypt secrets in the given text.

        Args:
            text: The text to protect
            secrets_manager: ..
            session_id: ..
            context: The pipeline context to be able to log alerts
        Returns:
            Protected text with encrypted values
        """
        # Find secrets in the text
        matches = CodegateSignatures.find_in_string(text)
        if not matches:
            return text

        logger.debug(f"Found {len(matches)} secrets in the user message")

        # Convert line positions to absolute positions and extend boundaries
        absolute_matches = []
        for match in matches:
            start = self._get_absolute_position(match.line_number, match.start_index, text)
            end = self._get_absolute_position(match.line_number, match.end_index, text)

            # Extend boundaries to include full token
            start, end = self._extend_match_boundaries(text, start, end)

            # Get the full value
            full_value = text[start:end]
            context.add_alert(
                self.name, trigger_string=full_value, severity_category=AlertSeverity.CRITICAL
            )
            absolute_matches.append((start, end, match._replace(value=full_value)))

        # Sort matches in reverse order to replace from end to start
        absolute_matches.sort(key=lambda x: x[0], reverse=True)

        # Make a mutable copy of the text
        protected_text = list(text)

        # Store matches for logging
        found_secrets = []

        # Replace each match with its encrypted value
        for start, end, match in absolute_matches:
            # Encrypt and store the value
            encrypted_value = secrets_manager.store_secret(
                match.value,
                match.service,
                match.type,
                session_id,
            )

            # Create the replacement string
            replacement = f"REDACTED<${encrypted_value}>"

            # Replace the secret in the text
            protected_text[start:end] = replacement
            # Store for logging
            found_secrets.append(
                {
                    "service": match.service,
                    "type": match.type,
                    "original": match.value,
                    "encrypted": encrypted_value,
                }
            )

            # Convert back to string
            protected_string = "".join(protected_text)

            # Log the findings
            logger.info("\nFound secrets:")

            for secret in found_secrets:
                logger.info(f"\nService: {secret['service']}")
                logger.info(f"Type: {secret['type']}")
                logger.info(f"Original: {secret['original']}")
                logger.info(f"Encrypted: REDACTED<${secret['encrypted']}>")

            logger.info(f"\nProtected text:\n{protected_string}")
            return "".join(protected_text)

    async def process(
        self, request: ChatCompletionRequest, context: PipelineContext
    ) -> PipelineResult:
        """
        Process the request to find and protect secrets in all messages.

        Args:
            request: The chat completion request
            context: The pipeline context

        Returns:
            PipelineResult containing the processed request
        """
        secrets_manager = context.sensitive.manager
        if not secrets_manager or not isinstance(secrets_manager, SecretsManager):
            # Should this be an error?
            raise ValueError("Secrets manager not found in context")
        session_id = context.sensitive.session_id
        if not session_id:
            raise ValueError("Session ID not found in context")

        new_request = request.copy()

        # Process all messages
        for i, message in enumerate(new_request["messages"]):
            if "content" in message and message["content"]:
                # Protect the text
                protected_string = self._redeact_text(
                    message["content"], secrets_manager, session_id, context
                )
                new_request["messages"][i]["content"] = protected_string

        return PipelineResult(request=request, context=context)


class SecretUnredactionStep(OutputPipelineStep):
    """Pipeline step that unredacts protected content in the stream"""

    def __init__(self):
        self.redacted_pattern = re.compile(r"REDACTED<\$([^>]+)>")
        self.marker_start = "REDACTED<$"
        self.marker_end = ">"

    @property
    def name(self) -> str:
        return "secret-unredaction"

    def _is_partial_marker_prefix(self, text: str) -> bool:
        """Check if text ends with a partial marker prefix"""
        for i in range(1, len(self.marker_start) + 1):
            if text.endswith(self.marker_start[:i]):
                return True
        return False

    def _find_complete_redaction(self, text: str) -> tuple[Optional[re.Match[str]], str]:
        """
        Find the first complete REDACTED marker in text.
        Returns (match, remaining_text) if found, (None, original_text) if not.
        """
        matches = list(self.redacted_pattern.finditer(text))
        if not matches:
            return None, text

        # Get the first complete match
        match = matches[0]
        return match, text[match.end() :]

    async def process_chunk(
        self,
        chunk: ModelResponse,
        context: OutputPipelineContext,
        input_context: Optional[PipelineContext] = None,
    ) -> list[ModelResponse]:
        """Process a single chunk of the stream"""
        if not input_context:
            raise ValueError("Input context not found")
        if input_context.sensitive is None or input_context.sensitive.manager is None:
            raise ValueError("Secrets manager not found in input context")
        if input_context.sensitive.session_id == "":
            raise ValueError("Session ID not found in input context")

        if not chunk.choices[0].delta.content:
            return [chunk]

        # Check the buffered content
        buffered_content = "".join(context.buffer)

        # Look for complete REDACTED markers first
        match, remaining = self._find_complete_redaction(buffered_content)
        if match:
            # Found a complete marker, process it
            encrypted_value = match.group(1)
            original_value = input_context.sensitive.manager.get_original_value(
                encrypted_value,
                input_context.sensitive.session_id,
            )

            if original_value is None:
                # If value not found, leave as is
                original_value = match.group(0)  # Keep the REDACTED marker

            # Unredact the content, post an alert and return the chunk
            unredacted_content = buffered_content[: match.start()] + original_value + remaining
            input_context.add_alert(self.name, trigger_string=unredacted_content)
            # Return the unredacted content up to this point
            chunk.choices = [
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        content=unredacted_content,
                        role="assistant",
                    ),
                    logprobs=None,
                )
            ]
            return [chunk]

        # If we have a partial marker at the end, keep buffering
        if self.marker_start in buffered_content or self._is_partial_marker_prefix(
            buffered_content
        ):
            return []

        # No markers or partial markers, let pipeline handle the chunk normally
        return [chunk]
