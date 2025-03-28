from abc import abstractmethod
from typing import Any, List, Optional, Tuple

import regex as re
import structlog

from codegate.db.models import AlertSeverity
from codegate.extract_snippets.factory import MessageCodeExtractorFactory
from codegate.pipeline.base import (
    CodeSnippet,
    PipelineContext,
    PipelineResult,
    PipelineStep,
)
from codegate.pipeline.output import OutputPipelineContext, OutputPipelineStep
from codegate.pipeline.secrets.signatures import CodegateSignatures, Match
from codegate.pipeline.sensitive_data.manager import SensitiveData, SensitiveDataManager

logger = structlog.get_logger("codegate")


class SecretsModifier:
    """
    A class that helps obfuscate text by piping it through the secrets manager
    that finds the secrets and then calling hide_secret to modify them.

    What modifications are done is up to the user who subclasses SecretsModifier
    """

    def __init__(self):
        """Initialize the CodegateSecrets pipeline step."""
        super().__init__()
        # Initialize and load signatures immediately
        CodegateSignatures.initialize("signatures.yaml")

    @abstractmethod
    def _hide_secret(self, match: Match) -> str:
        """
        User-defined callable to hide a secret match to either obfuscate
        it or reversibly encrypt
        """
        pass

    @abstractmethod
    def _notify_secret(
        self, match: Match, code_snippet: Optional[CodeSnippet], protected_text: List[str]
    ) -> None:
        """
        Notify about a found secret
        TODO: If the secret came from a CodeSnippet we should notify about that. This would
        involve using the CodeSnippetExtractor step that is further down the pipeline.
        """
        pass

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

    def _get_surrounding_secret_lines(
        self, protected_text: List[str], secret_line: int, surrounding_lines: int = 3
    ) -> str:
        """
        Get the lines before and after the secret line to provide context.

        Args:
            protected_text: The text with secrets replaced
            secret_line: The line number of the secret
        """
        lines = "".join(protected_text).split("\n")
        start_line = max(secret_line - surrounding_lines, 0)
        end_line = min(secret_line + surrounding_lines, len(lines))
        return "\n".join(lines[start_line:end_line])

    def obfuscate(self, text: str, snippet: Optional[CodeSnippet]) -> tuple[str, List[Match]]:
        if snippet:
            text = snippet.code
        matches = CodegateSignatures.find_in_string(text)
        if not matches:
            return text, []

        logger.debug(f"Found {len(matches)} secrets in the user message")

        # Convert line positions to absolute positions and extend boundaries
        absolute_matches: List[Tuple[int, int, Match]] = []
        for match in matches:
            start = self._get_absolute_position(match.line_number, match.start_index, text)
            end = self._get_absolute_position(match.line_number, match.end_index, text)

            # Extend boundaries to include full token
            start, end = self._extend_match_boundaries(text, start, end)

            # Get the full value
            full_value = text[start:end]
            absolute_matches.append((start, end, match._replace(value=full_value)))

        # Sort matches in reverse order to replace from end to start
        absolute_matches.sort(key=lambda x: x[0], reverse=True)

        # Make a mutable copy of the text
        protected_text = list(text)

        # Store matches for logging
        found_secrets = []

        # First pass. Replace each match with its encrypted value
        logger.info(f"\nFound {len(absolute_matches)} secrets:")
        for start, end, match in absolute_matches:
            hidden_secret = self._hide_secret(match)

            # Replace the secret in the text
            protected_text[start:end] = hidden_secret
            found_secrets.append(match)
            # Log the findings
            logger.info(
                f"\nService: {match.service}"
                f"\nType: {match.type}"
                f"\nKey: {match.secret_key}"
                f"\nOriginal: {match.value}"
                f"\nEncrypted: {hidden_secret}"
            )

        # Second pass. Notify the secrets in DB over the complete protected text.
        for _, _, match in absolute_matches:
            self._notify_secret(match, code_snippet=snippet, protected_text=protected_text)

        # Convert back to string
        protected_string = "".join(protected_text)
        return protected_string, found_secrets


class SecretsEncryptor(SecretsModifier):
    def __init__(
        self,
        sensitive_data_manager: SensitiveDataManager,
        context: PipelineContext,
        session_id: str,
    ):
        self._sensitive_data_manager = sensitive_data_manager
        self._session_id = session_id
        self._context = context
        self._name = "codegate-secrets"

        super().__init__()

    def _hide_secret(self, match: Match) -> str:
        # Encrypt and store the value
        if not self._session_id:
            raise ValueError("Session id must be provided")

        if not match.value:
            raise ValueError("Value must be provided")
        if not match.service:
            raise ValueError("Service must be provided")
        if not match.type:
            raise ValueError("Secret type must be provided")

        obj = SensitiveData(original=match.value, service=match.service, type=match.type)
        uuid_placeholder = self._sensitive_data_manager.store(self._session_id, obj)
        logger.debug(
            "Stored secret", service=match.service, type=match.type, placeholder=uuid_placeholder
        )
        return f"REDACTED<{uuid_placeholder}>"

    def _notify_secret(
        self, match: Match, code_snippet: Optional[CodeSnippet], protected_text: List[str]
    ) -> None:
        secret_lines = self._get_surrounding_secret_lines(protected_text, match.line_number)
        notify_string = (
            f"**Secret Detected** 🔒\n"
            f"- Service: {match.service}\n"
            f"- Type: {match.type}\n"
            f"- Key: {match.secret_key if match.secret_key else '(Unknown)'}\n"
            f"- Line Number: {match.line_number}\n"
            f"- Context:\n```\n{secret_lines}\n```"
        )
        self._context.add_alert(
            self._name,
            trigger_string=notify_string,
            severity_category=AlertSeverity.CRITICAL,
            code_snippet=code_snippet,
        )


class SecretsObfuscator(SecretsModifier):
    def __init__(
        self,
    ):
        super().__init__()

    def _hide_secret(self, match: Match) -> str:
        """
        Obfuscate the secret value. We use a hardcoded number of asterisks
        to not leak the length of the secret.
        """
        return "*" * 32

    def _notify_secret(
        self, match: Match, code_snippet: Optional[CodeSnippet], protected_text: List[str]
    ) -> None:
        pass


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

    def _redact_text(
        self,
        text: str,
        snippet: Optional[CodeSnippet],
        sensitive_data_manager: SensitiveDataManager,
        session_id: str,
        context: PipelineContext,
    ) -> tuple[str, List[Match]]:
        """
        Find and encrypt secrets in the given text.

        Args:
            text: The text to protect
            sensitive_data_manager: ..
            session_id: ..
            context: The pipeline context to be able to log alerts
        Returns:
            Tuple containing protected text with encrypted values and the count of redacted secrets
        """
        # Find secrets in the text
        text_encryptor = SecretsEncryptor(sensitive_data_manager, context, session_id)
        return text_encryptor.obfuscate(text, snippet)

    async def process(self, request: Any, context: PipelineContext) -> PipelineResult:
        """
        Process the request to find and protect secrets in all messages.

        Args:
            request: The chat completion request
            context: The pipeline context

        Returns:
            PipelineResult containing the processed request and context with redaction metadata
        """

        sensitive_data_manager = context.sensitive.manager
        if not sensitive_data_manager or not isinstance(
            sensitive_data_manager, SensitiveDataManager
        ):
            raise ValueError("Secrets manager not found in context")
        session_id = context.sensitive.session_id
        if not session_id:
            raise ValueError("Session ID not found in context")

        total_matches = []

        # get last user message block to get index for the first relevant user message
        last_user_message = self.get_last_user_message_block(request)
        last_assistant_idx = last_user_message[1] - 1 if last_user_message else -1

        # Process all messages
        for i, message in enumerate(request.get_messages()):
            for content in message.get_content():
                txt = content.get_text()
                if txt is not None:
                    redacted_content, secrets_matched = self._redact_message_content(
                        "".join(txt for txt in content.get_text()),
                        sensitive_data_manager,
                        session_id,
                        context,
                    )
                    content.set_text(redacted_content)
                    if i > last_assistant_idx:
                        total_matches += secrets_matched

        # Not count repeated secret matches
        request = self._finalize_redaction(context, total_matches, request)
        return PipelineResult(request=request, context=context)

    def _redact_message_content(self, message_content, sensitive_data_manager, session_id, context):
        # Extract any code snippets
        extractor = MessageCodeExtractorFactory.create_snippet_extractor(context.client)
        snippets = extractor.extract_snippets(message_content)
        redacted_snippets = {}
        total_matches = []

        for snippet in snippets:
            redacted_snippet, secrets_matched = self._redact_text(
                snippet, snippet, sensitive_data_manager, session_id, context
            )
            redacted_snippets[snippet.code] = redacted_snippet
            total_matches.extend(secrets_matched)

        non_snippet_parts = []
        last_end = 0

        for snippet in snippets:
            snippet_text = snippet.code
            start_index = message_content.find(snippet_text, last_end)
            if start_index > last_end:
                non_snippet_part = message_content[last_end:start_index]
                redacted_part, secrets_matched = self._redact_text(
                    non_snippet_part, "", sensitive_data_manager, session_id, context
                )
                non_snippet_parts.append(redacted_part)
                total_matches.extend(secrets_matched)

            non_snippet_parts.append(redacted_snippets[snippet_text])
            last_end = start_index + len(snippet_text)

        if last_end < len(message_content):
            remaining_text = message_content[last_end:]
            redacted_remaining, secrets_matched = self._redact_text(
                remaining_text, "", sensitive_data_manager, session_id, context
            )
            non_snippet_parts.append(redacted_remaining)
            total_matches.extend(secrets_matched)

        return "".join(non_snippet_parts), total_matches

    def _finalize_redaction(self, context, total_matches, new_request):
        set_secrets_value = set(match.value for match in total_matches)
        total_redacted = len(set_secrets_value)
        context.secrets_found = total_redacted > 0
        logger.info(f"Total secrets redacted since last assistant message: {total_redacted}")
        context.metadata["redacted_secrets_count"] = total_redacted
        return new_request


class SecretUnredactionStep(OutputPipelineStep):
    """Pipeline step that unredacts protected content in the stream"""

    def __init__(self):
        self.redacted_pattern = re.compile(r"REDACTED<(\$?[^>]+)>")
        self.marker_start = "REDACTED<"
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
        chunk: Any,
        context: OutputPipelineContext,
        input_context: Optional[PipelineContext] = None,
    ) -> list[Any]:
        """Process a single chunk of the stream"""
        if not input_context:
            raise ValueError("Input context not found")
        if input_context.sensitive is None or input_context.sensitive.manager is None:
            raise ValueError("Secrets manager not found in input context")
        if input_context.sensitive.session_id == "":
            raise ValueError("Session ID not found in input context")

        for content in chunk.get_content():
            # Check the buffered content
            buffered_content = "".join(context.buffer)

            # Look for complete REDACTED markers first
            match, remaining = self._find_complete_redaction(buffered_content)
            if match:
                # Found a complete marker, process it
                encrypted_value = match.group(1)
                if encrypted_value.startswith("$"):
                    encrypted_value = encrypted_value[1:]

                session_id = input_context.sensitive.session_id
                if not session_id:
                    raise ValueError("Session ID not found in context")

                original_value = input_context.sensitive.manager.get_original_value(
                    input_context.sensitive.session_id,
                    encrypted_value,
                )

                if original_value is None:
                    # If value not found, leave as is
                    original_value = match.group(0)  # Keep the REDACTED marker

                # Post an alert with the redacted content
                input_context.add_alert(self.name, trigger_string=encrypted_value)

                # Unredact the content and return the chunk
                unredacted_content = buffered_content[: match.start()] + original_value + remaining
                # Return the unredacted content up to this point
                content.set_text(unredacted_content)
                return [chunk]

            # If we have a partial marker at the end, keep buffering
            if self.marker_start in buffered_content:
                context.prefix_buffer = ""
                return []

            if self._is_partial_marker_prefix(buffered_content):
                context.prefix_buffer = buffered_content
                return []

            # No markers or partial markers, let pipeline handle the chunk normally
            text = content.get_text()
            content.set_text(context.prefix_buffer + text if text else "")
            context.prefix_buffer = ""
            return [chunk]
        else:
            return [chunk]


class SecretRedactionNotifier(OutputPipelineStep):
    """Pipeline step that notifies about redacted secrets in the stream"""

    @property
    def name(self) -> str:
        return "secret-redaction-notifier"

    def _create_chunk(self, original_chunk: Any, content: str) -> Any:
        """
        Creates a new chunk with the given content, preserving the original chunk's metadata
        """
        copy = original_chunk.model_copy(deep=True)
        copy.set_text(content)
        return copy

    async def process_chunk(
        self,
        chunk: Any,
        context: OutputPipelineContext,
        input_context: Optional[PipelineContext] = None,
    ) -> list[Any]:
        """Process a single chunk of the stream"""
        if (
            not input_context
            or not input_context.metadata
            or input_context.metadata.get("redacted_secrets_count", 0) == 0
        ):
            return [chunk]

        tool_name = next(
            (
                tool.lower()
                for tool in ["Cline", "Kodu"]
                for message in input_context.alerts_raised or []
                if tool in str(message.trigger_string or "")
            ),
            "",
        )

        # If the chunk has no content, we do not touch it, as it is
        # likely to break the communication protocol. As of the time
        # of this writing, this is probably only valid for Anthropic,
        # and we might want to abstract this away in the interface by
        # answering a question like "is this chunk modifiable?"
        if next(chunk.get_content(), None) is None:
            return [chunk]
        for content in chunk.get_content():
            if content.get_text() is None or content.get_text() == "":
                return [chunk]

        # Check if this is the first chunk (delta role will be present, others will not)
        redacted_count = input_context.metadata["redacted_secrets_count"]
        secret_text = "secret" if redacted_count == 1 else "secrets"
        # Create notification chunk
        if tool_name in ["cline", "kodu"]:
            # NOTE: Original code was ensuring that role was
            # "assistant" here, we might have to do that as well,
            # but I believe it was defensive programming or
            # leftover of some refactoring.
            notification_chunk = self._create_chunk(
                chunk,
                f"<thinking>\n🛡️ [CodeGate prevented {redacted_count} {secret_text}]"
                f"(http://localhost:9090/?view=codegate-secrets) from being leaked "
                f"by redacting them.</thinking>\n\n",
            )
        else:
            notification_chunk = self._create_chunk(
                chunk,
                f"\n🛡️ [CodeGate prevented {redacted_count} {secret_text}]"
                f"(http://localhost:9090/?view=codegate-secrets) from being leaked "
                f"by redacting them.\n\n",
            )

        # Reset the counter
        input_context.metadata["redacted_secrets_count"] = 0

        # Return both the notification and original chunk
        return [notification_chunk, chunk]
