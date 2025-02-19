from typing import Any, Dict, List, Optional

import regex as re
import structlog

from codegate.config import Config
from codegate.pipeline.base import (
    PipelineContext,
    PipelineResult,
    PipelineStep,
)
from codegate.pipeline.output import OutputPipelineContext, OutputPipelineStep
from codegate.pipeline.pii.manager import PiiManager
from codegate.types.anthropic import UserMessage as AnthropicUserMessage
from codegate.types.ollama import UserMessage as OllamaUserMessage
from codegate.types.openai import UserMessage as OpenaiUserMessage

logger = structlog.get_logger("codegate")


class CodegatePii(PipelineStep):
    """
    CodegatePii is a pipeline step that handles the detection and redaction of PII
    in chat completion requests.

    Methods:
        __init__:
            Initializes the CodegatePii pipeline step and sets up the PiiManager.

        name:
            Returns the name of the pipeline step.

        _get_redacted_snippet(message: str, pii_details: List[Dict[str, Any]]) -> str:
            Extracts a snippet of the message containing all detected PII.

        process(request: ChatCompletionRequest, context: PipelineContext) -> PipelineResult:
            Processes the chat completion request to detect and redact PII. Updates the request with
            anonymized text and stores PII details in the context metadata.

        restore_pii(anonymized_text: str) -> str:
            Restores the original PII from the anonymized text using the PiiManager.
    """

    def __init__(self):
        """Initialize the CodegatePii pipeline step."""
        super().__init__()
        self.pii_manager = PiiManager()

    @property
    def name(self) -> str:
        return "codegate-pii"

    def _get_redacted_snippet(self, message: str, pii_details: List[Dict[str, Any]]) -> str:
        # If no PII found, return empty string
        if not pii_details:
            return ""

        # Find the first occurrence of PII and get surrounding context
        first_pii = min(pii_details, key=lambda x: x["start"])
        last_pii = max(pii_details, key=lambda x: x["end"])

        # Get the text snippet containing all PII
        start = max(0, first_pii["start"])
        end = min(len(message), last_pii["end"] + 1)

        return message[start:end]

    async def process(
        self, request: Any, context: PipelineContext
    ) -> PipelineResult:
        total_pii_found = 0
        all_pii_details: List[Dict[str, Any]] = []
        last_redacted_text = ""

        for message in request.get_messages():
            for content in message.get_content():
                # This is where analyze and anonymize the text
                if content.get_text() is None:
                    continue
                original_text = content.get_text()
                anonymized_text, pii_details = self.pii_manager.analyze(original_text, context)

                if pii_details:
                    total_pii_found += len(pii_details)
                    all_pii_details.extend(pii_details)
                    content.set_text(anonymized_text)

                    # If this is a user message, grab the redacted snippet!
                    if (
                            # This is suboptimal and should be an
                            # interface.
                            isinstance(message, AnthropicUserMessage) or
                            isinstance(message, OllamaUserMessage) or
                            isinstance(message, OpenaiUserMessage)
                    ):
                        last_redacted_text = self._get_redacted_snippet(
                            anonymized_text, pii_details
                        )

        logger.info(f"Total PII instances redacted: {total_pii_found}")

        # Store the count, details, and redacted text in context metadata
        context.metadata["redacted_pii_count"] = total_pii_found
        context.metadata["redacted_pii_details"] = all_pii_details
        context.metadata["redacted_text"] = last_redacted_text

        if total_pii_found > 0:
            context.metadata["pii_manager"] = self.pii_manager
            request.add_system_prompt(
                Config.get_config().prompts.pii_redacted,
            )

        logger.debug(f"Redacted text: {last_redacted_text}")

        return PipelineResult(request=request, context=context)

    def restore_pii(self, anonymized_text: str) -> str:
        return self.pii_manager.restore_pii(anonymized_text)


class PiiUnRedactionStep(OutputPipelineStep):
    """
    A pipeline step that processes chunks of text to unredact PII
    that has been previously redacted using UUID markers.

    Attributes:
        redacted_pattern (re.Pattern): A regex pattern to identify redacted UUID markers.
        complete_uuid_pattern (re.Pattern): A regex pattern to validate complete UUIDs.
        marker_start (str): The starting marker for redacted UUIDs.
        marker_end (str): The ending marker for redacted UUIDs.

    Methods:
        name: Returns the name of the pipeline step.
        _is_complete_uuid(uuid_str: str) -> bool: Checks if the given string is a complete UUID.
        process_chunk()
            Processes a single chunk of the stream to unredact PII.
    """

    def __init__(self):
        self.redacted_pattern = re.compile(r"<([0-9a-f-]{0,36})>")
        self.complete_uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )  # noqa: E501
        self.marker_start = "<"
        self.marker_end = ">"

    @property
    def name(self) -> str:
        return "pii-unredaction-step"

    def _is_complete_uuid(self, uuid_str: str) -> bool:
        """Check if the string is a complete UUID"""
        return bool(self.complete_uuid_pattern.match(uuid_str))

    async def process_chunk(
        self,
        chunk: Any,
        context: OutputPipelineContext,
        input_context: Optional[PipelineContext] = None,
    ) -> list[Any]:
        """Process a single chunk of the stream"""
        if not input_context:
            return [chunk]

        chunk_has_text = any(content.get_text() for content in chunk.get_content())
        if not chunk_has_text:
            return [chunk]

        for content in chunk.get_content():
            text = content.get_text()
            if text is None or text == "":
                # Nothing to do with this content item
                continue

            # Add current chunk to buffer
            if context.prefix_buffer:
                text = context.prefix_buffer + text
                context.prefix_buffer = ""

            # Find all potential UUID markers in the content
            current_pos = 0
            result = []
            while current_pos < len(text):
                start_idx = text.find("<", current_pos)
                if start_idx == -1:
                    # No more markers!, add remaining content
                    result.append(text[current_pos:])
                    break

                end_idx = text.find(">", start_idx)
                if end_idx == -1:
                    # Incomplete marker, buffer the rest
                    context.prefix_buffer = text[current_pos:]
                    break

                # Add text before marker
                if start_idx > current_pos:
                    result.append(text[current_pos:start_idx])

                # Extract potential UUID if it's a valid format!
                uuid_marker = text[start_idx : end_idx + 1]
                uuid_value = uuid_marker[1:-1]  # Remove < >

                if self._is_complete_uuid(uuid_value):
                    # Get the PII manager from context metadata
                    logger.debug(f"Valid UUID found: {uuid_value}")
                    pii_manager = input_context.metadata.get("pii_manager") if input_context else None
                    if pii_manager and pii_manager.session_store:
                        # Restore original value from PII manager
                        logger.debug("Attempting to restore PII from UUID marker")
                        original = pii_manager.session_store.get_pii(uuid_marker)
                        logger.debug(f"Restored PII: {original}")
                        result.append(original)
                    else:
                        logger.debug("No PII manager or session found, keeping original marker")
                        result.append(uuid_marker)
                else:
                    # Not a valid UUID, treat as normal text
                    logger.debug(f"Invalid UUID format: {uuid_value}")
                    result.append(uuid_marker)

                current_pos = end_idx + 1

            if result:
                # Create new chunk with processed content
                final_content = "".join(result)
                logger.debug(f"Final processed content: {final_content}")
                content.set_text(final_content)
                return [chunk]

        # If we only have buffered content, return empty list
        return []


class PiiRedactionNotifier(OutputPipelineStep):
    """
    PiiRedactionNotifier is a pipeline step that processes chunks of data to notify on redacted PII.

    Methods:
        name: Returns the name of the pipeline step.
        _create_chunk: Creates a new chunk with the given content.
        _format_pii_summary: Formats PII details into a readable summary.
        process_chunk: Processes a single chunk of stream and adds a notification if PII redacted.

    Attributes:
        None
    """

    @property
    def name(self) -> str:
        return "pii-redaction-notifier"

    def _create_chunk(self, original_chunk: Any, content: str) -> Any:
        # TODO verify if deep-copy is necessary
        copy = original_chunk.model_copy(deep=True)
        copy.set_text(content)
        return copy

    def _format_pii_summary(self, pii_details: List[Dict[str, Any]]) -> str:
        """Format PII details into a readable summary"""
        # Group PII by type
        pii_types = {}
        for pii in pii_details:
            pii_type = pii["type"]
            if pii_type not in pii_types:
                pii_types[pii_type] = 0
            pii_types[pii_type] += 1

        # Format the summary
        summary_parts = []
        for pii_type, count in pii_types.items():
            # Make the type more readable
            readable_type = pii_type.lower().replace("_", " ")
            # Add plural 's' if needed
            if count > 1:
                summary_parts.append(f"{count} {readable_type}s")
            else:
                summary_parts.append(f"{count} {readable_type}")
        logger.debug(f"PII summary: {summary_parts}")
        return ", ".join(summary_parts)

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
            or input_context.metadata.get("redacted_pii_count", 0) == 0
        ):
            return [chunk]

        is_cline_client = any(
            "Cline" in str(message.trigger_string or "")
            for message in input_context.alerts_raised or []
        )

        for content in chunk.get_content():
            # This if is a safety check for some SSE protocols
            # (e.g. Anthropic) that have different message types, some
            # of which have empty content and are not meant to be
            # modified.
            if content.get_text() is None or content.get_text() == "":
                continue

            redacted_count = input_context.metadata["redacted_pii_count"]
            pii_details = input_context.metadata.get("redacted_pii_details", [])
            pii_summary = self._format_pii_summary(pii_details)

            # The following can be uncommented to assist with debugging
            # redacted_text = input_context.metadata.get("redacted_text", "")

            # # Clean up the redacted text - remove extra newlines and spaces and backticks!
            # if redacted_text:
            #     redacted_text = " ".join(redacted_text.split())
            #     # Remove any backticks that might have come from the text
            #     redacted_text = redacted_text.replace("`", "")

            # logger.debug(f"Redacted text: {redacted_text}")

            # Create notification chunk with redacted snippet
            # TODO: Might want to check these  with James!
            notification_text = (
                f"🛡️ [CodeGate protected {redacted_count} instances of PII, including {pii_summary}]"
                f"(http://localhost:9090/?search=codegate-pii) from being leaked "
                f"by redacting them.\n\n"
            )

            # Create notification chunk for cline weirdness
            if is_cline_client:
                notification_chunk = self._create_chunk(
                    chunk,
                    f"<thinking>{notification_text}</thinking>\n",
                )
            else:
                notification_chunk = self._create_chunk(
                    chunk,
                    notification_text,
                )

            # Reset the counter
            input_context.metadata["redacted_pii_count"] = 0
            input_context.metadata["redacted_pii_details"] = []

            # Return both the notification and original chunk
            return [notification_chunk, chunk]

        # Job done son!
        return [chunk]
