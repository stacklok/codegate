from typing import Any, Dict, List, Optional, Tuple

import regex as re
import structlog

from codegate.config import Config
from codegate.db.models import AlertSeverity
from codegate.pipeline.base import (
    PipelineContext,
    PipelineResult,
    PipelineStep,
)
from codegate.pipeline.output import OutputPipelineContext, OutputPipelineStep
from codegate.pipeline.pii.analyzer import PiiAnalyzer
from codegate.pipeline.sensitive_data.manager import SensitiveData, SensitiveDataManager
from codegate.types.anthropic import UserMessage as AnthropicUserMessage
from codegate.types.ollama import UserMessage as OllamaUserMessage
from codegate.types.openai import UserMessage as OpenaiUserMessage


logger = structlog.get_logger("codegate")


def can_be_uuid(buffer):
    """
    This is a way to check if a buffer can be a UUID. It aims to return as soon as possible
    meaning that we buffer as little as possible. This is important for performance reasons
    but also to make sure other steps don't wait too long as we don't buffer more than we need to.
    """
    # UUID structure: 8-4-4-4-12 hex digits
    # Expected positions of hyphens
    hyphen_positions = {8, 13, 18, 23}

    # Maximum length of a UUID
    max_uuid_length = 36

    if buffer == "":
        return True

    # If buffer is longer than a UUID, it can't be a UUID
    if len(buffer) > max_uuid_length:
        return False

    for i, char in enumerate(buffer):
        # Check if hyphens are in the right positions
        if i in hyphen_positions:
            if char != "-":
                return False
        # Check if non-hyphen positions contain hex digits
        elif not (char.isdigit() or char.lower() in "abcdef"):
            return False

    return True


class CodegatePii(PipelineStep):
    """
    CodegatePii is a pipeline step that handles the detection and redaction of PII
    in chat completion requests.

    Methods:
        __init__:
            Initializes the CodegatePii pipeline step and sets up the SensitiveDataManager.

        name:
            Returns the name of the pipeline step.

        _get_redacted_snippet(message: str, pii_details: List[Dict[str, Any]]) -> str:
            Extracts a snippet of the message containing all detected PII.

        process(request: ChatCompletionRequest, context: PipelineContext) -> PipelineResult:
            Processes the chat completion request to detect and redact PII. Updates the request with
            anonymized text and stores PII details in the context metadata.

        restore_pii(session_id: str, anonymized_text: str) -> str:
            Restores the original PII from the anonymized text using the SensitiveDataManager.
    """

    def __init__(self, sensitive_data_manager: SensitiveDataManager):
        """Initialize the CodegatePii pipeline step."""
        super().__init__()
        self.sensitive_data_manager = sensitive_data_manager
        self.analyzer = PiiAnalyzer.get_instance()

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

    def process_results(
        self, session_id: str, text: str, results: List, context: PipelineContext
    ) -> Tuple[List, str]:
        # Track found PII
        found_pii = []

        # Log each found PII instance and anonymize
        anonymized_text = text
        for result in results:
            pii_value = text[result.start : result.end]

            # add to session store
            obj = SensitiveData(original=pii_value, service="pii", type=result.entity_type)
            uuid_placeholder = self.sensitive_data_manager.store(session_id, obj)
            anonymized_text = anonymized_text.replace(pii_value, uuid_placeholder)

            # Add to found PII list
            pii_info = {
                "type": result.entity_type,
                "value": pii_value,
                "score": result.score,
                "start": result.start,
                "end": result.end,
                "uuid_placeholder": uuid_placeholder,
            }
            found_pii.append(pii_info)

            # Log each PII detection with its UUID mapping
            logger.info(
                "PII detected and mapped",
                pii_type=result.entity_type,
                score=f"{result.score:.2f}",
                uuid=uuid_placeholder,
                # Don't log the actual PII value for security
                value_length=len(pii_value),
                session_id=session_id,
            )

        # Log summary of all PII found in this analysis
        if found_pii and context:
            # Create notification string for alert
            notify_string = (
                f"**PII Detected** 🔒\n"
                f"- Total PII Found: {len(found_pii)}\n"
                f"- Types Found: {', '.join(set(p['type'] for p in found_pii))}\n"
            )
            context.add_alert(
                self.name,
                trigger_string=notify_string,
                severity_category=AlertSeverity.CRITICAL,
            )

            logger.info(
                "PII analysis complete",
                total_pii_found=len(found_pii),
                pii_types=[p["type"] for p in found_pii],
                session_id=session_id,
            )

        # Return the anonymized text, PII details, and session store
        return found_pii, anonymized_text

    async def process(
        self, request: Any, context: PipelineContext
    ) -> PipelineResult:
        total_pii_found = 0
        all_pii_details: List[Dict[str, Any]] = []
        last_redacted_text = ""
        session_id = context.sensitive.session_id

        for message in request.get_messages():
            for content in message.get_content():
                # This is where analyze and anonymize the text
                if content.get_text() is None:
                    continue
                original_text = content.get_text()
                results = self.analyzer.analyze(original_text, context)
                if results:
                    pii_details, anonymized_text = self.process_results(
                        session_id, original_text, results, context
                    )

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
        context.metadata["session_id"] = session_id

        if total_pii_found > 0:
            # TODO(jakub): Storing per-step booleans is a temporary hack. We should
            # instead let the steps store the system message contents they want to
            # have added and then have a separate step that only adds them without
            # passing around bools in the context
            context.pii_found = True
            context.metadata["sensitive_data_manager"] = self.sensitive_data_manager

        logger.debug(f"Redacted text: {last_redacted_text}")

        return PipelineResult(request=request, context=context)

    def restore_pii(self, session_id: str, anonymized_text: str) -> str:
        """
        Restore the original PII (Personally Identifiable Information) in the given anonymized text.

        This method replaces placeholders in the anonymized text with their corresponding original
        PII values using the mappings stored in the provided SessionStore.

        Args:
            anonymized_text (str): The text containing placeholders for PII.
            session_id (str): The session id containing mappings of placeholders
            to original PII.

        Returns:
            str: The text with the original PII restored.
        """
        session_data = self.sensitive_data_manager.get_by_session_id(session_id)
        if not session_data:
            logger.warning(
                "No active PII session found for given session ID. Unable to restore PII."
            )
            return anonymized_text

        for uuid_placeholder, original_pii in session_data.items():
            anonymized_text = anonymized_text.replace(uuid_placeholder, original_pii)
        return anonymized_text


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
        self.redacted_pattern = re.compile(r"#([0-9a-f-]{0,36})#")
        self.complete_uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )  # noqa: E501
        self.marker_start = "#"
        self.marker_end = "#"

    @property
    def name(self) -> str:
        return "pii-unredaction-step"

    def _is_complete_uuid(self, uuid_str: str) -> bool:
        """Check if the string is a complete UUID"""
        return bool(self.complete_uuid_pattern.match(uuid_str))

    async def process_chunk(  # noqa: C901
        self,
        chunk: Any,
        context: OutputPipelineContext,
        input_context: Optional[PipelineContext] = None,
    ) -> list[Any]:
        """Process a single chunk of the stream"""
        if not input_context:
            return [chunk]

        session_id = input_context.sensitive.session_id
        if not session_id:
            logger.error("Could not get any session id, cannot process pii")
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
                start_idx = text.find(self.marker_start, current_pos)
                if start_idx == -1:
                    # No more markers!, add remaining content
                    result.append(text[current_pos:])
                    break

                end_idx = text.find(self.marker_end, start_idx+1)
                if end_idx == -1:
                    # Incomplete marker, buffer the rest only if it can be a UUID
                    if start_idx + 1 < len(content) and not can_be_uuid(content[start_idx + 1 :]):
                        # the buffer can't be a UUID, so we can't process it, just return
                        result.append(content[current_pos:])
                    else:
                        # this can still be a UUID
                        context.prefix_buffer = content[current_pos:]
                    break

                # Add text before marker
                if start_idx > current_pos:
                    result.append(text[current_pos:start_idx])

                # Extract potential UUID if it's a valid format!
                uuid_marker = text[start_idx : end_idx + 1]
                uuid_value = uuid_marker[1:-1]  # Remove # #

                if self._is_complete_uuid(uuid_value):
                    # Get the PII manager from context metadata
                    logger.debug(f"Valid UUID found: {uuid_value}")
                    sensitive_data_manager = (
                        input_context.metadata.get("sensitive_data_manager") if input_context else None
                    )
                    if sensitive_data_manager and sensitive_data_manager.session_store:
                        # Restore original value from PII manager
                        logger.debug("Attempting to restore PII from UUID marker")
                        original = sensitive_data_manager.get_original_value(session_id, uuid_marker)
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
                f"(http://localhost:9090/?view=codegate-pii) from being leaked "
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
