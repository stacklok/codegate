from typing import Optional
from urllib.parse import quote

import structlog

from codegate.db.models import AlertSeverity
from codegate.extract_snippets.message_extractor import (
    CodeSnippet,
    DefaultCodeSnippetExtractor,
)
from codegate.pipeline.base import PipelineContext
from codegate.pipeline.output import OutputPipelineContext, OutputPipelineStep

# from codegate.pipeline.suspicious_commands.suspicious_commands import check_suspicious_code
from codegate.storage import StorageEngine
from codegate.utils.package_extractor import PackageExtractor
from codegate.types.common import ModelResponse, Delta, StreamingChoices

logger = structlog.get_logger("codegate")


class CodeCommentStep(OutputPipelineStep):
    """Pipeline step that adds comments after code blocks"""

    def __init__(self):
        self.extractor = DefaultCodeSnippetExtractor()

    @property
    def name(self) -> str:
        return "code-comment"

    def _create_chunk(self, original_chunk: ModelResponse, content: str) -> ModelResponse:
        """
        Creates a new chunk with the given content, preserving the original chunk's metadata
        """
        if isinstance(original_chunk, ModelResponse):
            return ModelResponse(
                id=original_chunk.id,
                choices=[
                    StreamingChoices(
                        finish_reason=None,
                        index=0,
                        delta=Delta(content=content, role="assistant"),
                        logprobs=None,
                    )
                ],
                created=original_chunk.created,
                model=original_chunk.model,
                object="chat.completion.chunk",
            )
        else:
            # TODO verify if deep-copy is necessary
            copy = original_chunk.model_copy(deep=True)
            copy.set_text(content)
            return copy

    async def _snippet_comment(self, snippet: CodeSnippet, context: PipelineContext) -> str:
        """Create a comment for a snippet"""
        comment = ""

        # Remove this for now. We need to find a better place for it.
        # comment, is_suspicious = await check_suspicious_code(snippet.code, snippet.language)
        # if is_suspicious:
        #     comment += comment

        snippet.libraries = PackageExtractor.extract_packages(snippet.code, snippet.language)

        # If no libraries are found, just return empty comment
        if len(snippet.libraries) == 0:
            return comment

        # Check if any of the snippet libraries is a bad package
        storage_engine = StorageEngine()
        libobjects = await storage_engine.search(
            language=snippet.language, packages=snippet.libraries
        )
        logger.info(f"Found {len(libobjects)} libraries in the storage engine")

        # If no bad packages are found, just return empty comment
        if len(libobjects) == 0:
            return ""

        # Otherwise, generate codegate warning message
        warnings = []

        # Use libobjects to generate a CSV list of bad libraries
        libobjects_text = ", ".join([f"""`{lib["properties"]["name"]}`""" for lib in libobjects])

        for lib in libobjects:
            lib_name = lib["properties"]["name"]
            lib_type = lib["properties"]["type"]
            lib_status = lib["properties"]["status"]
            lib_url = (
                f"https://www.insight.stacklok.com/report/{lib_type}/"
                f"{quote(lib_name, safe='')}?utm_source=codegate"
            )

            warnings.append(
                f"- The package `{lib_name}` is marked as **{lib_status}**.\n"
                f"- More information: [{lib_url}]({lib_url})\n"
            )

        # Add a codegate warning for the bad packages found in the snippet
        comment = f"{comment}\n\nWarning: CodeGate detected one or more potentially malicious or \
archived packages: {libobjects_text}\n"
        comment += "\n### 🚨 Warnings\n" + "\n".join(warnings) + "\n"

        # Add an alert to the context
        context.add_alert(
            self.name, trigger_string=comment, severity_category=AlertSeverity.CRITICAL
        )

        return comment

    def _split_chunk_at_code_end(self, content: str) -> tuple[str, str]:
        """Split content at the end of a code block (```)"""
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.strip() == "```":
                # Return content up to and including ```, and the rest
                before = "\n".join(lines[: i + 1])
                after = "\n".join(lines[i + 1 :])
                return before, after
        return content, ""

    async def process_chunk(
        self,
        chunk: ModelResponse,
        context: OutputPipelineContext,
        input_context: Optional[PipelineContext] = None,
    ) -> list[ModelResponse]:
        """Process a single chunk of the stream"""
        for content in chunk.get_content():
            # Get current content plus this new chunk
            text = content.get_text()
            current_content = "".join(context.processed_content + [text if text else ""])

            # Extract snippets from current content
            snippets = self.extractor.extract_snippets(current_content)

            # Check if a new snippet has been completed
            if len(snippets) > len(context.snippets):
                # Get the last completed snippet
                last_snippet = snippets[-1]
                context.snippets = snippets  # Update context with new snippets

                # Keep track of all the commented code
                # complete_comment = ""

                # Split the chunk content if needed
                text = content.get_text()
                before, after = self._split_chunk_at_code_end(text if text else "")

                chunks = []

                # Add the chunk with content up to the end of code block
                if before:
                    chunks.append(self._create_chunk(chunk, before))
                    # complete_comment += before

                comment = await self._snippet_comment(last_snippet, input_context)
                # complete_comment += comment
                chunks.append(
                    self._create_chunk(
                        chunk,
                        comment,
                    )
                )

                # Add the remaining content if any
                if after:
                    chunks.append(self._create_chunk(chunk, after))
                    # complete_comment += after

                return chunks

        # Pass through all other content that does not create a new snippet
        return [chunk]
