from typing import Optional

import structlog
from litellm import ModelResponse
from litellm.types.utils import Delta, StreamingChoices

from codegate.llm_utils.extractor import PackageExtractor
from codegate.pipeline.base import CodeSnippet, PipelineContext, PipelineSensitiveData
from codegate.pipeline.extract_snippets.extract_snippets import extract_snippets
from codegate.pipeline.output import OutputPipelineContext, OutputPipelineStep

logger = structlog.get_logger("codegate")


class CodeCommentStep(OutputPipelineStep):
    """Pipeline step that adds comments after code blocks"""

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "code-comment"

    def _create_chunk(self, original_chunk: ModelResponse, content: str) -> ModelResponse:
        """
        Creates a new chunk with the given content, preserving the original chunk's metadata
        """
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

    @staticmethod
    async def _snippet_comment(snippet: CodeSnippet, secrets: PipelineSensitiveData) -> str:
        """Create a comment for a snippet"""
        snippet.libraries = await PackageExtractor.extract_packages(
            content=snippet.code,
            provider=secrets.provider,
            model=secrets.model,
            api_key=secrets.api_key,
            base_url=secrets.api_base,
        )

        libraries_text = ""
        if snippet.libraries:
            libraries_text = " ".join(f"`{lib}`" for lib in snippet.libraries)
        if libraries_text:
            comment = f"\nThe above code snippet uses the following libraries: {libraries_text}\n"
        else:
            comment = "\ncodegate didn't detect any libraries in the snippet\n"
        comment += "\n"
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
        if not chunk.choices[0].delta.content:
            return [chunk]

        # Get current content plus this new chunk
        current_content = "".join(context.processed_content + [chunk.choices[0].delta.content])

        # Extract snippets from current content
        snippets = extract_snippets(current_content)

        # Check if a new snippet has been completed
        if len(snippets) > len(context.snippets):
            # Get the last completed snippet
            last_snippet = snippets[-1]
            context.snippets = snippets  # Update context with new snippets

            # Keep track of all the commented code
            complete_comment = ""

            # Split the chunk content if needed
            before, after = self._split_chunk_at_code_end(chunk.choices[0].delta.content)

            chunks = []

            # Add the chunk with content up to the end of code block
            if before:
                chunks.append(self._create_chunk(chunk, before))
                complete_comment += before

            comment = await self._snippet_comment(last_snippet, input_context.sensitive)
            complete_comment += comment
            chunks.append(
                self._create_chunk(
                    chunk,
                    comment,
                )
            )

            # Add the remaining content if any
            if after:
                chunks.append(self._create_chunk(chunk, after))
                complete_comment += after

            # Add an alert to the context
            input_context.add_alert(self.name, trigger_string=complete_comment)

            return chunks

        # Pass through all other content that does not create a new snippet
        return [chunk]
