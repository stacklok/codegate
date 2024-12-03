from typing import Optional

import structlog
from litellm import ModelResponse
from litellm.types.utils import Delta, StreamingChoices

from codegate.pipeline.base import PipelineContext
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

            # Split the chunk content if needed
            before, after = self._split_chunk_at_code_end(chunk.choices[0].delta.content)

            chunks = []

            # Add the chunk with content up to the end of code block
            if before:
                chunks.append(self._create_chunk(chunk, before))

            # Add the comment
            chunks.append(
                self._create_chunk(
                    chunk, f"\nThe above is a {last_snippet.language or 'unknown'} code snippet\n\n"
                )
            )

            # Add the remaining content if any
            if after:
                chunks.append(self._create_chunk(chunk, after))

            return chunks

        # Pass through all other content that does not create a new snippet
        return [chunk]
