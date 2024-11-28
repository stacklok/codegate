import os
import re
from typing import List, Optional

import structlog
from litellm.types.llms.openai import ChatCompletionRequest

from codegate.pipeline.base import CodeSnippet, PipelineContext, PipelineResult, PipelineStep

CODE_BLOCK_PATTERN = re.compile(
    r"```(?:(?P<language>\w+)\s+)?(?P<filename>[^\s\(]+)?(?:\s*\((?P<lineinfo>[^)]+)\))?\n(?P<content>(?:.|\n)*?)```"
)

logger = structlog.get_logger("codegate")

def ecosystem_from_filepath(filepath: str) -> Optional[str]:
    """
    Determine language from filepath.

    Args:
        filepath: Path to the file

    Returns:
        Determined language based on file extension
    """
    # Implement file extension to language mapping
    extension_mapping = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
    }

    # Get the file extension
    ext = os.path.splitext(filepath)[1].lower()
    return extension_mapping.get(ext, None)


def ecosystem_from_message(message: str) -> Optional[str]:
    """
    Determine language from message.

    Args:
        message: The language from the message. Some extensions send a different
        format where the language is present in the snippet,
        e.g. "py /path/to/file (lineFrom-lineTo)"

    Returns:
        Determined language based on message content
    """
    language_mapping = {
        "py": "python",
        "js": "javascript",
        "ts": "typescript",
        "tsx": "typescript",
        "go": "go",
    }
    return language_mapping.get(message, None)


def extract_snippets(message: str) -> List[CodeSnippet]:
    """
    Extract code snippets from a message.

    Args:
        message: Input text containing code snippets

    Returns:
        List of extracted code snippets
    """
    # Regular expression to find code blocks

    snippets: List[CodeSnippet] = []

    # Find all code block matches
    for match in CODE_BLOCK_PATTERN.finditer(message):
        filename = match.group("filename")
        content = match.group("content")
        matched_language = match.group("language")

        # Determine language
        lang = None
        if matched_language:
            lang = ecosystem_from_message(matched_language.strip())
        if lang is None and filename:
            filename = filename.strip()
            # Determine language from the filename
            lang = ecosystem_from_filepath(filename)

        snippets.append(CodeSnippet(filepath=filename, code=content, language=lang))

    return snippets


class CodeSnippetExtractor(PipelineStep):
    """
    Pipeline step that merely extracts code snippets from the user message.
    """

    def __init__(self):
        """Initialize the CodeSnippetExtractor pipeline step."""
        super().__init__()

    @property
    def name(self) -> str:
        return "code-snippet-extractor"

    async def process(
        self,
        request: ChatCompletionRequest,
        context: PipelineContext,
    ) -> PipelineResult:
        last_user_message = self.get_last_user_message(request)
        if not last_user_message:
            return PipelineResult(request=request, context=context)
        msg_content, _ = last_user_message
        snippets = extract_snippets(msg_content)

        logger.info(f"Extracted {len(snippets)} code snippets from the user message")

        if len(snippets) > 0:
            for snippet in snippets:
                logger.debug(f"Code snippet: {snippet}")
                context.add_code_snippet(snippet)

        return PipelineResult(
            context=context,
        )
