import os
import re
from typing import List, Optional

import structlog
from litellm.types.llms.openai import ChatCompletionRequest
from pygments.lexers import guess_lexer

from codegate.pipeline.base import CodeSnippet, PipelineContext, PipelineResult, PipelineStep

CODE_BLOCK_PATTERN = re.compile(
    r"```"  # Opening backticks, no whitespace after backticks and before language
    r"(?:(?P<language>[a-zA-Z0-9_+-]+)\s+)?"  # Language must be followed by whitespace if present
    r"(?:(?P<filename>[^\s\(\n]+))?"  # Optional filename (cannot contain spaces or parentheses)
    r"(?:\s+\([0-9]+-[0-9]+\))?"  # Optional line numbers in parentheses
    r"\s*\n"  # Required newline after metadata
    r"(?P<content>.*?)"  # Content (non-greedy match)
    r"```",  # Closing backticks
    re.DOTALL,
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
        "rs": "rust",
        "java": "java",
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
    available_languages = ["python", "javascript", "typescript", "go", "rust", "java"]

    # Find all code block matches
    for match in CODE_BLOCK_PATTERN.finditer(message):
        matched_language = match.group("language") if match.group("language") else None
        filename = match.group("filename") if match.group("filename") else None
        content = match.group("content")

        # If we have a single word without extension after the backticks,
        # it's a language identifier, not a filename. Typicaly used in the
        # format ` ```python ` in output snippets
        if filename and not matched_language and "." not in filename:
            lang = filename
            if lang not in available_languages:
                #  try to get it from the extension
                lang = ecosystem_from_message(filename)
                if lang not in available_languages:
                    lang = None
            filename = None
        else:
            # Determine language from the message, either by the short
            # language identifier or by the filename
            lang = None
            if matched_language:
                lang = ecosystem_from_message(matched_language.strip())
            if lang is None and filename:
                filename = filename.strip()
                # Determine language from the filename
                lang = ecosystem_from_filepath(filename)
            if lang is None:
                # try to guess it from the code
                lexer = guess_lexer(content)
                if lexer and lexer.name:
                    lang = lexer.name.lower()
                    # only add available languages
                    if lang not in available_languages:
                        lang = None

        #  just correct the typescript exception
        lang_map = {"typescript": "javascript"}
        if lang:
            lang = lang_map.get(lang, lang)
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
        last_message = self.get_last_user_message_block(request, context.client)
        if not last_message:
            return PipelineResult(request=request, context=context)
        msg_content, _ = last_message
        snippets = extract_snippets(msg_content)

        logger.info(f"Extracted {len(snippets)} code snippets from the user message")

        if len(snippets) > 0:
            for snippet in snippets:
                context.add_alert(self.name, code_snippet=snippet)
                logger.debug(f"Code snippet: {snippet}")
                context.add_code_snippet(snippet)

        return PipelineResult(
            context=context,
        )
