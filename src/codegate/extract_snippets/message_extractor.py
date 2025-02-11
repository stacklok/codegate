import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Self

import structlog
from pydantic import BaseModel, field_validator, model_validator
from pygments.lexers import guess_lexer

logger = structlog.get_logger("codegate")

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

CODE_BLOCK_WITH_FILENAME_PATTERN = re.compile(
    r"```"  # Opening backticks, no whitespace after backticks and before language
    r"(?:(?P<language>[a-zA-Z0-9_+-]+)\s+)?"  # Language must be followed by whitespace if present
    r"(?P<filename>[^\s\(\n]+)"  # Mandatory filename (cannot contain spaces or parentheses)
    r"(?:\s+\([0-9]+-[0-9]+\))?"  # Optional line numbers in parentheses
    r"\s*\n"  # Required newline after metadata
    r"(?P<content>.*?)"  # Content (non-greedy match)
    r"```",  # Closing backticks
    re.DOTALL,
)

CLINE_FILE_CONTENT_PATTERN = re.compile(
    r"<file_content\s+path=\"(?P<filename>[^\"]+)\">"  # Match the opening tag with mandatory file
    r"(?P<content>.*?)"  # Match the content (non-greedy)
    r"</file_content>",  # Match the closing tag
    re.DOTALL,
)

AIDER_SUMMARIES_CONTENT_PATTERN = re.compile(
    r"^(?P<filename>[^\n]+):\n"  # Match the filepath as the header
    r"(?P<content>.*?)"  # Match the content (non-greedy)
    r"⋮...\n\n",  # Match the ending pattern with dots
    re.DOTALL | re.MULTILINE,
)

AIDER_FILE_CONTENT_PATTERN = re.compile(
    r"^(?P<filename>[^\n]+)\n"  # Match the filepath as the header
    r"```"  # Match the opening triple backticks
    r"(?P<content>.*?)"  # Match the content (non-greedy)
    r"```",  # Match the closing triple backticks
    re.DOTALL | re.MULTILINE,
)

OPEN_INTERPRETER_CONTENT_PATTERN = re.compile(
    r"# Attempting to read the content of `(?P<filename>[^`]+)`"  # Match the filename backticks
    r".*?"  # Match any characters non-greedily
    r"File read successfully\.\n"  # Match the "File read successfully." text
    r"'(?P<content>.*?)'",  # Match the content wrapped in single quotes
    re.DOTALL,
)

OPEN_INTERPRETER_Y_CONTENT_PATTERN = re.compile(
    r"# Open and read the contents of the (?P<filename>[^\s]+) file"  # Match the filename
    r".*?"  # Match any characters non-greedily
    r"\n\n"  # Match the double line break
    r"(?P<content>.*)",  # Match everything that comes after the double line break
    re.DOTALL,
)

KODU_CONTENT_PATTERN = re.compile(
    r"<file\s+path=\"(?P<filename>[^\n>]+)\">"  # Match the opening tag with path attribute
    r"(?P<content>.*?)"  # Match the content (non-greedy)
    r"</file>",  # Match the closing tag
    re.DOTALL,
)


class MatchedPatternSnippet(BaseModel):
    """
    Represents a match from the code snippet patterns.
    Meant to be used by all CodeSnippetExtractors.
    """

    language: Optional[str]
    filename: Optional[str]
    content: str


class CodeSnippet(BaseModel):
    """
    Represents a code snippet with its programming language.

    Args:
        language: The programming language identifier (e.g., 'python', 'javascript')
        code: The actual code content
    """

    code: str
    language: Optional[str]
    filepath: Optional[str]
    libraries: List[str] = []
    file_extension: Optional[str] = None

    @field_validator("language", mode="after")
    @classmethod
    def ensure_lowercase(cls, value: str) -> str:
        if value is not None:
            value = value.strip().lower()
        return value

    @model_validator(mode="after")
    def fill_file_extension(self) -> Self:
        if self.filepath is not None:
            self.file_extension = Path(self.filepath).suffix
        return self


class CodeSnippetExtractor(ABC):

    def __init__(self):
        self._extension_mapping = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
        }
        self._language_mapping = {
            "py": "python",
            "js": "javascript",
            "ts": "typescript",
            "tsx": "typescript",
            "go": "go",
            "rs": "rust",
            "java": "java",
        }
        self._available_languages = ["python", "javascript", "typescript", "go", "rust", "java"]

    @property
    @abstractmethod
    def codeblock_pattern(self) -> List[re.Pattern]:
        """
        List of regex patterns to match code blocks without filenames.
        """
        pass

    @property
    @abstractmethod
    def codeblock_with_filename_pattern(self) -> List[re.Pattern]:
        """
        List of regex patterns to match code blocks with filenames.
        """
        pass

    @abstractmethod
    def _get_match_pattern_snippet(self, match: re.Match) -> MatchedPatternSnippet:
        pass

    def _choose_regex(self, require_filepath: bool) -> List[re.Pattern]:
        if require_filepath:
            return self.codeblock_with_filename_pattern
        else:
            return self.codeblock_pattern

    def _ecosystem_from_filepath(self, filepath: str):
        """
        Determine language from filepath.

        Args:
            filepath: Path to the file

        Returns:
            Determined language based on file extension
        """

        # Get the file extension
        path_filename = Path(filepath)
        file_extension = path_filename.suffix.lower()
        return self._extension_mapping.get(file_extension, None)

    def _ecosystem_from_message(self, message: str):
        """
        Determine language from message.

        Args:
            message: The language from the message. Some extensions send a different
            format where the language is present in the snippet,
            e.g. "py /path/to/file (lineFrom-lineTo)"

        Returns:
            Determined language based on message content
        """
        return self._language_mapping.get(message, None)

    def _get_snippet_for_match(self, match: re.Match) -> CodeSnippet:
        matched_snippet = self._get_match_pattern_snippet(match)

        # If we have a single word without extension after the backticks,
        # it's a language identifier, not a filename. Typicaly used in the
        # format ` ```python ` in output snippets
        if (
            matched_snippet.filename
            and not matched_snippet.language
            and "." not in matched_snippet.filename
        ):
            lang = matched_snippet.filename
            if lang not in self._available_languages:
                #  try to get it from the extension
                lang = self._ecosystem_from_message(matched_snippet.filename)
                if lang not in self._available_languages:
                    lang = None
            matched_snippet.filename = None
        else:
            # Determine language from the message, either by the short
            # language identifier or by the filename
            lang = None
            if matched_snippet.language:
                lang = self._ecosystem_from_message(matched_snippet.language.strip())
            if lang is None and matched_snippet.filename:
                matched_snippet.filename = matched_snippet.filename.strip()
                # Determine language from the filename
                lang = self._ecosystem_from_filepath(matched_snippet.filename)
            if lang is None:
                # try to guess it from the code
                lexer = guess_lexer(matched_snippet.content)
                if lexer and lexer.name:
                    lang = lexer.name.lower()
                    # only add available languages
                    if lang not in self._available_languages:
                        lang = None

        # just correct the typescript exception
        lang_map = {"typescript": "javascript"}
        if lang:
            lang = lang_map.get(lang, lang)
        return CodeSnippet(
            filepath=matched_snippet.filename, code=matched_snippet.content, language=lang
        )

    def extract_snippets(self, message: str, require_filepath: bool = False) -> List[CodeSnippet]:
        """
        Extract code snippets from a message.

        Args:
            message: Input text containing code snippets

        Returns:
            List of extracted code snippets
        """
        regexes = self._choose_regex(require_filepath)
        # Find all code block matches
        if isinstance(message, str):
            return [
                self._get_snippet_for_match(match)
                for regex in regexes
                for match in regex.finditer(message)
            ]
        return [
            self._get_snippet_for_match(match)
            for regex in regexes
            for match in regex.finditer(message)
        ]

    def extract_unique_snippets(self, message: str) -> Dict[str, CodeSnippet]:
        """
        Extract unique filpaths from a message. Uses the filepath as key.

        Args:
            message: Input text containing code snippets

        Returns:
            Dictionary of unique code snippets with the filepath as key
        """
        regexes = self._choose_regex(require_filepath=True)
        unique_snippets: Dict[str, CodeSnippet] = {}
        for regex in regexes:
            for match in regex.finditer(message):
                snippet = self._get_snippet_for_match(match)
                filename = Path(snippet.filepath).name if snippet.filepath else None
                if filename and filename not in unique_snippets:
                    unique_snippets[filename] = snippet

        return unique_snippets


class DefaultCodeSnippetExtractor(CodeSnippetExtractor):

    @property
    def codeblock_pattern(self) -> re.Pattern:
        return [CODE_BLOCK_PATTERN]

    @property
    def codeblock_with_filename_pattern(self) -> re.Pattern:
        return [CODE_BLOCK_WITH_FILENAME_PATTERN]

    def _get_match_pattern_snippet(self, match: re.Match) -> MatchedPatternSnippet:
        matched_language = match.group("language") if match.group("language") else None
        filename = match.group("filename") if match.group("filename") else None
        content = match.group("content")
        return MatchedPatternSnippet(language=matched_language, filename=filename, content=content)


class ClineCodeSnippetExtractor(CodeSnippetExtractor):

    @property
    def codeblock_pattern(self) -> re.Pattern:
        return [CLINE_FILE_CONTENT_PATTERN]

    @property
    def codeblock_with_filename_pattern(self) -> re.Pattern:
        return [CLINE_FILE_CONTENT_PATTERN]

    def _get_match_pattern_snippet(self, match: re.Match) -> MatchedPatternSnippet:
        # We don't have language in the cline pattern
        matched_language = None
        filename = match.group("filename")
        content = match.group("content")
        return MatchedPatternSnippet(language=matched_language, filename=filename, content=content)


class AiderCodeSnippetExtractor(CodeSnippetExtractor):

    @property
    def codeblock_pattern(self) -> re.Pattern:
        return [AIDER_SUMMARIES_CONTENT_PATTERN, AIDER_FILE_CONTENT_PATTERN]

    @property
    def codeblock_with_filename_pattern(self) -> re.Pattern:
        return [AIDER_SUMMARIES_CONTENT_PATTERN, AIDER_FILE_CONTENT_PATTERN]

    def _get_match_pattern_snippet(self, match: re.Match) -> MatchedPatternSnippet:
        # We don't have language in the cline pattern
        matched_language = None
        filename = match.group("filename")
        content = match.group("content")
        return MatchedPatternSnippet(language=matched_language, filename=filename, content=content)


class OpenInterpreterCodeSnippetExtractor(CodeSnippetExtractor):

    @property
    def codeblock_pattern(self) -> re.Pattern:
        return [OPEN_INTERPRETER_CONTENT_PATTERN, OPEN_INTERPRETER_Y_CONTENT_PATTERN]

    @property
    def codeblock_with_filename_pattern(self) -> re.Pattern:
        return [OPEN_INTERPRETER_CONTENT_PATTERN, OPEN_INTERPRETER_Y_CONTENT_PATTERN]

    def _get_match_pattern_snippet(self, match: re.Match) -> MatchedPatternSnippet:
        # We don't have language in the cline pattern
        matched_language = None
        filename = match.group("filename")
        content = match.group("content")
        return MatchedPatternSnippet(language=matched_language, filename=filename, content=content)


class KoduCodeSnippetExtractor(CodeSnippetExtractor):

    @property
    def codeblock_pattern(self) -> re.Pattern:
        return [KODU_CONTENT_PATTERN]

    @property
    def codeblock_with_filename_pattern(self) -> re.Pattern:
        return [KODU_CONTENT_PATTERN]

    def _get_match_pattern_snippet(self, match: re.Match) -> MatchedPatternSnippet:
        # We don't have language in the cline pattern
        matched_language = None
        filename = match.group("filename")
        content = match.group("content")
        return MatchedPatternSnippet(language=matched_language, filename=filename, content=content)
