from abc import ABC, abstractmethod
from typing import List, Optional

from codegate.extract_snippets.message_extractor import (
    AiderCodeSnippetExtractor,
    ClineCodeSnippetExtractor,
    CodeSnippetExtractor,
    DefaultCodeSnippetExtractor,
    OpenInterpreterCodeSnippetExtractor,
)


class BodyCodeSnippetExtractorError(Exception):
    pass


class BodyCodeSnippetExtractor(ABC):

    def __init__(self):
        # Initialize the extractor in parent class. The child classes will set the extractor.
        self._snippet_extractor: Optional[CodeSnippetExtractor] = None

    def _extract_from_user_messages(self, data: dict) -> set[str]:
        """
        The method extracts the code snippets from the user messages in the data got from the
        clients.

        It returns a set of filenames extracted from the code snippets.
        """
        if self._snippet_extractor is None:
            raise BodyCodeSnippetExtractorError("Code Extractor not set.")

        filenames: List[str] = []
        for msg in data.get("messages", []):
            if msg.get("role", "") == "user":
                extracted_snippets = self._snippet_extractor.extract_unique_snippets(
                    msg.get("content")
                )
                filenames.extend(extracted_snippets.keys())
        return set(filenames)

    @abstractmethod
    def extract_unique_filenames(self, data: dict) -> set[str]:
        """
        Extract the unique filenames from the data received by the clients (Cline, Continue, ...)
        """
        pass


class ContinueBodySnippetExtractor(BodyCodeSnippetExtractor):

    def __init__(self):
        self._snippet_extractor = DefaultCodeSnippetExtractor()

    def extract_unique_filenames(self, data: dict) -> set[str]:
        return self._extract_from_user_messages(data)


class AiderBodySnippetExtractor(BodyCodeSnippetExtractor):

    def __init__(self):
        self._snippet_extractor = AiderCodeSnippetExtractor()

    def extract_unique_filenames(self, data: dict) -> set[str]:
        return self._extract_from_user_messages(data)


class ClineBodySnippetExtractor(BodyCodeSnippetExtractor):

    def __init__(self):
        self._snippet_extractor = ClineCodeSnippetExtractor()

    def _extract_from_user_messages(self, data: dict) -> set[str]:
        """
        The method extracts the code snippets from the user messages in the data got from Cline.

        It returns a set of filenames extracted from the code snippets.
        """

        filenames: List[str] = []
        for msg in data.get("messages", []):
            if msg.get("role", "") == "user":
                msgs_content = msg.get("content", [])
                for msg_content in msgs_content:
                    if msg_content.get("type", "") == "text":
                        extracted_snippets = self._snippet_extractor.extract_unique_snippets(
                            msg_content.get("text")
                        )
                        filenames.extend(extracted_snippets.keys())
        return set(filenames)

    def extract_unique_filenames(self, data: dict) -> set[str]:
        return self._extract_from_user_messages(data)


class OpenInterpreterBodySnippetExtractor(BodyCodeSnippetExtractor):

    def __init__(self):
        self._snippet_extractor = OpenInterpreterCodeSnippetExtractor()

    def _is_msg_tool_call(self, msg: dict) -> bool:
        return msg.get("role", "") == "assistant" and msg.get("tool_calls", [])

    def _is_msg_tool_result(self, msg: dict) -> bool:
        return msg.get("role", "") == "tool" and msg.get("content", "")

    def _extract_args_from_tool_call(self, msg: dict) -> str:
        """
        Extract the arguments from the tool call message.
        """
        tool_calls = msg.get("tool_calls", [])
        if not tool_calls:
            return ""
        return tool_calls[0].get("function", {}).get("arguments", "")

    def _extract_result_from_tool_result(self, msg: dict) -> str:
        """
        Extract the result from the tool result message.
        """
        return msg.get("content", "")

    def extract_unique_filenames(self, data: dict) -> set[str]:
        messages = data.get("messages", [])
        if not messages:
            return set()

        filenames: List[str] = []
        for i_msg in range(len(messages) - 1):
            msg = messages[i_msg]
            next_msg = messages[i_msg + 1]
            if self._is_msg_tool_call(msg) and self._is_msg_tool_result(next_msg):
                tool_args = self._extract_args_from_tool_call(msg)
                tool_response = self._extract_result_from_tool_result(next_msg)
                extracted_snippets = self._snippet_extractor.extract_unique_snippets(
                    f"{tool_args}\n{tool_response}"
                )
                filenames.extend(extracted_snippets.keys())
        return set(filenames)
