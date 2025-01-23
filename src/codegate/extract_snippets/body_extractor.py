from abc import ABC, abstractmethod
from typing import List, Optional

from codegate.extract_snippets.message_extractor import (
    AiderCodeSnippetExtractor,
    ClineCodeSnippetExtractor,
    CodeSnippetExtractor,
    DefaultCodeSnippetExtractor,
    KoduCodeSnippetExtractor,
    OpenInterpreterCodeSnippetExtractor,
)
from codegate.types.common import MessageTypeFilter


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
        for msg in data.get_messages(filters=[MessageTypeFilter.USER]):
            for content in msg.get_content():
                extracted_snippets = self._snippet_extractor.extract_unique_snippets(
                    content.get_text(),
                )
                filenames.extend(extracted_snippets.keys())
        return set(filenames)

    def _extract_from_list_user_messages(self, data: dict) -> set[str]:
        filenames: List[str] = []
        for msg in data.get_messages(filters=[MessageTypeFilter.USER]):
            for content in msg.get_content():
                extracted_snippets = self._snippet_extractor.extract_unique_snippets(
                    content.get_text(),
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

    def extract_unique_filenames(self, data: dict) -> set[str]:
        return self._extract_from_list_user_messages(data)


class OpenInterpreterBodySnippetExtractor(BodyCodeSnippetExtractor):

    def __init__(self):
        self._snippet_extractor = OpenInterpreterCodeSnippetExtractor()

    def extract_unique_filenames(self, data: dict) -> set[str]:
        filenames: List[str] = []
        # Note: the previous version of this code used to analyze
        # tool-call and tool-results pairs to ensure that the regex
        # matched.
        #
        # Given it was not a business or functional requirement, but
        # rather an technical decision to avoid adding more regexes,
        # we decided to analysis contents on a per-message basis, to
        # avoid creating more dependency on the behaviour of the
        # coding assistant.
        #
        # We still filter only tool-calls and tool-results.
        filters = [MessageTypeFilter.ASSISTANT, MessageTypeFilter.TOOL]
        for msg in data.get_messages(filters=filters):
            for content in msg.get_content():
                if content.get_text() is not None:
                    extracted_snippets = self._snippet_extractor.extract_unique_snippets(
                        content.get_text()
                    )
                    filenames.extend(extracted_snippets.keys())
        return set(filenames)


class KoduBodySnippetExtractor(BodyCodeSnippetExtractor):

    def __init__(self):
        self._snippet_extractor = KoduCodeSnippetExtractor()

    def extract_unique_filenames(self, data: dict) -> set[str]:
        return self._extract_from_list_user_messages(data)
