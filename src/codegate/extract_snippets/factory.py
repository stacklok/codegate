from codegate.clients.clients import ClientType
from codegate.extract_snippets.body_extractor import (
    AiderBodySnippetExtractor,
    BodyCodeSnippetExtractor,
    ClineBodySnippetExtractor,
    ContinueBodySnippetExtractor,
    OpenInterpreterBodySnippetExtractor,
)


class CodeSnippetExtractorFactory:

    @staticmethod
    def create_snippet_extractor(detected_client: ClientType) -> BodyCodeSnippetExtractor:
        mapping_client_extractor = {
            ClientType.GENERIC: ContinueBodySnippetExtractor(),
            ClientType.CLINE: ClineBodySnippetExtractor(),
            ClientType.AIDER: AiderBodySnippetExtractor(),
            ClientType.OPEN_INTERPRETER: OpenInterpreterBodySnippetExtractor(),
        }
        return mapping_client_extractor.get(detected_client, ContinueBodySnippetExtractor())
