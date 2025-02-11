from codegate.clients.clients import ClientType
from codegate.extract_snippets.body_extractor import (
    AiderBodySnippetExtractor,
    BodyCodeSnippetExtractor,
    ClineBodySnippetExtractor,
    ContinueBodySnippetExtractor,
    KoduBodySnippetExtractor,
    OpenInterpreterBodySnippetExtractor,
)
from codegate.extract_snippets.message_extractor import (
    AiderCodeSnippetExtractor,
    ClineCodeSnippetExtractor,
    CodeSnippetExtractor,
    DefaultCodeSnippetExtractor,
    KoduCodeSnippetExtractor,
    OpenInterpreterCodeSnippetExtractor,
)


class BodyCodeExtractorFactory:

    @staticmethod
    def create_snippet_extractor(detected_client: ClientType) -> BodyCodeSnippetExtractor:
        mapping_client_extractor = {
            ClientType.GENERIC: ContinueBodySnippetExtractor(),
            ClientType.CLINE: ClineBodySnippetExtractor(),
            ClientType.AIDER: AiderBodySnippetExtractor(),
            ClientType.OPEN_INTERPRETER: OpenInterpreterBodySnippetExtractor(),
            ClientType.KODU: KoduBodySnippetExtractor(),
        }
        return mapping_client_extractor.get(detected_client, ContinueBodySnippetExtractor())


class MessageCodeExtractorFactory:

    @staticmethod
    def create_snippet_extractor(detected_client: ClientType) -> CodeSnippetExtractor:
        mapping_client_extractor = {
            ClientType.GENERIC: DefaultCodeSnippetExtractor(),
            ClientType.CLINE: ClineCodeSnippetExtractor(),
            ClientType.AIDER: AiderCodeSnippetExtractor(),
            ClientType.OPEN_INTERPRETER: OpenInterpreterCodeSnippetExtractor(),
            ClientType.KODU: KoduCodeSnippetExtractor(),
        }
        return mapping_client_extractor.get(detected_client, DefaultCodeSnippetExtractor())
