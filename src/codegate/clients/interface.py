import re
from abc import ABC, abstractmethod
from typing import Dict, Type

from codegate.clients.clients import ClientType


class ClientInterface(ABC):
    """Secure interface for client-specific message processing"""

    @abstractmethod
    def strip_code_snippets(self, message: str) -> str:
        """Remove code blocks and file listings to prevent context pollution"""
        pass


class GenericClient(ClientInterface):
    """Default implementation with strict input validation"""

    _MARKDOWN_CODE_REGEX = re.compile(r"```.*?```", re.DOTALL)
    _MARKDOWN_FILE_LISTING = re.compile(r"⋮...*?⋮...\n\n", flags=re.DOTALL)
    _ENVIRONMENT_DETAILS = re.compile(
        r"<environment_details>.*?</environment_details>", flags=re.DOTALL
    )

    _CLI_REGEX = re.compile(r"^codegate\s+(.*)$", re.IGNORECASE)

    def strip_code_snippets(self, message: str) -> str:
        message = self._MARKDOWN_CODE_REGEX.sub("", message)
        message = self._MARKDOWN_FILE_LISTING.sub("", message)
        message = self._ENVIRONMENT_DETAILS.sub("", message)
        return message


class ClineClient(ClientInterface):
    """Cline-specific client interface"""

    _CLINE_FILE_REGEX = re.compile(
        r"(?i)<\s*file_content\s*[^>]*>.*?</\s*file_content\s*>", re.DOTALL
    )

    def __init__(self):
        self.generic_client = GenericClient()

    def strip_code_snippets(self, message: str) -> str:
        message = self.generic_client.strip_code_snippets(message)
        return self._CLINE_FILE_REGEX.sub("", message)


class ClientFactory:
    """Secure factory with updated client mappings"""

    _implementations: Dict[ClientType, Type[ClientInterface]] = {
        ClientType.GENERIC: GenericClient,
        ClientType.CLINE: ClineClient,
        ClientType.KODU: ClineClient,
    }

    @classmethod
    def create(cls, client_type: ClientType) -> ClientInterface:
        return cls._implementations.get(client_type, GenericClient)()
