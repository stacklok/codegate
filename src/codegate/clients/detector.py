import re
from abc import ABC, abstractmethod
from functools import wraps
from typing import List, Optional

import structlog
from fastapi import Request

from codegate.clients.clients import ClientType

logger = structlog.get_logger("codegate")


class HeaderDetector:
    """
    Base utility class for header-based detection
    """

    def __init__(self, header_name: str, header_value: Optional[str] = None):
        self.header_name = header_name
        self.header_value = header_value

    def detect(self, request: Request) -> bool:
        logger.debug(
            "checking header detection",
            header_name=self.header_name,
            header_value=self.header_value,
            request_headers=dict(request.headers),
        )
        # Check if the header is present, if not we didn't detect the client
        if self.header_name not in request.headers:
            return False
        # now we know that the header is present, if we don't care about the value
        # we detected the client
        if self.header_value is None:
            return True
        # finally, if we care about the value, we need to check if it matches
        return request.headers[self.header_name] == self.header_value


class UserAgentDetector(HeaderDetector):
    """
    A variant of the HeaderDetector that specifically looks for a user-agent pattern
    """

    def __init__(self, user_agent_pattern: str):
        super().__init__("user-agent")
        self.pattern = re.compile(user_agent_pattern, re.IGNORECASE)

    def detect(self, request: Request) -> bool:
        user_agent = request.headers.get(self.header_name)
        if not user_agent:
            return False
        return bool(self.pattern.search(user_agent))


class ContentDetector:
    """
    Detector for message content patterns
    """

    def __init__(self, pattern: str):
        self.pattern = pattern

    async def detect(self, request: Request) -> bool:
        try:
            data = await request.json()
            for message in data.get("messages", []):
                message_content = str(message.get("content", ""))
                if self.pattern in message_content:
                    return True
            # This is clearly a hack and won't be needed when we get rid of the normalizers and will
            # be able to access the system message directly from the on-wire format
            system_content = str(data.get("system", ""))
            if self.pattern in system_content:
                return True
            return False
        except Exception as e:
            logger.error(f"Error in content detection: {str(e)}")
            return False


class BaseClientDetector(ABC):
    """
    Base class for all client detectors using composition of detection methods
    """

    def __init__(self):
        self.header_detector: Optional[HeaderDetector] = None
        self.user_agent_detector: Optional[UserAgentDetector] = None
        self.content_detector: Optional[ContentDetector] = None

    @property
    @abstractmethod
    def client_name(self) -> ClientType:
        """
        Returns the name of the client
        """
        pass

    async def detect(self, request: Request) -> bool:
        """
        Tries each configured detection method in sequence
        """
        # Try user agent first if configured
        if self.user_agent_detector and self.user_agent_detector.detect(request):
            return True

        # Then try header if configured
        if self.header_detector and self.header_detector.detect(request):
            return True

        # Finally try content if configured
        if self.content_detector:
            return await self.content_detector.detect(request)

        return False


class ClineDetector(BaseClientDetector):
    """
    Detector for Cline client based on message content
    """

    def __init__(self):
        super().__init__()
        self.content_detector = ContentDetector("Cline")

    @property
    def client_name(self) -> ClientType:
        return ClientType.CLINE


class KoduDetector(BaseClientDetector):
    """
    Detector for Kodu client based on message content
    """

    def __init__(self):
        super().__init__()
        self.user_agent_detector = UserAgentDetector("Kodu")
        self.content_detector = ContentDetector("Kodu")

    @property
    def client_name(self) -> ClientType:
        return ClientType.KODU


class OpenInterpreter(BaseClientDetector):
    """
    Detector for Kodu client based on message content
    """

    def __init__(self):
        super().__init__()
        self.content_detector = ContentDetector("Open Interpreter")

    @property
    def client_name(self) -> ClientType:
        return ClientType.OPEN_INTERPRETER


class CopilotDetector(HeaderDetector):
    """
    Detector for Copilot client based on user agent
    """

    def __init__(self):
        super().__init__("user-agent", "Copilot")

    @property
    def client_name(self) -> ClientType:
        return ClientType.COPILOT


class DetectClient:
    """
    Decorator class for detecting clients from request system messages

    Usage:
        @app.post("/v1/chat/completions")
        @DetectClient()
        async def chat_completions(request: Request):
            client = request.state.detected_client
    """

    def __init__(self):
        self.detectors: List[BaseClientDetector] = [
            ClineDetector(),
            KoduDetector(),
            OpenInterpreter(),
            CopilotDetector(),
        ]

    def __call__(self, func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            try:
                client = await self.detect(request)
                request.state.detected_client = client
            except Exception as e:
                logger.error(f"Error in client detection: {str(e)}")
                request.state.detected_client = ClientType.GENERIC

            return await func(request, *args, **kwargs)

        return wrapper

    async def detect(self, request: Request) -> ClientType:
        """
        Detects the client from the request by trying each detector in sequence.
        Returns the name of the first detected client, or GENERIC if no specific client is detected.
        """
        for detector in self.detectors:
            try:
                if await detector.detect(request):
                    client_name = detector.client_name
                    logger.info(f"{client_name} client detected")
                    return client_name
            except Exception as e:
                logger.error(f"Error in {detector.client_name} detection: {str(e)}")
                continue
        logger.info("No particilar client detected, using generic client")
        return ClientType.GENERIC
