import json
from unittest.mock import Mock

import pytest
from fastapi import Request
from fastapi.datastructures import Headers

from codegate.clients.clients import ClientType
from codegate.clients.detector import (
    BaseClientDetector,
    ClineDetector,
    ContentDetector,
    CopilotDetector,
    DetectClient,
    HeaderDetector,
    KoduDetector,
    OpenInterpreter,
    UserAgentDetector,
)


@pytest.fixture
def mock_request():
    """Create a mock FastAPI request with configurable headers and body"""

    async def get_json():
        return {"messages": []}

    request = Mock(spec=Request)
    request.headers = Headers()
    request.json = get_json
    return request


@pytest.fixture
def mock_request_with_messages(mock_request):
    """Create a mock request with configurable message content"""

    async def get_json():
        return {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "system", "content": "Test message"},
            ]
        }

    mock_request.json = get_json
    return mock_request


class TestHeaderDetector:
    def test_header_present_with_matching_value(self, mock_request):
        detector = HeaderDetector("X-Test-Header", "test-value")
        mock_request.headers = Headers({"X-Test-Header": "test-value"})
        assert detector.detect(mock_request) is True

    def test_header_present_without_value_check(self, mock_request):
        detector = HeaderDetector("X-Test-Header")
        mock_request.headers = Headers({"X-Test-Header": "any-value"})
        assert detector.detect(mock_request) is True

    def test_header_missing(self, mock_request):
        detector = HeaderDetector("X-Test-Header", "test-value")
        assert detector.detect(mock_request) is False

    def test_header_present_with_non_matching_value(self, mock_request):
        detector = HeaderDetector("X-Test-Header", "test-value")
        mock_request.headers = Headers({"X-Test-Header": "wrong-value"})
        assert detector.detect(mock_request) is False


class TestUserAgentDetector:
    def test_matching_user_agent_pattern(self, mock_request):
        detector = UserAgentDetector("Test.*Browser")
        mock_request.headers = Headers({"user-agent": "Test/1.0 Browser"})
        assert detector.detect(mock_request) is True

    def test_non_matching_user_agent_pattern(self, mock_request):
        detector = UserAgentDetector("Test.*Browser")
        mock_request.headers = Headers({"user-agent": "Different/1.0 Agent"})
        assert detector.detect(mock_request) is False

    def test_missing_user_agent(self, mock_request):
        detector = UserAgentDetector("Test.*Browser")
        assert detector.detect(mock_request) is False

    def test_case_insensitive_matching(self, mock_request):
        detector = UserAgentDetector("test.*browser")
        mock_request.headers = Headers({"user-agent": "TEST/1.0 BROWSER"})
        assert detector.detect(mock_request) is True


class TestContentDetector:
    @pytest.mark.asyncio
    async def test_matching_content_pattern(self, mock_request):
        detector = ContentDetector("test-pattern")

        async def get_json():
            return {"messages": [{"content": "this is a test-pattern message"}]}

        mock_request.json = get_json
        assert await detector.detect(mock_request) is True

    @pytest.mark.asyncio
    async def test_non_matching_content_pattern(self, mock_request):
        detector = ContentDetector("test-pattern")

        async def get_json():
            return {"messages": [{"content": "this is a different message"}]}

        mock_request.json = get_json
        assert await detector.detect(mock_request) is False

    @pytest.mark.asyncio
    async def test_malformed_json(self, mock_request):
        detector = ContentDetector("test-pattern")

        async def get_json():
            raise json.JSONDecodeError("Invalid JSON", "", 0)

        mock_request.json = get_json
        assert await detector.detect(mock_request) is False

    @pytest.mark.asyncio
    async def test_empty_messages(self, mock_request):
        detector = ContentDetector("test-pattern")

        async def get_json():
            return {"messages": []}

        mock_request.json = get_json
        assert await detector.detect(mock_request) is False

    @pytest.mark.asyncio
    async def test_missing_content_field(self, mock_request):
        detector = ContentDetector("test-pattern")

        async def get_json():
            return {"messages": [{"role": "user"}]}

        mock_request.json = get_json
        assert await detector.detect(mock_request) is False


class MockClientDetector(BaseClientDetector):
    """Mock implementation of BaseClientDetector for testing"""

    @property
    def client_name(self) -> ClientType:
        return ClientType.GENERIC


class TestBaseClientDetector:
    @pytest.mark.asyncio
    async def test_user_agent_detection(self, mock_request):
        detector = MockClientDetector()
        detector.user_agent_detector = UserAgentDetector("Test.*Browser")
        mock_request.headers = Headers({"user-agent": "Test/1.0 Browser"})
        assert await detector.detect(mock_request) is True

    @pytest.mark.asyncio
    async def test_header_detection(self, mock_request):
        detector = MockClientDetector()
        detector.header_detector = HeaderDetector("X-Test", "value")
        mock_request.headers = Headers({"X-Test": "value"})
        assert await detector.detect(mock_request) is True

    @pytest.mark.asyncio
    async def test_content_detection(self, mock_request):
        detector = MockClientDetector()
        detector.content_detector = ContentDetector("test-pattern")

        async def get_json():
            return {"messages": [{"content": "test-pattern"}]}

        mock_request.json = get_json
        assert await detector.detect(mock_request) is True

    @pytest.mark.asyncio
    async def test_no_detectors_configured(self, mock_request):
        detector = MockClientDetector()
        assert await detector.detect(mock_request) is False


class TestClineDetector:
    @pytest.mark.asyncio
    async def test_successful_detection(self, mock_request):
        detector = ClineDetector()

        async def get_json():
            return {"messages": [{"content": "Cline"}]}

        mock_request.json = get_json
        assert await detector.detect(mock_request) is True
        assert detector.client_name == ClientType.CLINE

    @pytest.mark.asyncio
    async def test_failed_detection(self, mock_request):
        detector = ClineDetector()

        async def get_json():
            return {"messages": [{"content": "Different Client"}]}

        mock_request.json = get_json
        assert await detector.detect(mock_request) is False


class TestKoduDetector:
    @pytest.mark.asyncio
    async def test_user_agent_detection(self, mock_request):
        detector = KoduDetector()
        mock_request.headers = Headers({"user-agent": "Kodu"})
        assert await detector.detect(mock_request) is True
        assert detector.client_name == ClientType.KODU

    @pytest.mark.asyncio
    async def test_content_detection(self, mock_request):
        detector = KoduDetector()
        mock_request.headers = Headers({"user-agent": "Different Client"})

        async def get_json():
            return {"messages": [{"content": "Kodu"}]}

        mock_request.json = get_json
        assert await detector.detect(mock_request) is True
        assert detector.client_name == ClientType.KODU

    @pytest.mark.asyncio
    async def test_failed_detection(self, mock_request):
        detector = KoduDetector()
        mock_request.headers = Headers({"user-agent": "Different Client"})

        async def get_json():
            return {"messages": [{"content": "Different Client"}]}

        mock_request.json = get_json
        assert await detector.detect(mock_request) is False

    @pytest.mark.asyncio
    async def test_no_user_agent_content_detection(self, mock_request):
        detector = KoduDetector()

        async def get_json():
            return {"messages": [{"content": "Kodu"}]}

        mock_request.json = get_json
        assert await detector.detect(mock_request) is True
        assert detector.client_name == ClientType.KODU


class TestOpenInterpreterDetector:
    @pytest.mark.asyncio
    async def test_successful_detection(self, mock_request):
        detector = OpenInterpreter()

        async def get_json():
            return {"messages": [{"content": "Open Interpreter"}]}

        mock_request.json = get_json
        assert await detector.detect(mock_request) is True
        assert detector.client_name == ClientType.OPEN_INTERPRETER

    @pytest.mark.asyncio
    async def test_failed_detection(self, mock_request):
        detector = OpenInterpreter()

        async def get_json():
            return {"messages": [{"content": "Different Client"}]}

        mock_request.json = get_json
        assert await detector.detect(mock_request) is False


class TestCopilotDetector:
    @pytest.mark.asyncio
    async def test_successful_detection(self, mock_request):
        detector = CopilotDetector()
        mock_request.headers = Headers({"user-agent": "Copilot"})
        assert await detector.detect(mock_request) is True
        assert detector.client_name == ClientType.COPILOT

    @pytest.mark.asyncio
    async def test_failed_detection(self, mock_request):
        detector = CopilotDetector()
        mock_request.headers = Headers({"user-agent": "Different Client"})
        assert await detector.detect(mock_request) is False

    @pytest.mark.asyncio
    async def test_missing_user_agent(self, mock_request):
        detector = CopilotDetector()
        assert await detector.detect(mock_request) is False


class TestDetectClient:
    @pytest.mark.asyncio
    async def test_successful_client_detection(self, mock_request):
        detect_client = DetectClient()

        async def get_json():
            return {"messages": [{"content": "Cline"}]}

        mock_request.json = get_json

        @detect_client
        async def test_endpoint(request: Request):
            return request.state.detected_client

        result = await test_endpoint(mock_request)
        assert result == ClientType.CLINE

    @pytest.mark.asyncio
    async def test_fallback_to_generic(self, mock_request):
        detect_client = DetectClient()

        async def get_json():
            return {"messages": [{"content": "Unknown Client"}]}

        mock_request.json = get_json

        @detect_client
        async def test_endpoint(request: Request):
            return request.state.detected_client

        result = await test_endpoint(mock_request)
        assert result == ClientType.GENERIC

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_request):
        detect_client = DetectClient()

        async def get_json():
            raise Exception("Test error")

        mock_request.json = get_json

        @detect_client
        async def test_endpoint(request: Request):
            return request.state.detected_client

        result = await test_endpoint(mock_request)
        assert result == ClientType.GENERIC

    @pytest.mark.asyncio
    async def test_state_setting(self, mock_request):
        detect_client = DetectClient()

        async def get_json():
            return {"messages": [{"content": "Kodu"}]}

        mock_request.json = get_json

        @detect_client
        async def test_endpoint(request: Request):
            assert hasattr(request.state, "detected_client")
            return request.state.detected_client

        result = await test_endpoint(mock_request)
        assert result == ClientType.KODU

    @pytest.mark.asyncio
    async def test_copilot_detection_in_detect_client(self, mock_request):
        detect_client = DetectClient()
        mock_request.headers = Headers({"user-agent": "Copilot"})

        @detect_client
        async def test_endpoint(request: Request):
            return request.state.detected_client

        result = await test_endpoint(mock_request)
        assert result == ClientType.COPILOT
