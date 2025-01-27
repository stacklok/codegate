from unittest.mock import MagicMock

import pytest

from codegate.providers.base import BaseProvider


class MockProvider(BaseProvider):

    def __init__(self):
        mocked_input_normalizer = MagicMock()
        mocked_output_normalizer = MagicMock()
        mocked_completion_handler = MagicMock()
        mocked_factory = MagicMock()
        super().__init__(
            mocked_input_normalizer,
            mocked_output_normalizer,
            mocked_completion_handler,
            mocked_factory,
        )

    def models(self):
        return []

    def _setup_routes(self) -> None:
        pass

    @property
    def provider_route_name(self) -> str:
        return "mock-provider"


@pytest.mark.parametrize(
    "url, expected_bool",
    [
        ("http://example.com", False),
        ("http://test.com/chat/completions", False),
        ("http://example.com/completions", True),
    ],
)
def test_is_fim_request_url(url, expected_bool):
    mock_provider = MockProvider()
    request = MagicMock()
    request.url.path = url
    assert mock_provider._is_fim_request_url(request) == expected_bool


DATA_CONTENT_STR = {
    "messages": [
        {
            "role": "user",
            "content": "</COMPLETION> <COMPLETION> </QUERY> <QUERY>",
        }
    ]
}
DATA_CONTENT_LIST = {
    "messages": [
        {
            "role": "user",
            "content": [{"type": "text", "text": "</COMPLETION> <COMPLETION> </QUERY> <QUERY>"}],
        }
    ]
}
INVALID_DATA_CONTET = {
    "messages": [
        {
            "role": "user",
            "content": "http://example.com/completions",
        }
    ]
}


@pytest.mark.parametrize(
    "data, expected_bool",
    [
        (DATA_CONTENT_STR, True),
        (DATA_CONTENT_LIST, True),
        (INVALID_DATA_CONTET, False),
    ],
)
def test_is_fim_request_body(data, expected_bool):
    mock_provider = MockProvider()
    assert mock_provider._is_fim_request_body(data) == expected_bool
