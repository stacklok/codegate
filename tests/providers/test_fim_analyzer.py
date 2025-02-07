import pytest

from codegate.providers.fim_analyzer import FIMAnalyzer


@pytest.mark.parametrize(
    "url, expected_bool",
    [
        ("http://localhost:8989", False),
        ("http://test.com/chat/completions", False),
        ("http://localhost:8989/completions", True),
    ],
)
def test_is_fim_request_url(url, expected_bool):
    assert FIMAnalyzer._is_fim_request_url(url) == expected_bool


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
            "content": "http://localhost:8989/completions",
        }
    ]
}
TOOL_DATA = {
    "prompt": "cline",
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
    assert FIMAnalyzer._is_fim_request_body(data) == expected_bool


@pytest.mark.parametrize(
    "url, data, expected_bool",
    [
        ("http://localhost:8989", DATA_CONTENT_STR, True),  # True because of the data
        (
            "http://test.com/chat/completions",
            INVALID_DATA_CONTET,
            False,
        ),  # False because of the url
        ("http://localhost:8989/completions", DATA_CONTENT_STR, True),  # True because of the url
        ("http://localhost:8989/completions", TOOL_DATA, False),  # False because of the tool data
    ],
)
def test_is_fim_request(url, data, expected_bool):
    assert FIMAnalyzer.is_fim_request(url, data) == expected_bool
