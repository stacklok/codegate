import pytest

from codegate.db.models import ProviderType
from codegate.muxing.adapter import BodyAdapter, ChatStreamChunkFormatter


class MockedEndpoint:
    def __init__(self, provider_type: ProviderType, endpoint_route: str):
        self.provider_type = provider_type
        self.endpoint = endpoint_route


class MockedModelRoute:
    def __init__(self, provider_type: ProviderType, endpoint_route: str):
        self.endpoint = MockedEndpoint(provider_type, endpoint_route)


@pytest.mark.parametrize(
    "provider_type, endpoint_route, expected_route",
    [
        (ProviderType.openai, "https://api.openai.com/", "https://api.openai.com/v1"),
        (ProviderType.openrouter, "https://openrouter.ai/api", "https://openrouter.ai/api/v1"),
        (ProviderType.openrouter, "https://openrouter.ai/", "https://openrouter.ai/api/v1"),
        (ProviderType.ollama, "http://localhost:11434", "http://localhost:11434"),
        (ProviderType.vllm, "http://localhost:8000", "http://localhost:8000/v1"),
    ],
)
def test_catch_all(provider_type, endpoint_route, expected_route):
    body_adapter = BodyAdapter()
    model_route = MockedModelRoute(provider_type, endpoint_route)
    actual_route = body_adapter._get_provider_formatted_url(model_route)
    assert actual_route == expected_route


@pytest.mark.parametrize(
    "chunk, expected_cleaned_chunk",
    [
        (
            (
                'event: content_block_delta\ndata:{"type": "content_block_delta", "index": 0, '
                '"delta": {"type": "text_delta", "text": "\n  metadata:\n    name: trusty"}}'
            ),
            (
                '{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", '
                '"text": "\n  metadata:\n    name: trusty"}}'
            ),
        ),
        (
            (
                "event: content_block_delta\n"
                'data:{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", '
                '"text": "v1\nkind: NetworkPolicy\nmetadata:"}}'
            ),
            (
                '{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text"'
                ': "v1\nkind: NetworkPolicy\nmetadata:"}}'
            ),
        ),
    ],
)
def test_clean_chunk(chunk, expected_cleaned_chunk):
    formatter = ChatStreamChunkFormatter()
    gotten_chunk = formatter._clean_chunk(chunk)
    assert gotten_chunk == expected_cleaned_chunk
