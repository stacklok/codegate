from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Union,
)

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from codegate.providers.base import BaseCompletionHandler, BaseProvider
from codegate.providers.normalizer import ModelInputNormalizer, ModelOutputNormalizer
from codegate.providers.registry import ProviderRegistry
from codegate.types.common import ChatCompletionRequest, ModelResponse


class MockCompletionHandler(BaseCompletionHandler):
    async def complete(self, data: Dict, api_key: str) -> AsyncIterator[Any]:
        yield "test"

    def translate_request(self, data: Dict, api_key: str) -> Any:
        return data

    def translate_response(self, response: Any) -> Any:
        return response

    def translate_streaming_response(
        self,
        response: AsyncIterator[Any],
    ) -> AsyncIterator[Any]:
        return response

    def execute_completion(
        self,
        request: ChatCompletionRequest,
        api_key: Optional[str],
        stream: bool = False,
    ) -> Any:
        pass

    def _create_streaming_response(
        self,
        stream: AsyncIterator[Any],
    ) -> StreamingResponse:
        return StreamingResponse(stream)

    def _create_json_response(self, response: Any) -> Any:
        raise NotImplementedError


class MockInputNormalizer(ModelInputNormalizer):
    def normalize(self, data: Dict) -> Dict:
        return data

    def denormalize(self, data: Dict) -> Dict:
        return data


class MockOutputNormalizer(ModelOutputNormalizer):
    def normalize_streaming(
        self,
        model_reply: Union[AsyncIterable[Any], Iterable[Any]],
    ) -> Union[AsyncIterator[ModelResponse], Iterator[ModelResponse]]:
        pass

    def normalize(self, model_reply: Any) -> ModelResponse:
        pass

    def denormalize(self, normalized_reply: ModelResponse) -> Any:
        pass

    def denormalize_streaming(
        self,
        normalized_reply: Union[AsyncIterable[ModelResponse], Iterable[ModelResponse]],
    ) -> Union[AsyncIterator[Any], Iterator[Any]]:
        pass


class MockProvider(BaseProvider):
    def __init__(
        self,
    ):
        super().__init__(
            MockInputNormalizer(), MockOutputNormalizer(), MockCompletionHandler(), None
        )

    @property
    def provider_route_name(self) -> str:
        return "mock_provider"

    async def process_request(self, data: dict, api_key: str, base_url: str, request_url_path: str):
        return {"message": "test"}

    def models(self):
        return []

    def _setup_routes(self) -> None:
        @self.router.get(f"/{self.provider_route_name}/test")
        async def test_route():
            return await self.process_request({})


@pytest.fixture
def mock_completion_handler():
    return MockCompletionHandler()


@pytest.fixture
def app():
    return FastAPI()


@pytest.fixture
def registry(app):
    return ProviderRegistry(app)


def test_add_provider(registry):
    provider = MockProvider()
    registry.add_provider("test", provider)

    assert "test" in registry.providers
    assert registry.providers["test"] == provider


def test_get_provider(registry):
    provider = MockProvider()
    registry.add_provider("test", provider)

    assert registry.get_provider("test") == provider
    assert registry.get_provider("nonexistent") is None


def test_provider_routes_added(app, registry):
    provider = MockProvider()
    registry.add_provider("test", provider)

    routes = [route for route in app.routes if route.path == "/mock_provider/test"]
    assert len(routes) == 1
