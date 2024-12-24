from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from codegate.config import Config
from codegate.storage.storage_engine import (
    StorageEngine,
)  # Adjust the import based on your actual path


@pytest.fixture
def mock_weaviate_client():
    client = MagicMock()
    response = MagicMock()
    obj = MagicMock()
    obj.properties = {
                "name": "test",
                "type": "library",
                "status": "active",
                "description": "test description",
    }

    response.objects = [obj]
    client.collections.get.return_value.query.near_vector.return_value = response
    return client


@pytest.fixture
def mock_inference_engine():
    engine = AsyncMock()
    engine.embed.return_value = [0.1, 0.2, 0.3]  # Adjust based on expected vector dimensions
    return engine


@pytest.mark.asyncio
async def test_search(mock_weaviate_client, mock_inference_engine):
    Config.load(config_path="./config.yaml")

    # Patch the LlamaCppInferenceEngine.embed method (not the entire class)
    with patch(
        "codegate.inference.inference_engine.LlamaCppInferenceEngine.embed",
        mock_inference_engine.embed,
    ):
        # Initialize StorageEngine
        with patch(
            "codegate.storage.storage_engine.StorageEngine.get_client",
            return_value=mock_weaviate_client,
        ):
            # Initialize StorageEngine
            # Need to recreate instance to use the mock
            storage_engine = StorageEngine.recreate_instance(data_path="./weaviate_data")

            # Invoke the search method
            results = await storage_engine.search(query="test query", limit=5, distance=0.3)

            # Assertions to validate the expected behavior
            assert len(results) == 1  # Assert that one result is returned
            assert results[0].properties["name"] == "test"
            mock_weaviate_client.connect.assert_called()
