import os
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from codegate.config import Config
from codegate.storage.storage_engine import StorageEngine


@pytest.fixture(scope="module")
def mock_sqlite_vec():
    with patch("sqlite_vec.load") as mock_load:
        # Mock the vector similarity extension loading
        def setup_vector_similarity(conn):
            cursor = conn.cursor()
            # Create a table to store our mock distance function
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS vector_distances (
                id INTEGER PRIMARY KEY,
                distance FLOAT
            )"""
            )
            # Insert a mock distance value that will be used in searches
            cursor.execute("INSERT INTO vector_distances (distance) VALUES (0.1)")

            # Create a view that simulates the vec_distance_cosine function
            cursor.execute(
                """
            CREATE VIEW IF NOT EXISTS vec_distance_cosine_view AS
            SELECT distance FROM vector_distances WHERE id = 1
            """
            )

            # Create a function that returns the mock distance
            conn.create_function("vec_distance_cosine", 2, lambda x, y: 0.1)

        mock_load.side_effect = setup_vector_similarity
        yield mock_load


@pytest.fixture(scope="module")
def test_db_path():
    return "./test_sqlite_data/vectordb.db"


@pytest.fixture(scope="module")
def mock_config(test_db_path):
    # Create a mock config instance
    config = Config()
    config.model_base_path = "./codegate_volume/models"
    config.embedding_model = "all-minilm-L6-v2-q5_k_m.gguf"
    config.vec_db_path = test_db_path

    # Mock Config.get_config to return our test config
    with patch("codegate.config.Config.get_config", return_value=config):
        yield config


@pytest.fixture(scope="module")
def storage_engine(mock_sqlite_vec, mock_config, test_db_path):
    # Setup: Create a temporary database for testing
    test_db_dir = os.path.dirname(test_db_path)
    os.makedirs(test_db_dir, exist_ok=True)

    engine = StorageEngine(data_path=test_db_dir)
    yield engine
    # Teardown: Remove the temporary database and directory
    del engine
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    os.rmdir(test_db_dir)


@pytest.fixture(autouse=True)
def clean_database(storage_engine):
    # Clear all data before each test
    cursor = storage_engine.conn.cursor()
    cursor.execute("DELETE FROM packages")
    storage_engine.conn.commit()
    yield


@pytest.mark.asyncio
async def test_search_by_property(storage_engine):
    # Insert test data
    cursor = storage_engine.conn.cursor()
    cursor.execute(
        """
        INSERT INTO packages (name, type, status, description)
        VALUES ('invokehttp', 'pypi', 'active', 'An evil package')
    """
    )
    storage_engine.conn.commit()

    # Test search by property
    results = await storage_engine.search_by_property("name", ["invokehttp"])
    assert len(results) == 1
    assert results[0]["properties"]["name"] == "invokehttp"
    assert results[0]["properties"]["type"] == "pypi"
    assert results[0]["properties"]["status"] == "active"
    assert results[0]["properties"]["description"] == "An evil package"


@pytest.mark.asyncio
async def test_search_by_package_names(storage_engine):
    # Insert test data
    cursor = storage_engine.conn.cursor()
    cursor.execute(
        """
        INSERT INTO packages (name, type, status, description)
        VALUES ('invokehttp', 'pypi', 'active', 'An evil package')
    """
    )
    storage_engine.conn.commit()

    # Test search by package names
    results = await storage_engine.search(packages=["invokehttp"])
    assert len(results) == 1
    assert results[0]["properties"]["name"] == "invokehttp"
    assert results[0]["properties"]["type"] == "pypi"
    assert results[0]["properties"]["status"] == "active"
    assert results[0]["properties"]["description"] == "An evil package"


@pytest.mark.asyncio
async def test_search_by_query(storage_engine):
    # Mock the inference engine to return a fixed embedding
    with patch.object(
        storage_engine.inference_engine, "embed", new=AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    ):
        # Insert test data with embedding
        cursor = storage_engine.conn.cursor()
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes()
        cursor.execute(
            """
            INSERT INTO packages (name, type, status, description, embedding)
            VALUES ('invokehttp', 'pypi', 'active', 'An evil package', ?)
        """,
            (embedding,),
        )
        storage_engine.conn.commit()

        # Test search by query
        results = await storage_engine.search(query="test invokehttp")
        assert len(results) == 1
        assert results[0]["properties"]["name"] == "invokehttp"
        assert results[0]["properties"]["type"] == "pypi"
        assert results[0]["properties"]["status"] == "active"
        assert results[0]["properties"]["description"] == "An evil package"
        assert "metadata" in results[0]
        assert "distance" in results[0]["metadata"]
