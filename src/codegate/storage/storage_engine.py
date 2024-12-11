from typing import List

import structlog
import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import DataType
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.embedded import EmbeddedOptions

from codegate.config import Config
from codegate.inference.inference_engine import LlamaCppInferenceEngine

logger = structlog.get_logger("codegate")
VALID_ECOSYSTEMS = ["npm", "pypi", "crates", "maven", "go"]

schema_config = [
    {
        "name": "Package",
        "properties": [
            {"name": "name", "data_type": DataType.TEXT},
            {"name": "type", "data_type": DataType.TEXT},
            {"name": "status", "data_type": DataType.TEXT},
            {"name": "description", "data_type": DataType.TEXT},
        ],
    },
]


class StorageEngine:
    __storage_engine = None

    def __new__(cls, *args, **kwargs):
        if cls.__storage_engine is None:
            cls.__storage_engine = super().__new__(cls)
        return cls.__storage_engine

    # This function is needed only for the unit testing for the
    # mocks to work.
    @classmethod
    def recreate_instance(cls, *args, **kwargs):
        cls.__storage_engine = None
        return cls(*args, **kwargs)

    def __init__(self, data_path="./weaviate_data"):
        if hasattr(self, "initialized"):
            return

        self.initialized = True
        self.data_path = data_path
        self.inference_engine = LlamaCppInferenceEngine()
        self.model_path = (
            f"{Config.get_config().model_base_path}/{Config.get_config().embedding_model}"
        )
        self.schema_config = schema_config

        # setup schema for weaviate
        self.weaviate_client = self.get_client(self.data_path)
        if self.weaviate_client is not None:
            try:
                self.weaviate_client.connect()
                self.setup_schema(self.weaviate_client)
            except Exception as e:
                logger.error(f"Failed to connect or setup schema: {str(e)}")

    def __del__(self):
        try:
            self.weaviate_client.close()
        except Exception as e:
            logger.error(f"Failed to close client: {str(e)}")

    def get_client(self, data_path):
        try:
            # Configure Weaviate logging
            additional_env_vars = {
                # Basic logging configuration
                "LOG_FORMAT": Config.get_config().log_format.value.lower(),
                "LOG_LEVEL": Config.get_config().log_level.value.lower(),
                # Disable colored output
                "LOG_FORCE_COLOR": "false",
                # Configure JSON format
                "LOG_JSON_FIELDS": "timestamp, level,message",
                # Configure text format
                "LOG_METHOD": Config.get_config().log_format.value.lower(),
                "LOG_LEVEL_IN_UPPER": "false",  # Keep level lowercase like codegate format
                # Disable additional fields
                "LOG_GIT_HASH": "false",
                "LOG_VERSION": "false",
                "LOG_BUILD_INFO": "false",
            }

            client = weaviate.WeaviateClient(
                embedded_options=EmbeddedOptions(
                    persistence_data_path=data_path,
                    additional_env_vars=additional_env_vars,
                ),
            )
            return client
        except Exception as e:
            logger.error(f"Error during client creation: {str(e)}")
            return None

    def setup_schema(self, client):
        for class_config in self.schema_config:
            if not client.collections.exists(class_config["name"]):
                client.collections.create(
                    class_config["name"], properties=class_config["properties"]
                )
            logger.info(f"Weaviate schema for class {class_config['name']} setup complete.")

    async def search_by_property(self, name: str, properties: List[str]) -> list[object]:
        if len(properties) == 0:
            return []

        # Perform the vector search
        if self.weaviate_client is None:
            logger.error("Could not find client, not returning results.")
            return []

        if not self.weaviate_client:
            logger.error("Invalid client, cannot perform search.")
            return []

        try:

            packages = self.weaviate_client.collections.get("Package")
            response = packages.query.fetch_objects(
                filters=Filter.by_property(name).contains_any(properties),
            )

            if not response:
                return []

            # Weaviate performs substring matching of the properties. So
            # we need to double check the response.
            properties = [prop.lower() for prop in properties]
            filterd_objects = []
            for object in response.objects:
                if object.properties[name].lower() in properties:
                    filterd_objects.append(object)
            response.objects = filterd_objects

            return response.objects
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            return []

    async def search(
        self, query: str, limit=5, distance=0.3, ecosystem=None, packages=None
    ) -> list[object]:
        """
        Search the 'Package' collection based on a query string.

        Args:
            query (str): The text query for which to search.
            limit (int): The number of results to return.
            distance (float): The distance threshold for the search.
            ecosystem (str): The ecosystem to search in.
            packages (list): The list of packages to filter the search.

        Returns:
            list: A list of matching results with their properties and distances.
        """
        # Generate the vector for the query
        query_vector = await self.inference_engine.embed(self.model_path, [query])

        # Perform the vector search
        try:
            collection = self.weaviate_client.collections.get("Package")
            if packages:
                # filter by packages and ecosystem if present
                filters = []
                if ecosystem and ecosystem in VALID_ECOSYSTEMS:
                    filters.append(wvc.query.Filter.by_property("type").equal(ecosystem))
                filters.append(wvc.query.Filter.by_property("name").contains_any(packages))
                response = collection.query.near_vector(
                    query_vector[0],
                    limit=limit,
                    distance=distance,
                    filters=wvc.query.Filter.all_of(filters),
                    return_metadata=MetadataQuery(distance=True),
                )
            else:
                response = collection.query.near_vector(
                    query_vector[0],
                    limit=limit,
                    distance=distance,
                    return_metadata=MetadataQuery(distance=True),
                )

            if not response:
                return []
            return response.objects

        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return []
