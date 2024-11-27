import asyncio
import json

import weaviate
from weaviate.classes.config import DataType, Property
from weaviate.embedded import EmbeddedOptions
from weaviate.util import generate_uuid5

from codegate.inference.inference_engine import LlamaCppInferenceEngine


class PackageImporter:
    def __init__(self):
        self.client = weaviate.WeaviateClient(
            embedded_options=EmbeddedOptions(
                persistence_data_path="./weaviate_data", grpc_port=50052
            )
        )
        self.json_files = [
            "data/archived.jsonl",
            "data/deprecated.jsonl",
            "data/malicious.jsonl",
        ]
        self.client.connect()
        self.inference_engine = LlamaCppInferenceEngine()
        self.model_path = "./models/all-minilm-L6-v2-q5_k_m.gguf"

    def setup_schema(self):
        if not self.client.collections.exists("Package"):
            self.client.collections.create(
                "Package",
                properties=[
                    Property(name="name", data_type=DataType.TEXT),
                    Property(name="type", data_type=DataType.TEXT),
                    Property(name="status", data_type=DataType.TEXT),
                    Property(name="description", data_type=DataType.TEXT),
                ],
            )

    def generate_vector_string(self, package):
        vector_str = f"{package['name']}"
        package_url = ""
        type_map = {
            "pypi": "Python package available on PyPI",
            "npm": "JavaScript package available on NPM",
            "go": "Go package",
            "crates": "Rust package available on Crates",
            "java": "Java package",
        }
        status_messages = {
            "archived": "However, this package is found to be archived and no longer maintained.",
            "deprecated": "However, this package is found to be deprecated and no longer "
            "recommended for use.",
            "malicious": "However, this package is found to be malicious.",
        }
        vector_str += f" is a {type_map.get(package['type'], 'unknown type')} "
        package_url = f"https://trustypkg.dev/{package['type']}/{package['name']}"

        # Add extra status
        status_suffix = status_messages.get(package["status"], "")
        if status_suffix:
            vector_str += f"{status_suffix} For additional information refer to {package_url}"
        return vector_str

    async def process_package(self, batch, package):
        vector_str = self.generate_vector_string(package)
        vector = await self.inference_engine.embed(self.model_path, [vector_str])
        # This is where the synchronous call is made
        batch.add_object(properties=package, vector=vector[0])

    async def add_data(self):
        collection = self.client.collections.get("Package")
        existing_packages = list(collection.iterator())
        packages_dict = {
            f"{package.properties['name']}/{package.properties['type']}": {
                "status": package.properties["status"],
                "description": package.properties["description"],
            }
            for package in existing_packages
        }

        for json_file in self.json_files:
            with open(json_file, "r") as f:
                print("Adding data from", json_file)
                packages_to_insert = []
                for line in f:
                    package = json.loads(line)
                    package["status"] = json_file.split("/")[-1].split(".")[0]
                    key = f"{package['name']}/{package['type']}"

                    if key in packages_dict and packages_dict[key] == {
                        "status": package["status"],
                        "description": package["description"],
                    }:
                        print("Package already exists", key)
                        continue

                    vector_str = self.generate_vector_string(package)
                    vector = await self.inference_engine.embed(self.model_path, [vector_str])
                    packages_to_insert.append((package, vector[0]))

                # Synchronous batch insert after preparing all data
                with collection.batch.dynamic() as batch:
                    for package, vector in packages_to_insert:
                        batch.add_object(
                            properties=package, vector=vector, uuid=generate_uuid5(package)
                        )

    async def run_import(self):
        self.setup_schema()
        await self.add_data()


if __name__ == "__main__":
    importer = PackageImporter()
    asyncio.run(importer.run_import())
    try:
        assert importer.client.is_live()
        pass
    finally:
        importer.client.close()
