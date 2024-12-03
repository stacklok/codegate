import asyncio
import json
import os
import shutil

import weaviate
from weaviate.classes.config import DataType, Property
from weaviate.embedded import EmbeddedOptions
from weaviate.util import generate_uuid5

from codegate.inference.inference_engine import LlamaCppInferenceEngine
from codegate.utils.utils import generate_vector_string


class PackageImporter:
    def __init__(self):
        self.client = weaviate.WeaviateClient(
            embedded_options=EmbeddedOptions(
                persistence_data_path="./weaviate_data",
                grpc_port=50052,
                additional_env_vars={
                    "ENABLE_MODULES": "backup-filesystem",
                    "BACKUP_FILESYSTEM_PATH": os.getenv("BACKUP_FILESYSTEM_PATH", "/tmp"),
                },
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

    def restore_backup(self):
        if os.getenv("BACKUP_FOLDER"):
            try:
                self.client.backup.restore(
                    backup_id=os.getenv("BACKUP_FOLDER"),
                    backend="filesystem",
                    wait_for_completion=True,
                )
            except Exception as e:
                print(f"Failed to restore backup: {e}")

    def take_backup(self):
        # if backup folder exists, remove it
        backup_path = os.path.join(
            os.getenv("BACKUP_FILESYSTEM_PATH", "/tmp"), os.getenv("BACKUP_TARGET_ID", "backup")
        )
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)

        #  take a backup of the data
        self.client.backup.create(
            backup_id=os.getenv("BACKUP_TARGET_ID", "backup"),
            backend="filesystem",
            wait_for_completion=True,
        )

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

    async def process_package(self, batch, package):
        vector_str = generate_vector_string(package)
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

                    vector_str = generate_vector_string(package)
                    vector = await self.inference_engine.embed(self.model_path, [vector_str])
                    packages_to_insert.append((package, vector[0]))

                # Synchronous batch insert after preparing all data
                with collection.batch.dynamic() as batch:
                    for package, vector in packages_to_insert:
                        batch.add_object(
                            properties=package, vector=vector, uuid=generate_uuid5(package)
                        )

    async def run_import(self):
        self.restore_backup()
        self.setup_schema()
        await self.add_data()
        self.take_backup()


if __name__ == "__main__":
    importer = PackageImporter()
    asyncio.run(importer.run_import())
    try:
        assert importer.client.is_live()
        pass
    finally:
        importer.client.close()
