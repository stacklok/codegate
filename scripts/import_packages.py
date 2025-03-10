import argparse
import asyncio
import json
import os
import sqlite3

import numpy as np
import sqlite_vec_sl_tmp

from codegate.config import Config
from codegate.inference.inference_engine import LlamaCppInferenceEngine
from codegate.utils.utils import generate_vector_string


class PackageImporter:
    def __init__(self, jsonl_dir="data", vec_db_path="./sqlite_data/vectordb.db"):
        os.makedirs(os.path.dirname(vec_db_path), exist_ok=True)
        self.vec_db_path = vec_db_path
        self.json_files = [
            os.path.join(jsonl_dir, "archived.jsonl"),
            os.path.join(jsonl_dir, "deprecated.jsonl"),
            os.path.join(jsonl_dir, "malicious.jsonl"),
            os.path.join(jsonl_dir, "vulnerable.jsonl"),
        ]
        self.conn = self._get_connection()
        Config.load()  # Load the configuration
        self.inference_engine = LlamaCppInferenceEngine()
        self.model_path = "./codegate_volume/models/all-minilm-L6-v2-q5_k_m.gguf"

    def _get_connection(self):
        conn = sqlite3.connect(self.vec_db_path)
        conn.enable_load_extension(True)
        sqlite_vec_sl_tmp.load(conn)
        conn.enable_load_extension(False)
        return conn

    def setup_schema(self):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS packages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                status TEXT NOT NULL,
                description TEXT,
                embedding BLOB
            )
        """
        )

        # table for packages that has at least one vulnerability high or critical
        cursor.execute(
            """
            CREATE TABLE cve_packages (
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                type TEXT NOT NULL
            )
        """
        )

        # Create indexes for faster querying
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_name ON packages(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_type ON packages(type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON packages(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pkg_cve_name ON cve_packages(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pkg_cve_type ON cve_packages(type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pkg_cve_version ON cve_packages(version)")

        self.conn.commit()

    async def process_cve_packages(self, package):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO cve_packages (name, version, type) VALUES (?, ?, ?)
        """,
            (
                package["name"],
                package["version"],
                package["type"],
            ),
        )
        self.conn.commit()

    async def process_package(self, package):
        vector_str = generate_vector_string(package)
        vector = await self.inference_engine.embed(
            self.model_path, [vector_str], n_gpu_layers=Config.get_config().chat_model_n_gpu_layers
        )
        vector_array = np.array(vector[0], dtype=np.float32)

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO packages (name, type, status, description, embedding)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                package["name"],
                package["type"],
                package["status"],
                package["description"],
                vector_array,  # sqlite-vec will handle numpy arrays directly
            ),
        )
        self.conn.commit()

    async def add_data(self):
        cursor = self.conn.cursor()

        # Get existing packages
        cursor.execute(
            """
            SELECT name, type, status, description
            FROM packages
        """
        )
        existing_packages = {
            f"{row[0]}/{row[1]}": {"status": row[2], "description": row[3]}
            for row in cursor.fetchall()
        }

        for json_file in self.json_files:
            print("Adding data from", json_file)
            with open(json_file, "r") as f:
                for line in f:
                    package = json.loads(line)
                    package["status"] = json_file.split("/")[-1].split(".")[0]
                    key = f"{package['name']}/{package['type']}"

                    if package["status"] == "vulnerable":
                        # Process vulnerable packages using the cve flow
                        await self.process_cve_packages(package)
                    else:
                        # For non-vulnerable packages, check for duplicates and process normally
                        if key in existing_packages and existing_packages[key] == {
                            "status": package["status"],
                            "description": package["description"],
                        }:
                            print("Package already exists", key)
                            continue

                        await self.process_package(package)

    async def run_import(self):
        self.setup_schema()
        await self.add_data()

    def __del__(self):
        try:
            if hasattr(self, "conn"):
                self.conn.close()
        except Exception as e:
            print(f"Failed to close connection: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Import packages into SQLite database with vector search capabilities."
    )
    parser.add_argument(
        "--jsonl-dir",
        type=str,
        default="data",
        help="Directory containing JSONL files. Default is 'data'.",
    )
    parser.add_argument(
        "--vec-db-path",
        type=str,
        default="./sqlite_data/vectordb.db",
        help="Path to SQLite database file. Default is './sqlite_data/vectordb.db'.",
    )
    args = parser.parse_args()

    importer = PackageImporter(jsonl_dir=args.jsonl_dir, vec_db_path=args.vec_db_path)
    asyncio.run(importer.run_import())
