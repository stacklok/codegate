from typing import List
import os
import sqlite3
import structlog
import numpy as np
import sqlite_vec

from codegate.config import Config
from codegate.inference.inference_engine import LlamaCppInferenceEngine

logger = structlog.get_logger("codegate")
VALID_ECOSYSTEMS = ["npm", "pypi", "crates", "maven", "go"]

class StorageEngine:
    __storage_engine = None

    def __new__(cls, *args, **kwargs):
        if cls.__storage_engine is None:
            cls.__storage_engine = super().__new__(cls)
        return cls.__storage_engine

    @classmethod
    def recreate_instance(cls, *args, **kwargs):
        cls.__storage_engine = None
        return cls(*args, **kwargs)

    def __init__(self, data_path="./sqlite_data"):
        if hasattr(self, "initialized"):
            return

        self.initialized = True
        self.data_path = data_path
        os.makedirs(data_path, exist_ok=True)
        self.db_path = os.path.join(data_path, "packages.db")
        self.inference_engine = LlamaCppInferenceEngine()
        self.model_path = (
            f"{Config.get_config().model_base_path}/{Config.get_config().embedding_model}"
        )
        
        self.conn = self._get_connection()
        self._setup_schema()

    def __del__(self):
        try:
            if hasattr(self, 'conn'):
                self.conn.close()
        except Exception as e:
            logger.error(f"Failed to close connection: {str(e)}")

    def _get_connection(self):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            return conn
        except Exception as e:
            logger.error("Failed to initialize database connection", error=str(e))
            raise

    def _setup_schema(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS packages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                status TEXT NOT NULL,
                description TEXT,
                embedding BLOB
            )
        """)
        
        # Create indexes for faster querying
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_name ON packages(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_type ON packages(type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON packages(status)")
        
        self.conn.commit()

    async def search_by_property(self, name: str, properties: List[str]) -> list[object]:
        if len(properties) == 0:
            return []

        try:
            cursor = self.conn.cursor()
            placeholders = ','.join('?' * len(properties))
            query = f"""
                SELECT name, type, status, description
                FROM packages
                WHERE LOWER({name}) IN ({placeholders})
            """
            
            cursor.execute(query, [prop.lower() for prop in properties])
            results = []
            for row in cursor.fetchall():
                results.append({
                    "properties": {
                        "name": row[0],
                        "type": row[1],
                        "status": row[2],
                        "description": row[3]
                    }
                })
            return results
        except Exception as e:
            logger.error(f"An error occurred during property search: {str(e)}")
            return []

    async def search(
        self,
        query: str = None,
        ecosystem: str = None,
        packages: List[str] = None,
        limit: int = 5,
        distance: float = 0.3,
    ) -> list[object]:
        """
        Search packages based on vector similarity or direct property matches.
        """
        try:
            cursor = self.conn.cursor()
            
            if packages and ecosystem and ecosystem in VALID_ECOSYSTEMS:
                placeholders = ','.join('?' * len(packages))
                query_sql = f"""
                    SELECT name, type, status, description
                    FROM packages
                    WHERE LOWER(name) IN ({placeholders})
                    AND LOWER(type) = ?
                """
                params = [p.lower() for p in packages] + [ecosystem.lower()]
                logger.debug(
                    "Searching by package names and ecosystem",
                    packages=packages,
                    ecosystem=ecosystem,
                    sql=query_sql,
                    params=params
                )
                cursor.execute(query_sql, params)
                
            elif packages and not ecosystem:
                placeholders = ','.join('?' * len(packages))
                query_sql = f"""
                    SELECT name, type, status, description
                    FROM packages
                    WHERE LOWER(name) IN ({placeholders})
                """
                params = [p.lower() for p in packages]
                logger.debug(
                    "Searching by package names only",
                    packages=packages,
                    sql=query_sql,
                    params=params
                )
                cursor.execute(query_sql, params)
                
            elif query:
                # Generate embedding for the query
                query_vector = await self.inference_engine.embed(self.model_path, [query])
                query_embedding = np.array(query_vector[0], dtype=np.float32)
                query_embedding_bytes = query_embedding.tobytes()
                
                query_sql = """
                    WITH distances AS (
                        SELECT name, type, status, description,
                               vss_distance(embedding, ?) as distance
                        FROM packages
                    )
                    SELECT name, type, status, description, distance
                    FROM distances
                    WHERE distance <= ?
                    ORDER BY distance ASC
                    LIMIT ?
                """
                logger.debug(
                    "Performing vector similarity search",
                    query=query,
                    distance_threshold=distance,
                    limit=limit
                )
                cursor.execute(query_sql, (
                    query_embedding_bytes,
                    distance,
                    limit
                ))
            else:
                return []

            # Log the raw SQL results
            rows = cursor.fetchall()
            logger.debug(
                "Raw SQL results",
                row_count=len(rows),
                rows=[{
                    "name": row[0],
                    "type": row[1],
                    "status": row[2],
                    "description": row[3]
                } for row in rows]
            )

            results = []
            for row in rows:
                result = {
                    "properties": {
                        "name": row[0],
                        "type": row[1],
                        "status": row[2],
                        "description": row[3]
                    }
                }
                if query:  # Add distance for vector searches
                    result["metadata"] = {"distance": row[4]}
                results.append(result)
                
            return results

        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return []
