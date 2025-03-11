from enum import Enum

import cachetools.func
import requests
import structlog

from codegate.db.connection import DbReader

logger = structlog.get_logger("codegate")


# Enum representing whether the request is coming from the front-end or the back-end.
class Origin(Enum):
    FrontEnd = "FE"
    BackEnd = "BE"


class UpdateClient:
    def __init__(self, update_url: str, current_version: str, db_reader: DbReader):
        self.__update_url = update_url
        self.__current_version = current_version
        self.__db_reader = db_reader
        self.__instance_id = None

    async def get_latest_version(self, origin: Origin) -> str:
        """
        Retrieves the latest version of CodeGate from updates.codegate.ai
        """
        logger.info(f"Fetching latest version from {self.__update_url}")
        instance_id = await self.__get_instance_id()
        return self.__fetch_latest_version(instance_id, origin)

    @cachetools.func.ttl_cache(maxsize=128, ttl=20 * 60)
    def __fetch_latest_version(self, instance_id: str, origin: Origin) -> str:
        headers = {
            "X-Instance-ID": instance_id,
            "User-Agent": f"codegate/{self.__current_version} {origin.value}",
        }

        try:
            response = requests.get(self.__update_url, headers=headers)
            # Throw if the request was not successful.
            response.raise_for_status()
            return response.json()["version"]
        except Exception as e:
            logger.error(f"Error fetching latest version from f{self.__update_url}: {e}")
            return "unknown"

    # Lazy load the instance ID from the DB.
    async def __get_instance_id(self):
        if self.__instance_id is None:
            instance_data = await self.__db_reader.get_instance()
            self.__instance_id = instance_data[0].id
        return self.__instance_id
