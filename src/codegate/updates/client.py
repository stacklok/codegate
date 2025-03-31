from enum import Enum

import requests
import structlog

logger = structlog.get_logger("codegate")


__update_client_singleton = None


# Enum representing whether the request is coming from the front-end or the back-end.
class Origin(Enum):
    FrontEnd = "FE"
    BackEnd = "BE"


class UpdateClient:
    def __init__(self, update_url: str, current_version: str, instance_id: str):
        self.__update_url = update_url
        self.__current_version = current_version
        self.__instance_id = instance_id

    def get_latest_version(self, origin: Origin) -> str:
        """
        Retrieves the latest version of CodeGate from updates.codegate.ai
        """

        user_agent = f"codegate/{self.__current_version} {origin.value}"
        if os.environ.get("CODEGATE_DEV_ENV"):
            user_agent += "-dev"
        headers = {
            "X-Instance-ID": self.__instance_id,
            "User-Agent": user_agent,
        }

        try:
            response = requests.get(self.__update_url, headers=headers, timeout=10)
            # Throw if the request was not successful.
            response.raise_for_status()
            return response.json()["version"]
        except Exception as e:
            logger.error(f"Error fetching latest version from f{self.__update_url}: {e}")
            return "unknown"


# Use a singleton since we do not have a good way of doing dependency injection
# with the API endpoints.
def init_update_client_singleton(
    update_url: str, current_version: str, instance_id: str
) -> UpdateClient:
    global __update_client_singleton
    __update_client_singleton = UpdateClient(update_url, current_version, instance_id)
    return __update_client_singleton


def get_update_client_singleton() -> UpdateClient:
    global __update_client_singleton
    if __update_client_singleton is None:
        raise ValueError("UpdateClient singleton not initialized")
    return __update_client_singleton
