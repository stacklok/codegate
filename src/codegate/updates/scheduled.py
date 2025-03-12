import threading
import time

import structlog

import codegate
from codegate.updates.client import Origin, UpdateClient

logger = structlog.get_logger("codegate")


class ScheduledUpdateChecker(threading.Thread):
    """
    ScheduledUpdateChecker calls the UpdateClient on a recurring interval.
    This is implemented as a separate thread to avoid blocking the main thread.
    A dedicated scheduling library could have been used, but the requirements
    are trivial, and a simple hand-rolled solution is sufficient.
    """

    def __init__(self, client: UpdateClient, interval_seconds: int = 14400):  # 4 hours in seconds
        super().__init__()
        self.__client = client
        self.__interval_seconds = interval_seconds

    def run(self):
        """
        Overrides the `run` method of threading.Thread.
        """
        while True:
            logger.info("Checking for CodeGate updates")
            latest = self.__client.get_latest_version(Origin.BackEnd)
            if latest != codegate.__version__:
                logger.warning(f"A new version of CodeGate is available: {latest}")
            time.sleep(self.__interval_seconds)
