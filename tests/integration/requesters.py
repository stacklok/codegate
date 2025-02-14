import json
import os
from abc import ABC, abstractmethod
from typing import Optional

import requests
import structlog

logger = structlog.get_logger("codegate")


class BaseRequester(ABC):
    @abstractmethod
    def make_request(
        self, url: str, headers: dict, data: dict, method: str = "POST"
    ) -> Optional[requests.Response]:
        pass


class StandardRequester(BaseRequester):
    def make_request(
        self, url: str, headers: dict, data: dict, method: str = "POST"
    ) -> Optional[requests.Response]:
        # Ensure Content-Type is always set correctly
        headers["Content-Type"] = "application/json"

        # Explicitly serialize to JSON string
        json_data = json.dumps(data)

        return requests.request(
            method=method,
            url=url,
            headers=headers,
            data=json_data,  # Use data instead of json parameter
        )


class CopilotRequester(BaseRequester):
    def make_request(
        self, url: str, headers: dict, data: dict, method: str = "POST"
    ) -> Optional[requests.Response]:
        # Ensure Content-Type is always set correctly
        headers["Content-Type"] = "application/json"

        # Explicitly serialize to JSON string
        json_data = json.dumps(data)

        return requests.request(
            method=method,
            url=url,
            data=json_data,  # Use data instead of json parameter
            headers=headers,
            proxies={"https": "https://localhost:8990", "http": "http://localhost:8990"},
            verify=os.environ.get("CA_CERT_FILE"),
            stream=True,
        )


class RequesterFactory:
    @staticmethod
    def create_requester(provider: str) -> BaseRequester:
        if provider.lower() == "copilot":
            return CopilotRequester()
        return StandardRequester()
