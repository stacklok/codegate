from typing import List, Optional
from urllib.parse import urlparse

import structlog
from pydantic import ValidationError

from codegate.api import v1_models as apimodelsv1
from codegate.config import Config
from codegate.db.connection import DbReader, DbRecorder

logger = structlog.get_logger("codegate")


class ProviderCrud:
    """The CRUD operations for the provider endpoint references within
    Codegate.

    This is meant to handle all the transformations in between the
    database and the API, as well as other sources of information. All
    operations should result in the API models being returned.
    """

    def __init__(self):
        self._db_reader = DbReader()
        self._db_writer = DbRecorder()
        config = Config.get_config()
        if config is None:
            logger.warning("OZZ: No configuration found.")
            provided_urls = {}
        else:
            logger.info("OZZ: Using configuration for provider URLs.")
            provided_urls = config.provider_urls

        self._provider_urls = provided_urls

    def list_endpoints(self) -> List[apimodelsv1.ProviderEndpoint]:
        """List all the endpoints."""

        endpoints = []
        for provider_name, provider_url in self._provider_urls.items():
            provend = self.__provider_endpoint_from_cfg(provider_name, provider_url)
            if provend is not None:
                endpoints.append(provend)

        return endpoints

    def __provider_endpoint_from_cfg(
        self, provider_name: str, provider_url: str
    ) -> Optional[apimodelsv1.ProviderEndpoint]:
        """Create a provider endpoint from the config entry."""

        try:
            _ = urlparse(provider_url)
        except Exception:
            logger.warning(
                "Invalid provider URL", provider_name=provider_name, provider_url=provider_url
            )
            return None

        try:
            return apimodelsv1.ProviderEndpoint(
                name=provider_name,
                endpoint=provider_url,
                descrption=("Endpoint for the {} provided via the CodeGate configuration.").format(
                    provider_name
                ),
                provider_type=provider_name,
                auth_type=apimodelsv1.ProviderAuthType.none,
                source=apimodelsv1.ProviderEndpointSource.config,
            )
        except ValidationError as err:
            logger.warning(
                "Invalid provider name",
                provider_name=provider_name,
                provider_url=provider_url,
                err=str(err),
            )
            return None
