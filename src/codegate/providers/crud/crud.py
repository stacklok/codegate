import asyncio
from typing import List, Optional
from urllib.parse import urlparse
from uuid import UUID, uuid4

import structlog
from pydantic import ValidationError

from codegate.api import v1_models as apimodelsv1
from codegate.config import Config
from codegate.db import models as dbmodels
from codegate.db.connection import DbReader, DbRecorder
from codegate.providers.base import BaseProvider
from codegate.providers.registry import ProviderRegistry

logger = structlog.get_logger("codegate")


class ProviderNotFoundError(Exception):
    pass


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

    async def list_endpoints(self) -> List[apimodelsv1.ProviderEndpoint]:
        """List all the endpoints."""

        outendpoints = []
        dbendpoints = await self._db_reader.get_provider_endpoints()
        for dbendpoint in dbendpoints:
            outendpoints.append(apimodelsv1.ProviderEndpoint.from_db_model(dbendpoint))

        return outendpoints

    async def get_endpoint_by_id(self, id: UUID) -> Optional[apimodelsv1.ProviderEndpoint]:
        """Get an endpoint by ID."""

        dbendpoint = await self._db_reader.get_provider_endpoint_by_id(str(id))
        if dbendpoint is None:
            return None

        return apimodelsv1.ProviderEndpoint.from_db_model(dbendpoint)

    async def get_endpoint_by_name(self, name: str) -> Optional[apimodelsv1.ProviderEndpoint]:
        """Get an endpoint by name."""

        dbendpoint = await self._db_reader.get_provider_endpoint_by_name(name)
        if dbendpoint is None:
            return None

        return apimodelsv1.ProviderEndpoint.from_db_model(dbendpoint)

    async def add_endpoint(
        self, endpoint: apimodelsv1.ProviderEndpoint
    ) -> apimodelsv1.ProviderEndpoint:
        """Add an endpoint."""
        dbend = endpoint.to_db_model()

        # We override the ID here, as we want to generate it.
        dbend.id = str(uuid4())

        dbendpoint = await self._db_writer.add_provider_endpoint()
        return apimodelsv1.ProviderEndpoint.from_db_model(dbendpoint)

    async def update_endpoint(
        self, endpoint: apimodelsv1.ProviderEndpoint
    ) -> apimodelsv1.ProviderEndpoint:
        """Update an endpoint."""

        dbendpoint = await self._db_writer.update_provider_endpoint(endpoint.to_db_model())
        return apimodelsv1.ProviderEndpoint.from_db_model(dbendpoint)

    async def delete_endpoint(self, provider_id: UUID):
        """Delete an endpoint."""

        dbendpoint = await self._db_reader.get_provider_endpoint_by_id(str(provider_id))
        if dbendpoint is None:
            raise ProviderNotFoundError("Provider not found")

        await self._db_writer.delete_provider_endpoint(dbendpoint)

    async def models_by_provider(self, provider_id: UUID) -> List[apimodelsv1.ModelByProvider]:
        """Get the models by provider."""

        # First we try to get the provider
        dbendpoint = await self._db_reader.get_provider_endpoint_by_id(str(provider_id))
        if dbendpoint is None:
            raise ProviderNotFoundError("Provider not found")

        outmodels = []
        dbmodels = await self._db_reader.get_provider_models_by_provider_id(str(provider_id))
        for dbmodel in dbmodels:
            outmodels.append(
                apimodelsv1.ModelByProvider(
                    name=dbmodel.name,
                    provider_id=dbmodel.provider_endpoint_id,
                    provider_name=dbendpoint.name,
                )
            )

        return outmodels

    async def get_all_models(self) -> List[apimodelsv1.ModelByProvider]:
        """Get all the models."""

        outmodels = []
        dbmodels = await self._db_reader.get_all_provider_models()
        for dbmodel in dbmodels:
            ename = dbmodel.provider_endpoint_name if dbmodel.provider_endpoint_name else ""
            outmodels.append(
                apimodelsv1.ModelByProvider(
                    name=dbmodel.name,
                    provider_id=dbmodel.provider_endpoint_id,
                    provider_name=ename,
                )
            )

        return outmodels


async def initialize_provider_endpoints(preg: ProviderRegistry):
    db_writer = DbRecorder()
    db_reader = DbReader()
    config = Config.get_config()
    if config is None:
        provided_urls = {}
    else:
        provided_urls = config.provider_urls

    for provider_name, provider_url in provided_urls.items():
        provend = __provider_endpoint_from_cfg(provider_name, provider_url)
        if provend is None:
            continue

        # Check if the provider is already in the db
        dbprovend = await db_reader.get_provider_endpoint_by_name(provend.name)
        if dbprovend is not None:
            logger.debug(
                "Provider already in DB. Not re-adding.",
                provider=provend.name,
                endpoint=provend.endpoint,
            )
            continue

        pimpl = provend.get_from_registry(preg)
        await try_initialize_provider_endpoints(provend, pimpl, db_writer)


async def try_initialize_provider_endpoints(
    provend: apimodelsv1.ProviderEndpoint,
    pimpl: BaseProvider,
    db_writer: DbRecorder,
):
    try:
        models = pimpl.models()
    except Exception as err:
        logger.debug(
            "Unable to get models from provider",
            provider=provend.name,
            err=str(err),
        )
        return

    logger.info(
        "initializing provider to DB",
        provider=provend.name,
        endpoint=provend.endpoint,
        models=models,
    )
    # We only try to add the provider if we have models
    await db_writer.add_provider_endpoint(provend.to_db_model())

    tasks = set()
    for model in models:
        tasks.add(
            db_writer.add_provider_model(
                dbmodels.ProviderModel(
                    provider_endpoint_id=provend.id,
                    name=model,
                )
            )
        )

    await asyncio.gather(*tasks)


def __provider_endpoint_from_cfg(
    provider_name: str, provider_url: str
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
            id=str(uuid4()),
            name=provider_name,
            endpoint=provider_url,
            description=("Endpoint for the {} provided via the CodeGate configuration.").format(
                provider_name
            ),
            provider_type=provider_name,
            auth_type=apimodelsv1.ProviderAuthType.passthrough,
        )
    except ValidationError as err:
        logger.warning(
            "Invalid provider name",
            provider_name=provider_name,
            provider_url=provider_url,
            err=str(err),
        )
        return None
