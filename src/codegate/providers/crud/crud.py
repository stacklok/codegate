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
from codegate.providers.registry import ProviderRegistry, get_provider_registry

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
        self, endpoint: apimodelsv1.AddProviderEndpointRequest
    ) -> apimodelsv1.ProviderEndpoint:
        """Add an endpoint."""

        if not endpoint.endpoint:
            endpoint.endpoint = provider_default_endpoints(endpoint.provider_type)

        # If we STILL don't have an endpoint, we can't continue
        if not endpoint.endpoint:
            raise ValueError("No endpoint provided and no default found for provider type")

        dbend = endpoint.to_db_model()
        provider_registry = get_provider_registry()

        # We override the ID here, as we want to generate it.
        dbend.id = str(uuid4())

        prov = endpoint.get_from_registry(provider_registry)
        if prov is None:
            raise ValueError("Unknown provider type: {}".format(endpoint.provider_type))

        models = []
        if endpoint.auth_type == apimodelsv1.ProviderAuthType.api_key and not endpoint.api_key:
            raise ValueError("API key must be provided for API auth type")
        if endpoint.auth_type != apimodelsv1.ProviderAuthType.passthrough:
            try:
                models = prov.models(endpoint=endpoint.endpoint, api_key=endpoint.api_key)
            except Exception as err:
                raise ValueError("Unable to get models from provider: {}".format(str(err)))

        dbendpoint = await self._db_writer.add_provider_endpoint(dbend)

        await self._db_writer.push_provider_auth_material(
            dbmodels.ProviderAuthMaterial(
                provider_endpoint_id=dbendpoint.id,
                auth_type=endpoint.auth_type,
                auth_blob=endpoint.api_key if endpoint.api_key else "",
            )
        )

        for model in models:
            await self._db_writer.add_provider_model(
                dbmodels.ProviderModel(
                    provider_endpoint_id=dbendpoint.id,
                    name=model,
                )
            )
        return apimodelsv1.ProviderEndpoint.from_db_model(dbendpoint)

    async def update_endpoint(
        self, endpoint: apimodelsv1.AddProviderEndpointRequest
    ) -> apimodelsv1.ProviderEndpoint:
        """Update an endpoint."""

        if not endpoint.endpoint:
            endpoint.endpoint = provider_default_endpoints(endpoint.provider_type)

        # If we STILL don't have an endpoint, we can't continue
        if not endpoint.endpoint:
            raise ValueError("No endpoint provided and no default found for provider type")

        provider_registry = get_provider_registry()
        prov = endpoint.get_from_registry(provider_registry)
        if prov is None:
            raise ValueError("Unknown provider type: {}".format(endpoint.provider_type))

        founddbe = await self._db_reader.get_provider_endpoint_by_id(str(endpoint.id))
        if founddbe is None:
            raise ProviderNotFoundError("Provider not found")

        models = []
        if endpoint.auth_type == apimodelsv1.ProviderAuthType.api_key and not endpoint.api_key:
            raise ValueError("API key must be provided for API auth type")
        if endpoint.auth_type != apimodelsv1.ProviderAuthType.passthrough:
            try:
                models = prov.models(endpoint=endpoint.endpoint, api_key=endpoint.api_key)
            except Exception as err:
                raise ValueError("Unable to get models from provider: {}".format(str(err)))

        # Reset all provider models.
        await self._db_writer.delete_provider_models(str(endpoint.id))

        for model in models:
            await self._db_writer.add_provider_model(
                dbmodels.ProviderModel(
                    provider_endpoint_id=founddbe.id,
                    name=model,
                )
            )

        dbendpoint = await self._db_writer.update_provider_endpoint(endpoint.to_db_model())

        await self._db_writer.push_provider_auth_material(
            dbmodels.ProviderAuthMaterial(
                provider_endpoint_id=dbendpoint.id,
                auth_type=endpoint.auth_type,
                auth_blob=endpoint.api_key if endpoint.api_key else "",
            )
        )

        return apimodelsv1.ProviderEndpoint.from_db_model(dbendpoint)

    async def configure_auth_material(
        self, provider_id: UUID, config: apimodelsv1.ConfigureAuthMaterial
    ):
        """Add an API key."""
        if config.auth_type == apimodelsv1.ProviderAuthType.api_key and not config.api_key:
            raise ValueError("API key must be provided for API auth type")
        elif config.auth_type != apimodelsv1.ProviderAuthType.api_key and config.api_key:
            raise ValueError("API key provided for non-API auth type")

        dbendpoint = await self._db_reader.get_provider_endpoint_by_id(str(provider_id))
        if dbendpoint is None:
            raise ProviderNotFoundError("Provider not found")

        await self._db_writer.push_provider_auth_material(
            dbmodels.ProviderAuthMaterial(
                provider_endpoint_id=dbendpoint.id,
                auth_type=config.auth_type,
                auth_blob=config.api_key if config.api_key else "",
            )
        )

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
        if pimpl is None:
            logger.warning(
                "Provider not found in registry",
                provider=provend.name,
                endpoint=provend.endpoint,
            )
            continue
        await try_initialize_provider_endpoints(provend, pimpl, db_writer)


async def try_initialize_provider_endpoints(
    provend: apimodelsv1.ProviderEndpoint,
    pimpl: BaseProvider,
    db_writer: DbRecorder,
):
    try:
        models = pimpl.models()

        # If we were able to get the models, we don't need auth
        provend.auth_type = apimodelsv1.ProviderAuthType.none
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
            provider_type=provider_overrides(provider_name),
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


def provider_default_endpoints(provider_type: str) -> str:
    defaults = {
        "openai": "https://api.openai.com",
        "anthropic": "https://api.anthropic.com",
    }

    # If we have a default, we return it
    # Otherwise, we return an empty string
    return defaults.get(provider_type, "")


def provider_overrides(provider_type: str) -> str:
    overrides = {
        "lm_studio": "openai",
    }

    # If we have an override, we return it
    # Otherwise, we return the type
    return overrides.get(provider_type, provider_type)
