from urllib.parse import urljoin

import structlog

from codegate.config import Config
from codegate.db import models as db_models
from codegate.muxing import rulematcher

logger = structlog.get_logger("codegate")


class MuxingAdapterError(Exception):
    pass


# Note: this is yet another awful hack to get the correct folder where
# llamacpp models are stored. This is currently retrieved inside the
# providers, but it should probably be refactored and injected,
# implementing a basic inversion-of-control pattern.
def get_llamacpp_models_folder():
    override = Config.get_config().provider_urls.get("llamacpp")
    return override if override else "./codegate_volume/models"


def get_provider_formatted_url(model_route: rulematcher.ModelRoute) -> str:
    """Get the provider formatted URL to use in base_url. Note this value comes from DB"""
    if model_route.endpoint.provider_type in [
        db_models.ProviderType.openai,
        db_models.ProviderType.vllm,
    ]:
        return urljoin(model_route.endpoint.endpoint, "/v1")
    if model_route.endpoint.provider_type == db_models.ProviderType.openrouter:
        return urljoin(model_route.endpoint.endpoint, "/api/v1")
    if model_route.endpoint.provider_type == db_models.ProviderType.llamacpp:
        return get_llamacpp_models_folder()
    return model_route.endpoint.endpoint


def get_destination_info(model_route: rulematcher.ModelRoute) -> dict:
    """Set the destination provider info."""
    return model_route.model.name, get_provider_formatted_url(model_route)
