from threading import Lock
from typing import Dict, Optional

from fastapi import FastAPI

from codegate.providers.base import BaseProvider

_provider_registry_lock = Lock()
_provider_registry_singleton: Optional["ProviderRegistry"] = None


def get_provider_registry(app: FastAPI = None) -> "ProviderRegistry":
    global _provider_registry_singleton

    if _provider_registry_singleton is None:
        if app is None:
            raise ValueError("Cannot initialize a ProviderRegistry without an app")
        with _provider_registry_lock:
            if _provider_registry_singleton is None:
                _provider_registry_singleton = ProviderRegistry(app)

    return _provider_registry_singleton


class ProviderRegistry:
    def __init__(self, app: FastAPI):
        self.app = app
        self.providers: Dict[str, BaseProvider] = {}

    def add_provider(self, name: str, provider: BaseProvider):
        """
        Adds a provider to the registry. This will also add the provider's routes
        to the FastAPI app.
        """
        self.providers[name] = provider
        self.app.include_router(provider.get_routes(), include_in_schema=False)

    def get_provider(self, name: str) -> Optional[BaseProvider]:
        """
        Retrieves a provider from the registry by name.
        """
        return self.providers.get(name)
