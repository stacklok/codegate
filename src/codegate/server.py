from fastapi import APIRouter, FastAPI

from . import __description__, __version__
from .providers.anthropic.provider import AnthropicProvider
from .providers.openai.provider import OpenAIProvider
from .providers.registry import ProviderRegistry


def init_app() -> FastAPI:
    app = FastAPI(
        title="CodeGate",
        description=__description__,
        version=__version__,
    )

    # Create provider registry
    registry = ProviderRegistry(app)

    # Register all known providers
    registry.add_provider("openai", OpenAIProvider())
    registry.add_provider("anthropic", AnthropicProvider())

    # Create and add system routes
    system_router = APIRouter(tags=["System"])  # Tags group endpoints in the docs

    @system_router.get("/health")
    async def health_check():
        return {"status": "healthy"}

    # Include the router in the app - this exposes the health check endpoint
    app.include_router(system_router)

    return app
