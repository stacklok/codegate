import json
import traceback
from unittest.mock import Mock

import structlog
from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.errors import ServerErrorMiddleware

from codegate import __description__, __version__
from codegate.api.v1 import v1
from codegate.db.models import ProviderType
from codegate.muxing.router import MuxRouter
from codegate.pipeline.factory import PipelineFactory
from codegate.providers.anthropic.provider import AnthropicProvider
from codegate.providers.llamacpp.provider import LlamaCppProvider
from codegate.providers.lm_studio.provider import LmStudioProvider
from codegate.providers.ollama.provider import OllamaProvider
from codegate.providers.openai.provider import OpenAIProvider
from codegate.providers.openrouter.provider import OpenRouterProvider
from codegate.providers.registry import ProviderRegistry, get_provider_registry
from codegate.providers.vllm.provider import VLLMProvider

logger = structlog.get_logger("codegate")


async def custom_error_handler(request, exc: Exception):
    """This is a Middleware to handle exceptions and log them."""
    # Capture the stack trace
    extracted_traceback = traceback.extract_tb(exc.__traceback__)
    # Log only the last 3 items of the stack trace. 3 is an arbitrary number.
    logger.error(traceback.print_list(extracted_traceback[-3:]), exc_info=exc)
    return JSONResponse({"error": str(exc)}, status_code=500)


class CodeGateServer(FastAPI):
    provider_registry: ProviderRegistry = None

    def set_provider_registry(self, registry: ProviderRegistry):
        self.provider_registry = registry


def init_app(pipeline_factory: PipelineFactory) -> CodeGateServer:
    """Create the FastAPI application."""
    app = CodeGateServer(
        title="CodeGate",
        description=__description__,
        version=__version__,
    )

    @app.middleware("http")
    async def log_user_agent(request: Request, call_next):
        user_agent = request.headers.get("user-agent")
        client_host = request.client.host if request.client else "unknown"
        logger.debug(f"User-Agent header received: {user_agent} from {client_host}")
        response = await call_next(request)
        return response

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Apply error handling middleware
    app.add_middleware(ServerErrorMiddleware, handler=custom_error_handler)

    # Create provider registry
    registry = get_provider_registry(app)
    app.set_provider_registry(registry)

    # Register all known providers
    registry.add_provider(
        ProviderType.openai,
        OpenAIProvider(pipeline_factory),
    )
    registry.add_provider(
        ProviderType.openrouter,
        OpenRouterProvider(pipeline_factory),
    )
    registry.add_provider(
        ProviderType.anthropic,
        AnthropicProvider(
            pipeline_factory,
        ),
    )
    registry.add_provider(
        ProviderType.llamacpp,
        LlamaCppProvider(
            pipeline_factory,
        ),
    )
    registry.add_provider(
        ProviderType.vllm,
        VLLMProvider(
            pipeline_factory,
        ),
    )
    registry.add_provider(
        ProviderType.ollama,
        OllamaProvider(
            pipeline_factory,
        ),
    )
    registry.add_provider(
        ProviderType.lm_studio,
        LmStudioProvider(
            pipeline_factory,
        ),
    )

    mux_router = MuxRouter(registry)
    app.include_router(mux_router.get_routes(), include_in_schema=False)

    # Create and add system routes
    system_router = APIRouter(tags=["System"])

    @system_router.get("/health")
    async def health_check():
        return {"status": "healthy"}

    app.include_router(system_router)

    # CodeGate API
    app.include_router(v1, prefix="/api/v1", tags=["CodeGate API"])

    return app


def generate_openapi():
    app = init_app(Mock(spec=PipelineFactory))

    # Generate OpenAPI JSON
    openapi_schema = app.openapi()

    # Convert the schema to JSON string for easier handling or storage
    openapi_json = json.dumps(openapi_schema, indent=2)
    print(openapi_json)
