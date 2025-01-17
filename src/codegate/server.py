import traceback

import structlog
from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.errors import ServerErrorMiddleware

from codegate import __description__, __version__
from codegate.api.v1 import v1
from codegate.dashboard.dashboard import dashboard_router
from codegate.pipeline.factory import PipelineFactory
from codegate.providers.anthropic.provider import AnthropicProvider
from codegate.providers.llamacpp.provider import LlamaCppProvider
from codegate.providers.ollama.provider import OllamaProvider
from codegate.providers.openai.provider import OpenAIProvider
from codegate.providers.registry import ProviderRegistry
from codegate.providers.vllm.provider import VLLMProvider

logger = structlog.get_logger("codegate")


async def custom_error_handler(request, exc: Exception):
    """This is a Middleware to handle exceptions and log them."""
    # Capture the stack trace
    extracted_traceback = traceback.extract_tb(exc.__traceback__)
    # Log only the last 3 items of the stack trace. 3 is an arbitrary number.
    logger.error(traceback.print_list(extracted_traceback[-3:]))
    return JSONResponse({"error": str(exc)}, status_code=500)


def init_app(pipeline_factory: PipelineFactory) -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
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
    registry = ProviderRegistry(app)

    # Register all known providers
    registry.add_provider(
        "openai",
        OpenAIProvider(pipeline_factory),
    )
    registry.add_provider(
        "anthropic",
        AnthropicProvider(
            pipeline_factory,
        ),
    )
    registry.add_provider(
        "llamacpp",
        LlamaCppProvider(
            pipeline_factory,
        ),
    )
    registry.add_provider(
        "vllm",
        VLLMProvider(
            pipeline_factory,
        ),
    )
    registry.add_provider(
        "ollama",
        OllamaProvider(
            pipeline_factory,
        ),
    )

    # Create and add system routes
    system_router = APIRouter(tags=["System"])

    @system_router.get("/health")
    async def health_check():
        return {"status": "healthy"}

    app.include_router(system_router)
    app.include_router(dashboard_router)

    # CodeGate API
    app.include_router(v1, prefix="/api/v1", tags=["CodeGate API"])

    return app
