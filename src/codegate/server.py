from typing import List

from fastapi import APIRouter, FastAPI

from codegate import __description__, __version__
from codegate.config import Config
from codegate.pipeline.base import PipelineStep, SequentialPipelineProcessor
from codegate.pipeline.codegate_system_prompt.codegate import CodegateSystemPrompt
from codegate.pipeline.version.version import CodegateVersion
from codegate.providers.anthropic.provider import AnthropicProvider
from codegate.providers.llamacpp.provider import LlamaCppProvider
from codegate.providers.openai.provider import OpenAIProvider
from codegate.providers.registry import ProviderRegistry
from codegate.providers.vllm.provider import VLLMProvider


def init_app() -> FastAPI:
    app = FastAPI(
        title="CodeGate",
        description=__description__,
        version=__version__,
    )

    steps: List[PipelineStep] = [
        CodegateVersion(),
        CodegateSystemPrompt(Config.get_config().prompts.codegate_chat),
        # CodegateSecrets(),
    ]
    # Leaving the pipeline empty for now
    fim_steps: List[PipelineStep] = []
    pipeline = SequentialPipelineProcessor(steps)
    fim_pipeline = SequentialPipelineProcessor(fim_steps)

    # Create provider registry
    registry = ProviderRegistry(app)

    # Initialize SignaturesFinder
    # CodegateSignatures.initialize("signatures.yaml")

    # Register all known providers
    registry.add_provider(
        "openai", OpenAIProvider(pipeline_processor=pipeline, fim_pipeline_processor=fim_pipeline)
    )
    registry.add_provider(
        "anthropic",
        AnthropicProvider(pipeline_processor=pipeline, fim_pipeline_processor=fim_pipeline),
    )
    registry.add_provider(
        "llamacpp",
        LlamaCppProvider(pipeline_processor=pipeline, fim_pipeline_processor=fim_pipeline),
    )
    registry.add_provider(
        "vllm", VLLMProvider(pipeline_processor=pipeline, fim_pipeline_processor=fim_pipeline)
    )

    # Create and add system routes
    system_router = APIRouter(tags=["System"])  # Tags group endpoints in the docs

    @system_router.get("/health")
    async def health_check():
        return {"status": "healthy"}

    # Include the router in the app - this exposes the health check endpoint
    app.include_router(system_router)

    return app
