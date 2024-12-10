from typing import List

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from codegate import __description__, __version__
from codegate.config import Config
from codegate.dashboard.dashboard import dashboard_router
from codegate.pipeline.base import PipelineStep, SequentialPipelineProcessor
from codegate.pipeline.codegate_context_retriever.codegate import CodegateContextRetriever
from codegate.pipeline.extract_snippets.extract_snippets import CodeSnippetExtractor
from codegate.pipeline.extract_snippets.output import CodeCommentStep
from codegate.pipeline.output import OutputPipelineProcessor, OutputPipelineStep
from codegate.pipeline.secrets.manager import SecretsManager
from codegate.pipeline.secrets.secrets import (
    CodegateSecrets,
    SecretRedactionNotifier,
    SecretUnredactionStep,
)
from codegate.pipeline.secrets.signatures import CodegateSignatures
from codegate.pipeline.system_prompt.codegate import SystemPrompt
from codegate.pipeline.version.version import CodegateVersion
from codegate.providers.anthropic.provider import AnthropicProvider
from codegate.providers.llamacpp.provider import LlamaCppProvider
from codegate.providers.ollama.provider import OllamaProvider
from codegate.providers.openai.provider import OpenAIProvider
from codegate.providers.registry import ProviderRegistry
from codegate.providers.vllm.provider import VLLMProvider


def init_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="CodeGate",
        description=__description__,
        version=__version__,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize secrets manager
    # TODO: we need to clean up the secrets manager
    # after the conversation is concluded
    # this was done in the pipeline step but I just removed it for now
    secrets_manager = SecretsManager()

    # Define input pipeline steps
    input_steps: List[PipelineStep] = [
        CodegateVersion(),
        CodeSnippetExtractor(),
        SystemPrompt(Config.get_config().prompts.default_chat),
        CodegateContextRetriever(),
        CodegateSecrets(),
    ]

    # Define FIM pipeline steps
    fim_steps: List[PipelineStep] = [
        CodegateSecrets(),
    ]

    # Initialize input pipeline processors
    input_pipeline_processor = SequentialPipelineProcessor(input_steps, secrets_manager)
    fim_pipeline_processor = SequentialPipelineProcessor(fim_steps, secrets_manager)

    # Define output pipeline steps
    output_steps: List[OutputPipelineStep] = [
        SecretRedactionNotifier(),
        SecretUnredactionStep(),
        CodeCommentStep(),
    ]
    fim_output_steps: List[OutputPipelineStep] = [
        # temporarily disabled
        # SecretUnredactionStep(),
    ]

    output_pipeline_processor = OutputPipelineProcessor(output_steps)
    fim_output_pipeline_processor = OutputPipelineProcessor(fim_output_steps)

    # Create provider registry
    registry = ProviderRegistry(app)

    # Initialize SignaturesFinder
    CodegateSignatures.initialize("signatures.yaml")

    # Register all known providers
    registry.add_provider(
        "openai",
        OpenAIProvider(
            pipeline_processor=input_pipeline_processor,
            fim_pipeline_processor=fim_pipeline_processor,
            output_pipeline_processor=output_pipeline_processor,
            fim_output_pipeline_processor=fim_output_pipeline_processor,
        ),
    )
    registry.add_provider(
        "anthropic",
        AnthropicProvider(
            pipeline_processor=input_pipeline_processor,
            fim_pipeline_processor=fim_pipeline_processor,
            output_pipeline_processor=output_pipeline_processor,
            fim_output_pipeline_processor=fim_output_pipeline_processor,
        ),
    )
    registry.add_provider(
        "llamacpp",
        LlamaCppProvider(
            pipeline_processor=input_pipeline_processor,
            fim_pipeline_processor=fim_pipeline_processor,
            output_pipeline_processor=output_pipeline_processor,
            fim_output_pipeline_processor=fim_output_pipeline_processor,
        ),
    )
    registry.add_provider(
        "vllm",
        VLLMProvider(
            pipeline_processor=input_pipeline_processor,
            fim_pipeline_processor=fim_pipeline_processor,
            output_pipeline_processor=output_pipeline_processor,
            fim_output_pipeline_processor=fim_output_pipeline_processor,
        ),
    )
    registry.add_provider(
        "ollama",
        OllamaProvider(
            pipeline_processor=input_pipeline_processor,
            fim_pipeline_processor=fim_pipeline_processor,
            output_pipeline_processor=output_pipeline_processor,
            fim_output_pipeline_processor=fim_output_pipeline_processor,
        ),
    )

    # Create and add system routes
    system_router = APIRouter(tags=["System"])  # Tags group endpoints in the docs

    @system_router.get("/health")
    async def health_check():
        return {"status": "healthy"}

    # Include the router in the app - this exposes the health check endpoint
    app.include_router(system_router)

    # Include the routes for the dashboard
    app.include_router(dashboard_router)

    return app
