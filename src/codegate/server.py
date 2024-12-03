from typing import List

from fastapi import APIRouter, FastAPI

from codegate import __description__, __version__
from codegate.config import Config
from codegate.pipeline.base import PipelineStep, SequentialPipelineProcessor
from codegate.pipeline.codegate_context_retriever.codegate import CodegateContextRetriever
from codegate.pipeline.codegate_system_prompt.codegate import CodegateSystemPrompt
from codegate.pipeline.extract_snippets.extract_snippets import CodeSnippetExtractor
from codegate.pipeline.extract_snippets.output import CodeCommentStep
from codegate.pipeline.output import OutputPipelineProcessor, OutputPipelineStep
from codegate.pipeline.secrets.manager import SecretsManager
from codegate.pipeline.secrets.secrets import CodegateSecrets, SecretUnredactionStep
from codegate.pipeline.secrets.signatures import CodegateSignatures
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

    # Initialize secrets manager
    # TODO: we need to clean up the secrets manager
    # after the conversation is concluded
    # this was done in the pipeline step but I just removed it for now
    secrets_manager = SecretsManager()

    steps: List[PipelineStep] = [
        CodegateVersion(),
        CodeSnippetExtractor(),
        CodegateSystemPrompt(Config.get_config().prompts.codegate_chat),
        CodegateContextRetriever(Config.get_config().prompts.codegate_chat),
        CodegateSecrets(),
    ]
    # Leaving the pipeline empty for now
    fim_steps: List[PipelineStep] = []
    pipeline = SequentialPipelineProcessor(steps)
    fim_pipeline = SequentialPipelineProcessor(fim_steps)

    output_steps: List[OutputPipelineStep] = [
        SecretUnredactionStep(),
        CodeCommentStep(),
    ]
    output_pipeline = OutputPipelineProcessor(output_steps)

    # Create provider registry
    registry = ProviderRegistry(app)

    # Initialize SignaturesFinder
    CodegateSignatures.initialize("signatures.yaml")

    # Register all known providers
    registry.add_provider(
        "openai",
        OpenAIProvider(
            secrets_manager=secrets_manager,
            pipeline_processor=pipeline,
            fim_pipeline_processor=fim_pipeline,
            output_pipeline_processor=output_pipeline,
        ),
    )
    registry.add_provider(
        "anthropic",
        AnthropicProvider(
            secrets_manager=secrets_manager,
            pipeline_processor=pipeline,
            fim_pipeline_processor=fim_pipeline,
            output_pipeline_processor=output_pipeline,
        ),
    )
    registry.add_provider(
        "llamacpp",
        LlamaCppProvider(
            secrets_manager=secrets_manager,
            pipeline_processor=pipeline,
            fim_pipeline_processor=fim_pipeline,
            output_pipeline_processor=output_pipeline,
        ),
    )
    registry.add_provider(
        "vllm",
        VLLMProvider(
            secrets_manager=secrets_manager,
            pipeline_processor=pipeline,
            fim_pipeline_processor=fim_pipeline,
            output_pipeline_processor=output_pipeline,
        ),
    )
    registry.add_provider(
        "ollama",
        OllamaProvider(
            secrets_manager=secrets_manager,
            pipeline_processor=pipeline,
            fim_pipeline_processor=fim_pipeline,
            output_pipeline_processor=output_pipeline,
        ),
    )

    # Create and add system routes
    system_router = APIRouter(tags=["System"])  # Tags group endpoints in the docs

    @system_router.get("/health")
    async def health_check():
        return {"status": "healthy"}

    # Include the router in the app - this exposes the health check endpoint
    app.include_router(system_router)

    return app
