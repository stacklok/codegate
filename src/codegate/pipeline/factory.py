from typing import List

from codegate.clients.clients import ClientType
from codegate.config import Config
from codegate.pipeline.base import PipelineStep, SequentialPipelineProcessor
from codegate.pipeline.cli.cli import CodegateCli
from codegate.pipeline.codegate_context_retriever.codegate import CodegateContextRetriever
from codegate.pipeline.comment.output import CodeCommentStep
from codegate.pipeline.output import OutputPipelineProcessor, OutputPipelineStep
from codegate.pipeline.pii.pii import (
    CodegatePii,
    PiiRedactionNotifier,
    PiiUnRedactionStep,
)
from codegate.pipeline.secrets.manager import SecretsManager
from codegate.pipeline.secrets.secrets import (
    CodegateSecrets,
    SecretRedactionNotifier,
    SecretUnredactionStep,
)
from codegate.pipeline.system_prompt.codegate import SystemPrompt


class PipelineFactory:
    def __init__(self, secrets_manager: SecretsManager):
        self.secrets_manager = secrets_manager

    def create_input_pipeline(self, client_type: ClientType) -> SequentialPipelineProcessor:
        input_steps: List[PipelineStep] = [
            # make sure that this step is always first in the pipeline
            # the other steps might send the request to a LLM for it to be analyzed
            # and without obfuscating the secrets, we'd leak the secrets during those
            # later steps
            CodegateSecrets(),
            CodegatePii(),
            CodegateCli(),
            CodegateContextRetriever(),
            SystemPrompt(
                Config.get_config().prompts.default_chat, Config.get_config().prompts.client_prompts
            ),
        ]
        return SequentialPipelineProcessor(
            input_steps,
            self.secrets_manager,
            client_type,
            is_fim=False,
        )

    def create_fim_pipeline(self, client_type: ClientType) -> SequentialPipelineProcessor:
        fim_steps: List[PipelineStep] = [
            CodegateSecrets(),
            CodegatePii(),
        ]
        return SequentialPipelineProcessor(
            fim_steps,
            self.secrets_manager,
            client_type,
            is_fim=True,
        )

    def create_output_pipeline(self) -> OutputPipelineProcessor:
        output_steps: List[OutputPipelineStep] = [
            SecretRedactionNotifier(),
            SecretUnredactionStep(),
            PiiRedactionNotifier(),
            PiiUnRedactionStep(),
            CodeCommentStep(),
        ]
        return OutputPipelineProcessor(output_steps)

    def create_fim_output_pipeline(self) -> OutputPipelineProcessor:
        fim_output_steps: List[OutputPipelineStep] = [
            # temporarily disabled
            # SecretUnredactionStep(),
        ]
        return OutputPipelineProcessor(fim_output_steps)
