from typing import List

from codegate.config import Config
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
from codegate.pipeline.system_prompt.codegate import SystemPrompt
from codegate.pipeline.version.version import CodegateVersion


class PipelineFactory:
    def __init__(self, secrets_manager: SecretsManager):
        self.secrets_manager = secrets_manager

    def create_input_pipeline(self) -> SequentialPipelineProcessor:
        input_steps: List[PipelineStep] = [
            # make sure that this step is always first in the pipeline
            # the other steps might send the request to a LLM for it to be analyzed
            # and without obfuscating the secrets, we'd leak the secrets during those
            # later steps
            CodegateSecrets(),
            CodegateVersion(),
            CodeSnippetExtractor(),
            SystemPrompt(Config.get_config().prompts.default_chat),
            CodegateContextRetriever(),
        ]
        return SequentialPipelineProcessor(input_steps, self.secrets_manager)

    def create_fim_pipeline(self) -> SequentialPipelineProcessor:
        fim_steps: List[PipelineStep] = [
            CodegateSecrets(),
        ]
        return SequentialPipelineProcessor(fim_steps, self.secrets_manager)

    def create_output_pipeline(self) -> OutputPipelineProcessor:
        output_steps: List[OutputPipelineStep] = [
            SecretRedactionNotifier(),
            SecretUnredactionStep(),
            CodeCommentStep(),
        ]
        return OutputPipelineProcessor(output_steps)

    def create_fim_output_pipeline(self) -> OutputPipelineProcessor:
        fim_output_steps: List[OutputPipelineStep] = [
            # temporarily disabled
            # SecretUnredactionStep(),
        ]
        return OutputPipelineProcessor(fim_output_steps)
