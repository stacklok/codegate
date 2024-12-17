from litellm import ChatCompletionRequest

from codegate import __version__
from codegate.pipeline.base import (
    PipelineContext,
    PipelineResponse,
    PipelineResult,
    PipelineStep,
)


class CodegateVersion(PipelineStep):
    """Pipeline step that handles version information requests."""

    @property
    def name(self) -> str:
        """
        Returns the name of this pipeline step.

        Returns:
            str: The identifier 'codegate-version'
        """
        return "codegate-version"

    async def process(
        self, request: ChatCompletionRequest, context: PipelineContext
    ) -> PipelineResult:
        """
        Checks if the last user message contains "codegate-version" and
        responds with the current version.
        This short-circuits the pipeline if the message is found.

        Args:
            request (ChatCompletionRequest): The chat completion request to process
            context (PipelineContext): The current pipeline context

        Returns:
            PipelineResult: Contains version response if triggered, otherwise continues
            pipeline
        """
        last_user_message = self.get_last_user_message(request)

        if last_user_message is not None:
            last_user_message_str, _ = last_user_message
            if "codegate-version" in last_user_message_str.lower():
                context.add_alert(self.name, trigger_string=last_user_message_str)
                return PipelineResult(
                    response=PipelineResponse(
                        step_name=self.name,
                        content="CodeGate version: {}".format(__version__),
                        model=request["model"],
                    ),
                    context=context,
                )

        # Fall through
        return PipelineResult(request=request, context=context)
