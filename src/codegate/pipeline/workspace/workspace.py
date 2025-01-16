from litellm import ChatCompletionRequest

from codegate.pipeline.base import (
    PipelineContext,
    PipelineResponse,
    PipelineResult,
    PipelineStep,
)
from codegate.pipeline.workspace.commands import WorkspaceCommands


class CodegateWorkspace(PipelineStep):
    """Pipeline step that handles workspace information requests."""

    @property
    def name(self) -> str:
        """
        Returns the name of this pipeline step.

        Returns:
            str: The identifier 'codegate-workspace'
        """
        return "codegate-workspace"

    async def process(
        self, request: ChatCompletionRequest, context: PipelineContext
    ) -> PipelineResult:
        """
        Checks if the last user message contains "codegate-workspace" and
        responds with command specified.
        This short-circuits the pipeline if the message is found.

        Args:
            request (ChatCompletionRequest): The chat completion request to process
            context (PipelineContext): The current pipeline context

        Returns:
            PipelineResult: Contains workspace response if triggered, otherwise continues
            pipeline
        """
        last_user_message = self.get_last_user_message(request)

        if last_user_message is not None:
            last_user_message_str, _ = last_user_message
            if "codegate-workspace" in last_user_message_str.lower():
                context.shortcut_response = True
                command_output = await WorkspaceCommands().parse_execute_cmd(last_user_message_str)
                return PipelineResult(
                    response=PipelineResponse(
                        step_name=self.name,
                        content=command_output,
                        model=request["model"],
                    ),
                    context=context,
                )

        # Fall through
        return PipelineResult(request=request, context=context)
