import asyncio

from litellm import ChatCompletionRequest

from codegate.db.connection import DbReader
from codegate.pipeline.base import (
    PipelineContext,
    PipelineResponse,
    PipelineResult,
    PipelineStep,
)


class WorkspaceCommands:

    def __init__(self):
        self._db_recorder = DbReader()
        self.commands = {
            "list": self._list_workspaces,
        }

    async def _list_workspaces(self, *args):
        """
        List all workspaces
        """
        workspaces = await self._db_recorder.get_workspaces()
        print(workspaces)
        respond_str = ""
        for workspace in workspaces:
            respond_str += f"{workspace.id} - {workspace.name}"
            if workspace.is_active:
                respond_str += " (active)"
            respond_str += "\n"
        return respond_str

    async def execute(self, command: str, *args) -> str:
        """
        Execute the given command

        Args:
            command (str): The command to execute
        """
        command_to_execute = self.commands.get(command)
        if command_to_execute is not None:
            return await command_to_execute(*args)
        else:
            return "Command not found"

    async def parse_execute_cmd(self, last_user_message: str) -> str:
        """
        Parse the last user message and execute the command

        Args:
            last_user_message (str): The last user message
        """
        command_and_args = last_user_message.split("codegate-workspace ")[1]
        command, *args = command_and_args.split(" ")
        return await self.execute(command, *args)


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
