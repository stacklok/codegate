import datetime

from litellm import ChatCompletionRequest

from codegate.db.connection import DbReader, DbRecorder
from codegate.pipeline.base import (
    PipelineContext,
    PipelineResponse,
    PipelineResult,
    PipelineStep,
)


class WorkspaceCommands:

    def __init__(self):
        self._db_reader = DbReader()
        self.commands = {
            "list": self._list_workspaces,
            "add": self._add_workspace,
            "activate": self._activate_workspace,
        }

    async def _list_workspaces(self, *args) -> str:
        """
        List all workspaces
        """
        workspaces = await self._db_reader.get_workspaces()
        respond_str = ""
        for workspace in workspaces:
            respond_str += f"- {workspace.name}"
            if workspace.active_workspace_id:
                respond_str += " **(active)**"
            respond_str += "\n"
        return respond_str

    async def _add_workspace(self, *args) -> str:
        """
        Add a workspace
        """
        if args is None or len(args) == 0:
            return "Please provide a name. Use `codegate-workspace add your_workspace_name`"

        new_workspace_name = args[0]
        if not new_workspace_name:
            return "Please provide a name. Use `codegate-workspace add your_workspace_name`"

        workspaces = await self._db_reader.get_workspace_by_name(new_workspace_name)
        if workspaces:
            return f"Workspace **{new_workspace_name}** already exists"

        db_recorder = DbRecorder()
        await db_recorder.add_workspace(new_workspace_name)
        return f"Workspace **{new_workspace_name}** has been added"

    async def _activate_workspace(self, *args) -> str:
        """
        Activate a workspace
        """
        if args is None or len(args) == 0:
            return "Please provide a name. Use `codegate-workspace activate workspace_name`"

        workspace_name = args[0]
        if not workspace_name:
            return "Please provide a name. Use `codegate-workspace activate workspace_name`"

        workspaces = await self._db_reader.get_workspace_by_name(workspace_name)
        if not workspaces:
            return (
                f"Workspace **{workspace_name}** does not exist. "
                f"Use `codegate-workspace add {workspace_name}` to add it"
            )
        selected_workspace = workspaces[0]

        sessions = await self._db_reader.get_sessions()
        # The current implementation expects only one active session
        if len(sessions) != 1:
            return "Something went wrong. No active session found."

        session = sessions[0]
        if session.active_workspace_id == selected_workspace.id:
            return f"Workspace **{workspace_name}** is already active"

        session.active_workspace_id = selected_workspace.id
        session.last_update = datetime.datetime.now(datetime.timezone.utc)
        db_recorder = DbRecorder()
        await db_recorder.update_session(session)
        return f"Workspace **{workspace_name}** has been activated"

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
        command_and_args = last_user_message.lower().split("codegate-workspace ")[1]
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
