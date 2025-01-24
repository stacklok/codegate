import re
import shlex

from litellm import ChatCompletionRequest

from codegate.pipeline.base import (
    PipelineContext,
    PipelineResponse,
    PipelineResult,
    PipelineStep,
)
from codegate.pipeline.cli.commands import CustomInstructions, Version, Workspace

HELP_TEXT = """
## CodeGate CLI\n
**Usage**: `codegate [-h] <command> [args]`\n
Check the help of each command by running `codegate <command> -h`\n
Available commands:
- `version`: Show the version of CodeGate
- `workspace`: Perform different operations on workspaces
- `custom-instructions`: Set custom instructions for the workspace
"""

NOT_FOUND_TEXT = "Command not found. Use `codegate -h` to see available commands."


async def codegate_cli(command):
    """
    Process the 'codegate' command.
    """
    if len(command) == 0:
        return HELP_TEXT

    available_commands = {
        "version": Version().exec,
        "workspace": Workspace().exec,
        "custom-instructions": CustomInstructions().exec,
    }
    out_func = available_commands.get(command[0])
    if out_func is None:
        if command[0] == "-h":
            return HELP_TEXT
        return NOT_FOUND_TEXT

    return await out_func(command[1:])


class CodegateCli(PipelineStep):
    """Pipeline step that handles codegate cli."""

    @property
    def name(self) -> str:
        """
        Returns the name of this pipeline step.

        Returns:
            str: The identifier 'codegate-cli'
        """
        return "codegate-cli"

    async def process(
        self, request: ChatCompletionRequest, context: PipelineContext
    ) -> PipelineResult:
        """
        Checks if the last user message contains "codegate" and process the command.
        This short-circuits the pipeline if the message is found.

        Args:
            request (ChatCompletionRequest): The chat completion request to process
            context (PipelineContext): The current pipeline context

        Returns:
            PipelineResult: Contains the response if triggered, otherwise continues
            pipeline
        """
        last_user_message = self.get_last_user_message(request)

        if last_user_message is not None:
            last_user_message_str, _ = last_user_message
            last_user_message_str = last_user_message_str.strip()
            is_cline_client = any(
                "Cline" in str(message.get("content", "")) for message in request.get("messages",
                                                                                      [])
            )
            if not is_cline_client:
                # Check if "codegate" is the first word in the message
                match = re.match(r"^codegate(?:\s+(\S+))?", last_user_message_str, re.IGNORECASE)
            else:
                # Check if "codegate" is the first word after the first XML tag
                xml_start = re.search(r"<[^>]+>", last_user_message_str)
                if xml_start:
                    # Start processing only from the first XML tag
                    relevant_message = last_user_message_str[xml_start.start():]
                    # Remove all XML tags and trim whitespace
                    stripped_message = re.sub(r"<[^>]+>", "", relevant_message).strip()
                    # Check if "codegate" is the first word
                    match = re.match(r"^codegate(?:\s+(\S+))?", stripped_message, re.IGNORECASE)
                else:
                    match = None
            if match:
                command = match.group(1)  # Extract the second word
                command = command.strip()

                # Process the command
                args = shlex.split(f"codegate {command}")
                if args:
                    cmd_out = await codegate_cli(args[1:])

                    if is_cline_client:
                        cmd_out = (
                            f"<attempt_completion><result>{cmd_out}</result></attempt_completion>\n"
                        )
                    return PipelineResult(
                        response=PipelineResponse(
                            step_name=self.name,
                            content=cmd_out,
                            model=request["model"],
                        ),
                        context=context,
                    )

        # Fall through
        return PipelineResult(request=request, context=context)
