import shlex
from typing import Optional

import regex as re
from litellm import ChatCompletionRequest

from codegate.clients.clients import ClientType
from codegate.pipeline.base import (
    PipelineContext,
    PipelineResponse,
    PipelineResult,
    PipelineStep,
)
from codegate.pipeline.cli.commands import CustomInstructions, Version, Workspace

codegate_regex = re.compile(r"^codegate(?:\s+(.*))?", re.IGNORECASE)

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


def _get_cli_from_cline(
    codegate_regex: re.Pattern[str], last_user_message_str: str
) -> Optional[re.Match[str]]:
    # Check if there are <task> or <feedback> tags
    tag_match = re.search(r"<(task|feedback)>(.*?)</\1>", last_user_message_str, re.DOTALL)
    if tag_match:
        # Extract the content between the tags
        stripped_message = tag_match.group(2).strip()
    else:
        # If no <task> or <feedback> tags, use the entire message
        stripped_message = last_user_message_str.strip()

    # Remove all other XML tags and trim whitespace
    stripped_message = re.sub(r"<[^>]+>", "", stripped_message).strip()

    # Check if "codegate" is the first word
    match = codegate_regex.match(stripped_message)

    return match


def _get_cli_from_open_interpreter(last_user_message_str: str) -> Optional[re.Match[str]]:
    # Extract the last "### User:" block
    user_blocks = re.findall(r"### User:\s*(.*?)(?=\n###|\Z)", last_user_message_str, re.DOTALL)
    last_user_block = user_blocks[-1].strip() if user_blocks else last_user_message_str.strip()

    # Match "codegate" only in the last user block or entire input
    return re.match(r"^codegate\s*(.*?)\s*$", last_user_block, re.IGNORECASE)


def _get_cli_from_continue(last_user_message_str: str) -> Optional[re.Match[str]]:
    """
    Continue sends a differently formatted message to the CLI if DeepSeek is used
    """
    deepseek_match = re.search(
        r"utilizing the DeepSeek Coder model.*?### Instruction:\s*codegate\s+(.*?)\s*### Response:",
        last_user_message_str,
        re.DOTALL | re.IGNORECASE,
    )
    if deepseek_match:
        command = deepseek_match.group(1).strip()
        return re.match(r"^(.*?)$", command)  # This creates a match object with the command

    return codegate_regex.match(last_user_message_str)


def _get_cli_from_copilot(last_user_message_str: str) -> Optional[re.Match[str]]:
    """
    Process Copilot-specific CLI command format.

    Copilot sends messages in the format:
    <attachment>file contents</attachment>codegate command

    Args:
        last_user_message_str (str): The message string from Copilot

    Returns:
        Optional[re.Match[str]]: A regex match object if command is found, None otherwise
    """
    cleaned_text = re.sub(
        r"<attachment>.*</attachment>", "", last_user_message_str, flags=re.DOTALL
    )
    return codegate_regex.match(cleaned_text.strip())


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

            # Check client-specific matchers first
            if context.client in [ClientType.CLINE, ClientType.KODU]:
                match = _get_cli_from_cline(codegate_regex, last_user_message_str)
            elif context.client in [ClientType.OPEN_INTERPRETER]:
                match = _get_cli_from_open_interpreter(last_user_message_str)
            elif context.client in [ClientType.CONTINUE]:
                match = _get_cli_from_continue(last_user_message_str)
            elif context.client in [ClientType.COPILOT]:
                match = _get_cli_from_copilot(last_user_message_str)
            else:
                # Check if "codegate" is the first word in the message
                match = codegate_regex.match(last_user_message_str)

            if match:
                command = match.group(1) or ""
                command = command.strip()

                # Process the command
                args = shlex.split(f"codegate {command}")
                if args:
                    context.shortcut_response = True
                    cmd_out = await codegate_cli(args[1:])
                    if context.client in [ClientType.CLINE, ClientType.KODU]:
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
