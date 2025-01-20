from abc import ABC, abstractmethod
from typing import Awaitable, Callable, Dict, List, Tuple

from pydantic import ValidationError

from codegate import __version__
from codegate.db.connection import AlreadyExistsError
from codegate.workspaces import crud


class NoFlagValueError(Exception):
    pass


class NoSubcommandError(Exception):
    pass


class CodegateCommand(ABC):
    @abstractmethod
    async def run(self, args: List[str]) -> str:
        pass

    @property
    @abstractmethod
    def command_name(self) -> str:
        pass

    @property
    @abstractmethod
    def help(self) -> str:
        pass

    async def exec(self, args: List[str]) -> str:
        if args and args[0] == "-h":
            return self.help
        return await self.run(args)


class Version(CodegateCommand):
    async def run(self, args: List[str]) -> str:
        return f"CodeGate version: {__version__}"

    @property
    def command_name(self) -> str:
        return "version"

    @property
    def help(self) -> str:
        return (
            "### CodeGate Version\n\n"
            "Prints the version of CodeGate.\n\n"
            "**Usage**: `codegate version`\n\n"
            "*args*: None"
        )


class CodegateCommandSubcommand(CodegateCommand):

    @property
    @abstractmethod
    def subcommands(self) -> Dict[str, Callable[[List[str]], Awaitable[str]]]:
        """
        List of subcommands that the command accepts.
        """
        pass

    @property
    @abstractmethod
    def flags(self) -> List[str]:
        """
        List of flags that the command accepts.
        Example: ["-w", "-f"]
        """
        pass

    def _parse_flags_and_subocomand(self, args: List[str]) -> Tuple[Dict[str, str], List[str], str]:
        """
        Reads the flags and subcommand from the args
        The flags are expected to be at the start of the args and are optional.
        """
        i = 0
        read_flags = {}
        # Parse all recognized flags at the start
        while i < len(args):
            if args[i] in self.flags:
                flag_name = args[i]
                if i + 1 >= len(args):
                    raise NoFlagValueError(f"Flag {flag_name} needs a value, but none provided.")
                read_flags[flag_name] = args[i + 1]
                i += 2
            else:
                # Once we encounter something that's not a recognized flag,
                # we assume it's the subcommand
                break

        if i >= len(args):
            raise NoSubcommandError("No subcommand found after optional flags.")

        subcommand = args[i]
        i += 1

        # The rest of the arguments after the subcommand
        rest = args[i:]
        return read_flags, rest, subcommand

    async def run(self, args: List[str]) -> str:
        """
        Try to parse the flags and subcommand and execute the subcommand
        """
        try:
            flags, rest, subcommand = self._parse_flags_and_subocomand(args)
        except NoFlagValueError:
            return (
                f"Error reading the command. Flag without value found. "
                f"Use `codegate {self.command_name} -h` to see available subcommands"
            )
        except NoSubcommandError:
            return (
                f"Submmand not found "
                f"Use `codegate {self.command_name} -h` to see available subcommands"
            )

        command_to_execute = self.subcommands.get(subcommand)
        if command_to_execute is None:
            return (
                f"Submmand not found "
                f"Use `codegate {self.command_name} -h` to see available subcommands"
            )

        return await command_to_execute(flags, rest)


class Workspace(CodegateCommandSubcommand):

    def __init__(self):
        self.workspace_crud = crud.WorkspaceCrud()

    @property
    def command_name(self) -> str:
        return "workspace"

    @property
    def flags(self) -> List[str]:
        """
        No flags for the workspace command
        """
        return []

    @property
    def subcommands(self) -> Dict[str, Callable[[List[str]], Awaitable[str]]]:
        return {
            "list": self._list_workspaces,
            "add": self._add_workspace,
            "activate": self._activate_workspace,
            "remove": self._remove_workspace,
        }

    async def _list_workspaces(self, flags: Dict[str, str], args: List[str]) -> str:
        """
        List all workspaces
        """
        workspaces = await self.workspace_crud.get_workspaces()
        respond_str = ""
        for workspace in workspaces:
            respond_str += f"- {workspace.name}"
            if workspace.active_workspace_id:
                respond_str += " **(active)**"
            respond_str += "\n"
        return respond_str

    async def _add_workspace(self, flags: Dict[str, str], args: List[str]) -> str:
        """
        Add a workspace
        """
        if args is None or len(args) == 0:
            return "Please provide a name. Use `codegate workspace add your_workspace_name`"

        new_workspace_name = args[0]
        if not new_workspace_name:
            return "Please provide a name. Use `codegate workspace add your_workspace_name`"

        try:
            _ = await self.workspace_crud.add_workspace(new_workspace_name)
        except ValidationError:
            return "Invalid workspace name: It should be alphanumeric and dashes"
        except AlreadyExistsError:
            return f"Workspace **{new_workspace_name}** already exists"
        except Exception:
            return "An error occurred while adding the workspace"

        return f"Workspace **{new_workspace_name}** has been added"

    async def _activate_workspace(self, flags: Dict[str, str], args: List[str]) -> str:
        """
        Activate a workspace
        """
        if args is None or len(args) == 0:
            return "Please provide a name. Use `codegate workspace activate workspace_name`"

        workspace_name = args[0]
        if not workspace_name:
            return "Please provide a name. Use `codegate workspace activate workspace_name`"

        try:
            await self.workspace_crud.activate_workspace(workspace_name)
        except crud.WorkspaceAlreadyActiveError:
            return f"Workspace **{workspace_name}** is already active"
        except crud.WorkspaceDoesNotExistError:
            return f"Workspace **{workspace_name}** does not exist"
        except Exception:
            return "An error occurred while activating the workspace"
        return f"Workspace **{workspace_name}** has been activated"

    async def _remove_workspace(self, args: List[str]) -> str:
        """
        Remove a workspace
        """
        if args is None or len(args) == 0:
            return "Please provide a name. Use `codegate workspace remove workspace_name`"

        workspace_name = args[0]
        if not workspace_name:
            return "Please provide a name. Use `codegate workspace remove workspace_name`"

        try:
            await self.workspace_crud.soft_delete_workspace(workspace_name)
        except crud.WorkspaceDoesNotExistError:
            return f"Workspace **{workspace_name}** does not exist"
        except Exception:
            return "An error occurred while removing the workspace"
        return f"Workspace **{workspace_name}** has been removed"

    @property
    def help(self) -> str:
        return (
            "### CodeGate Workspace\n\n"
            "Manage workspaces.\n\n"
            "**Usage**: `codegate workspace <command> [args]`\n\n"
            "Available commands:\n\n"
            "- `list`: List all workspaces\n\n"
            "  - *args*: None\n\n"
            "- `add`: Add a workspace\n\n"
            "  - *args*:\n\n"
            "    - `workspace_name`\n\n"
            "- `activate`: Activate a workspace\n\n"
            "  - *args*:\n\n"
            "    - `workspace_name`"
        )


class SystemPrompt(CodegateCommandSubcommand):

    def __init__(self):
        self.workspace_crud = crud.WorkspaceCrud()

    @property
    def command_name(self) -> str:
        return "system-prompt"

    @property
    def flags(self) -> List[str]:
        """
        Flags for the system-prompt command.
        -w: Workspace name
        """
        return ["-w"]

    @property
    def subcommands(self) -> Dict[str, Callable[[List[str]], Awaitable[str]]]:
        return {
            "set": self._set_system_prompt,
            "show": self._show_system_prompt,
        }

    async def _set_system_prompt(self, flags: Dict[str, str], args: List[str]) -> str:
        """
        Set the system prompt of a workspace
        If a workspace name is not provided, the active workspace is used
        """
        if len(args) == 0:
            return (
                "Please provide a workspace name and a system prompt. "
                "Use `codegate workspace system-prompt -w <workspace_name> <system_prompt>`"
            )

        workspace_name = flags.get("-w")
        if not workspace_name:
            active_workspace = await self.workspace_crud.get_active_workspace()
            workspace_name = active_workspace.name

        try:
            updated_worksapce = await self.workspace_crud.update_workspace_system_prompt(
                workspace_name, args
            )
        except crud.WorkspaceDoesNotExistError:
            return (
                f"Workspace system prompt not updated. Workspace `{workspace_name}` doesn't exist"
            )

        return f"Workspace `{updated_worksapce.name}` system prompt updated."

    async def _show_system_prompt(self, flags: Dict[str, str], args: List[str]) -> str:
        """
        Show the system prompt of a workspace
        If a workspace name is not provided, the active workspace is used
        """
        workspace_name = flags.get("-w")
        if not workspace_name:
            active_workspace = await self.workspace_crud.get_active_workspace()
            workspace_name = active_workspace.name

        try:
            workspace = await self.workspace_crud.get_workspace_by_name(workspace_name)
        except crud.WorkspaceDoesNotExistError:
            return f"Workspace `{workspace_name}` doesn't exist"

        return f"Workspace **{workspace.name}** system prompt:\n\n{workspace.system_prompt}."

    @property
    def help(self) -> str:
        return (
            "### CodeGate System Prompt\n"
            "Manage the system prompts of workspaces.\n\n"
            "*Note*: If you want to update the system prompt using files please go to the "
            "[dashboard](http://localhost:9090).\n\n"
            "**Usage**: `codegate system-prompt -w <workspace_name> <command>`\n\n"
            "*args*:\n"
            "- `workspace_name`: Optional workspace name. If not specified will use the "
            "active workspace\n\n"
            "Available commands:\n"
            "- `set`: Set the system prompt of the workspace\n"
            "  - *args*:\n"
            "    - `system_prompt`: The system prompt to set\n"
            "  - **Usage**: `codegate system-prompt -w <workspace_name> set <system_prompt>`\n"
        )
