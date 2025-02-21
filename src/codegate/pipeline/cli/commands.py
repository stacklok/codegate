from abc import ABC, abstractmethod
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

from cachetools import TTLCache
from pydantic import ValidationError

from codegate import __version__
from codegate.db.connection import AlreadyExistsError
from codegate.workspaces import crud


class NoFlagValueError(Exception):
    pass


class NoSubcommandError(Exception):
    pass


# 1 second cache. 1 second is to be short enough to not affect UX but long enough to
# reply the same to concurrent requests. Needed for Copilot.
command_cache = TTLCache(maxsize=10, ttl=1)


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

    async def _get_full_command(self, args: List[str]) -> str:
        """
        Get the full command string with the command name and args.
        """
        joined_args = " ".join(args)
        return f"{self.command_name} {joined_args}"

    async def _record_in_cache(self, args: List[str], cmd_out: str) -> None:
        """
        Record the command in the cache.
        """
        full_command = await self._get_full_command(args)
        command_cache[full_command] = cmd_out

    async def _cache_lookup(self, args: List[str]) -> Optional[str]:
        """
        Look up the command in the cache. If the command was executed less than 1 second ago,
        return the cached output.
        """
        full_command = await self._get_full_command(args)
        cmd_out = command_cache.get(full_command)
        return cmd_out

    async def exec(self, args: List[str]) -> str:
        """
        Execute the command and cache the output. The cache is invalidated after 1 second.

        1. Check if the command is help. If it is, return the help text.
        2. Check if the command is in the cache. If it is, return the cached output.
        3. Run the command and cache the output.
        4. Return the output.
        """
        if args and args[0] == "-h":
            return self.help
        cached_out = await self._cache_lookup(args)
        if cached_out:
            return cached_out
        cmd_out = await self.run(args)
        await self._record_in_cache(args, cmd_out)
        return cmd_out


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
            "archive": self._archive_workspace,
            "rename": self._rename_workspace,
            "list-archived": self._list_archived_workspaces,
            "restore": self._restore_workspace,
            "delete-archived": self._delete_archived_workspace,
        }

    async def _list_workspaces(self, flags: Dict[str, str], args: List[str]) -> str:
        """
        List all workspaces
        """
        workspaces = await self.workspace_crud.get_workspaces()
        respond_str = ""
        for workspace in workspaces:
            respond_str += f"- {workspace.name}"
            if workspace.session_id:
                respond_str += " **(active)**"
            respond_str += "\n"
        return respond_str

    async def _add_workspace(self, flags: Dict[str, str], args: List[str]) -> str:
        """
        Add a workspace
        """
        if args is None or len(args) == 0:
            return "Please provide a name. Use `codegate workspace add <workspace_name>`"

        new_workspace_name = args[0]
        if not new_workspace_name:
            return "Please provide a name. Use `codegate workspace add <workspace_name>`"

        try:
            ws = await self.workspace_crud.add_workspace(new_workspace_name)
        except ValidationError:
            return "Invalid workspace name: It should be alphanumeric with hyphens or underscores"
        except AlreadyExistsError:
            return f"Workspace **{new_workspace_name}** already exists"
        except crud.WorkspaceCrudError:
            return "An error occurred while adding the workspace"
        except Exception:
            return "An error occurred while adding the workspace"

        return f"Workspace **{ws.name}** has been added"

    async def _rename_workspace(self, flags: Dict[str, str], args: List[str]) -> str:
        """
        Rename a workspace
        """
        if args is None or len(args) < 2:
            return (
                "Please provide a name and a new name. "
                "Use `codegate workspace rename <workspace_name> <new_workspace_name>`"
            )

        old_workspace_name = args[0]
        new_workspace_name = args[1]
        if not old_workspace_name or not new_workspace_name:
            return (
                "Please provide a name and a new name. "
                "Use `codegate workspace rename <workspace_name> <new_workspace_name>`"
            )

        try:
            await self.workspace_crud.update_workspace(old_workspace_name, new_workspace_name)
        except crud.WorkspaceDoesNotExistError:
            return f"Workspace **{old_workspace_name}** does not exist"
        except AlreadyExistsError:
            return f"Workspace **{new_workspace_name}** already exists"
        except crud.WorkspaceCrudError:
            return "An error occurred while renaming the workspace"
        except Exception:
            return "An error occurred while renaming the workspace"

        return f"Workspace **{old_workspace_name}** has been renamed to **{new_workspace_name}**"

    async def _activate_workspace(self, flags: Dict[str, str], args: List[str]) -> str:
        """
        Activate a workspace
        """
        if args is None or len(args) == 0:
            return "Please provide a name. Use `codegate workspace activate <workspace_name>`"

        workspace_name = args[0]
        if not workspace_name:
            return "Please provide a name. Use `codegate workspace activate <workspace_name>`"

        try:
            await self.workspace_crud.activate_workspace(workspace_name)
        except crud.WorkspaceAlreadyActiveError:
            return f"Workspace **{workspace_name}** is already active"
        except crud.WorkspaceDoesNotExistError:
            return f"Workspace **{workspace_name}** does not exist"
        except Exception:
            return "An error occurred while activating the workspace"
        return f"Workspace **{workspace_name}** has been activated"

    async def _archive_workspace(self, flags: Dict[str, str], args: List[str]) -> str:
        """
        Remove a workspace
        """
        if args is None or len(args) == 0:
            return "Please provide a name. Use `codegate workspace archive <workspace_name>`"

        workspace_name = args[0]
        if not workspace_name:
            return "Please provide a name. Use `codegate workspace archive <workspace_name>`"

        try:
            await self.workspace_crud.soft_delete_workspace(workspace_name)
        except crud.WorkspaceDoesNotExistError:
            return f"Workspace **{workspace_name}** does not exist"
        except crud.WorkspaceCrudError as e:
            return str(e)
        except Exception:
            return "An error occurred while archiving the workspace"
        return f"Workspace **{workspace_name}** has been archived"

    async def _list_archived_workspaces(self, flags: Dict[str, str], args: List[str]) -> str:
        """
        List all archived workspaces
        """
        workspaces = await self.workspace_crud.get_archived_workspaces()
        respond_str = ""
        for workspace in workspaces:
            respond_str += f"- {workspace.name}\n"
        return respond_str

    async def _restore_workspace(self, flags: Dict[str, str], args: List[str]) -> str:
        """
        Restore an archived workspace
        """
        if args is None or len(args) == 0:
            return "Please provide a name. Use `codegate workspace restore <workspace_name>`"

        workspace_name = args[0]
        if not workspace_name:
            return "Please provide a name. Use `codegate workspace restore <workspace_name>`"

        try:
            await self.workspace_crud.recover_workspace(workspace_name)
        except crud.WorkspaceDoesNotExistError:
            return f"Workspace **{workspace_name}** does not exist"
        except crud.WorkspaceCrudError as e:
            return str(e)
        except Exception:
            return "An error occurred while restoring the workspace"
        return f"Workspace **{workspace_name}** has been restored"

    async def _delete_archived_workspace(self, flags: Dict[str, str], args: List[str]) -> str:
        """
        Hard delete an archived workspace
        """
        if args is None or len(args) == 0:
            return (
                "Please provide a name. Use `codegate workspace delete-archived <workspace_name>`"
            )

        workspace_name = args[0]
        if not workspace_name:
            return (
                "Please provide a name. Use `codegate workspace delete-archived <workspace_name>`"
            )

        try:
            await self.workspace_crud.hard_delete_workspace(workspace_name)
        except crud.WorkspaceDoesNotExistError:
            return f"Workspace **{workspace_name}** does not exist"
        except crud.WorkspaceCrudError as e:
            return str(e)
        except Exception:
            return "An error occurred while deleting the workspace"
        return f"Workspace **{workspace_name}** has been deleted"

    @property
    def help(self) -> str:
        return (
            "### CodeGate Workspace\n\n"
            "Manage workspaces.\n\n"
            "**Usage**: `codegate workspace <command> [args]`\n\n"
            "Available commands:\n"
            "- `list`: List all workspaces\n"
            "  - *args*: None\n"
            "  - **Usage**: `codegate workspace list`\n"
            "- `add`: Add a workspace\n"
            "  - *args*:\n"
            "    - `workspace_name`\n"
            "  - **Usage**: `codegate workspace add <workspace_name>`\n"
            "- `activate`: Activate a workspace\n"
            "  - *args*:\n"
            "    - `workspace_name`\n"
            "  - **Usage**: `codegate workspace activate <workspace_name>`\n"
            "- `archive`: Archive a workspace\n"
            "  - *args*:\n"
            "    - `workspace_name`\n"
            "  - **Usage**: `codegate workspace archive <workspace_name>`\n"
            "- `rename`: Rename a workspace\n"
            "  - *args*:\n"
            "    - `workspace_name`\n"
            "    - `new_workspace_name`\n"
            "  - **Usage**: `codegate workspace rename <workspace_name> <new_workspace_name>`\n"
            "- `list-archived`: List all archived workspaces\n"
            "  - *args*: None\n"
            "  - **Usage**: `codegate workspace list-archived`\n"
            "- `restore`: Restore an archived workspace\n"
            "  - *args*:\n"
            "    - `workspace_name`\n"
            "  - **Usage**: `codegate workspace restore <workspace_name>`\n"
            "- `delete-archived`: Hard delete an archived workspace\n"
            "  - *args*:\n"
            "    - `workspace_name`\n"
            "  - **Usage**: `codegate workspace delete-archived <workspace_name>`\n"
        )


class CustomInstructions(CodegateCommandSubcommand):
    def __init__(self):
        self.workspace_crud = crud.WorkspaceCrud()

    @property
    def command_name(self) -> str:
        return "custom-instructions"

    @property
    def flags(self) -> List[str]:
        """
        Flags for the custom-instructions command.
        -w: Workspace name
        """
        return ["-w"]

    @property
    def subcommands(self) -> Dict[str, Callable[[List[str]], Awaitable[str]]]:
        return {
            "set": self._set_custom_instructions,
            "show": self._show_custom_instructions,
            "reset": self._reset_custom_instructions,
        }

    async def _set_custom_instructions(self, flags: Dict[str, str], args: List[str]) -> str:
        """
        Set the custom instructions of a workspace
        If a workspace name is not provided, the active workspace is used
        """
        if len(args) == 0:
            return (
                "Please provide a workspace name and custom instructions to use. "
                "Use `codegate workspace custom-instructions -w <workspace_name> <instructions>`"
            )

        workspace_name = flags.get("-w")
        if not workspace_name:
            active_workspace = await self.workspace_crud.get_active_workspace()
            workspace_name = active_workspace.name

        try:
            updated_worksapce = await self.workspace_crud.update_workspace_custom_instructions(
                workspace_name, args
            )
        except crud.WorkspaceDoesNotExistError:
            return (
                f"Workspace custom instructions not updated. "
                f"Workspace **{workspace_name}** doesn't exist"
            )

        return f"Workspace **{updated_worksapce.name}** custom instructions updated."

    async def _show_custom_instructions(self, flags: Dict[str, str], args: List[str]) -> str:
        """
        Show the custom instructions of a workspace
        If a workspace name is not provided, the active workspace is used
        """
        workspace_name = flags.get("-w")
        if not workspace_name:
            active_workspace = await self.workspace_crud.get_active_workspace()
            workspace_name = active_workspace.name

        try:
            workspace = await self.workspace_crud.get_workspace_by_name(workspace_name)
        except crud.WorkspaceDoesNotExistError:
            return f"Workspace **{workspace_name}** doesn't exist"

        sysprompt = workspace.custom_instructions
        if not sysprompt:
            return f"Workspace **{workspace.name}** custom instructions is unset."

        return f"Workspace **{workspace.name}** custom instructions:\n\n{sysprompt}."

    async def _reset_custom_instructions(self, flags: Dict[str, str], args: List[str]) -> str:
        """
        Reset the custom instructions of a workspace
        If a workspace name is not provided, the active workspace is used
        """
        workspace_name = flags.get("-w")
        if not workspace_name:
            active_workspace = await self.workspace_crud.get_active_workspace()
            workspace_name = active_workspace.name

        try:
            updated_worksapce = await self.workspace_crud.update_workspace_custom_instructions(
                workspace_name, [""]
            )
        except crud.WorkspaceDoesNotExistError:
            return f"Workspace **{workspace_name}** doesn't exist"

        return f"Workspace **{updated_worksapce.name}** custom instructions reset."

    @property
    def help(self) -> str:
        return (
            "### CodeGate Custom Instructions\n"
            "Manage the custom instructionss of workspaces.\n\n"
            "*Note*: If you want to update the custom instructions using files please go to the "
            "[dashboard](http://localhost:9090).\n\n"
            "**Usage**: `codegate custom-instructions -w <workspace_name> <command> [args]`\n\n"
            "*args*:\n"
            "- `workspace_name`: Optional workspace name. If not specified will use the "
            "active workspace\n\n"
            "Available commands:\n"
            "- `set`: Set the custom instructions of the workspace\n"
            "  - *args*:\n"
            "    - `instructions`: The custom instructions to set\n"
            "  - **Usage**: `codegate custom-instructions -w <workspace_name> set <instructions>`\n"
            "- `show`: Show custom instructions of the workspace\n"
            "  - *args*: None\n"
            "  - **Usage**: `codegate custom-instructions -w <workspace_name> show`\n"
            "- `reset`: Reset the custom instructions of the workspace\n"
            "  - *args*: None\n"
            "  - **Usage**: `codegate custom-instructions -w <workspace_name> reset`\n"
        )
