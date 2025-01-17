from abc import ABC, abstractmethod
from typing import List

from pydantic import ValidationError

from codegate import __version__
from codegate.db.connection import AlreadyExistsError
from codegate.workspaces.crud import WorkspaceCrud


class CodegateCommand(ABC):
    @abstractmethod
    async def run(self, args: List[str]) -> str:
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
    def help(self) -> str:
        return (
            "### CodeGate Version\n\n"
            "Prints the version of CodeGate.\n\n"
            "**Usage**: `codegate version`\n\n"
            "*args*: None"
        )


class Workspace(CodegateCommand):

    def __init__(self):
        self.workspace_crud = WorkspaceCrud()
        self.commands = {
            "list": self._list_workspaces,
            "add": self._add_workspace,
            "activate": self._activate_workspace,
        }

    async def _list_workspaces(self, *args: List[str]) -> str:
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

    async def _add_workspace(self, args: List[str]) -> str:
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

    async def _activate_workspace(self, args: List[str]) -> str:
        """
        Activate a workspace
        """
        if args is None or len(args) == 0:
            return "Please provide a name. Use `codegate workspace activate workspace_name`"

        workspace_name = args[0]
        if not workspace_name:
            return "Please provide a name. Use `codegate workspace activate workspace_name`"

        was_activated = await self.workspace_crud.activate_workspace(workspace_name)
        if not was_activated:
            return (
                f"Workspace **{workspace_name}** does not exist or was already active. "
                f"Use `codegate workspace add {workspace_name}` to add it"
            )
        return f"Workspace **{workspace_name}** has been activated"

    async def run(self, args: List[str]) -> str:
        if not args:
            return "Please provide a command. Use `codegate workspace -h` to see available commands"
        command = args[0]
        command_to_execute = self.commands.get(command)
        if command_to_execute is not None:
            return await command_to_execute(args[1:])
        else:
            return "Command not found. Use `codegate workspace -h` to see available commands"

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
