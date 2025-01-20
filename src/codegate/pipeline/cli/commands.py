from abc import ABC, abstractmethod
from typing import List

from pydantic import ValidationError

from codegate import __version__
from codegate.db.connection import AlreadyExistsError
from codegate.workspaces import crud


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
            "### CodeGate Version\n"
            "Prints the version of CodeGate.\n\n"
            "*args*: None\n\n"
            "**Usage**: `codegate version`\n\n"
        )


class Workspace(CodegateCommand):

    def __init__(self):
        self.workspace_crud = crud.WorkspaceCrud()
        self.commands = {
            "list": self._list_workspaces,
            "add": self._add_workspace,
            "activate": self._activate_workspace,
            "system-prompt": self._add_system_prompt,
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
            return "Please provide a name. Use `codegate workspace add <workspace_name>`"

        new_workspace_name = args[0]
        if not new_workspace_name:
            return "Please provide a name. Use `codegate workspace add <workspace_name>`"

        try:
            _ = await self.workspace_crud.add_workspace(new_workspace_name)
        except ValidationError:
            return "Invalid workspace name: It should be alphanumeric and dashes"
        except AlreadyExistsError:
            return f"Workspace `{new_workspace_name}` already exists"
        except Exception:
            return "An error occurred while adding the workspace"

        return f"Workspace `{new_workspace_name}` has been added"

    async def _activate_workspace(self, args: List[str]) -> str:
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

    async def _add_system_prompt(self, args: List[str]):
        if len(args) < 2:
            return (
                "Please provide a workspace name and a system prompt. "
                "Use `codegate workspace system-prompt <workspace_name> <system_prompt>`"
            )

        workspace_name = args[0]
        sys_prompt_lst = args[1:]

        updated_worksapce = await self.workspace_crud.update_workspace_system_prompt(
            workspace_name, sys_prompt_lst
        )
        if not updated_worksapce:
            return (
                f"Workspace system prompt not updated. "
                f"Check if the workspace `{workspace_name}` exists"
            )
        return (
            f"Workspace `{updated_worksapce.name}` system prompt "
            f"updated to:\n```\n{updated_worksapce.system_prompt}\n```"
        )

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
            "### CodeGate Workspace\n"
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
            "- `system-prompt`: Modify the system-prompt of a workspace\n"
            "  - *args*:\n"
            "    - `workspace_name`\n"
            "    - `system_prompt`\n"
            "  - **Usage**: `codegate workspace system-prompt <workspace_name> <system_prompt>`\n"
        )
