from typing import Optional

from codegate.clients.clients import ClientType
from codegate.pipeline.base import (
    PipelineContext,
    PipelineResult,
    PipelineStep,
)
from codegate.workspaces.crud import WorkspaceCrud
from codegate.types.common import ChatCompletionRequest, ChatCompletionSystemMessage
from codegate.types.anthropic import (
    ChatCompletionRequest as CodegateChatCompletionRequest
)
from codegate.types.openai import (
    ChatCompletionRequest as CodegateOpenaiChatCompletionRequest
)


class SystemPrompt(PipelineStep):
    """
    Pipeline step that adds a system prompt to the completion request when it detects
    the word "codegate" in the user message.
    """

    def __init__(self, system_prompt: str, client_prompts: dict[str]):
        self.codegate_system_prompt = system_prompt
        self.client_prompts = client_prompts

    @property
    def name(self) -> str:
        """
        Returns the name of this pipeline step.
        """
        return "system-prompt"

    async def _get_workspace_custom_instructions(self) -> str:
        wksp_crud = WorkspaceCrud()
        workspace = await wksp_crud.get_active_workspace()
        if not workspace:
            return ""

        return workspace.custom_instructions

    async def _construct_system_prompt(
        self,
        client: ClientType,
        wrksp_custom_instr: str,
        req_sys_prompt: Optional[str],
        should_add_codegate_sys_prompt: bool,
    ) -> str:

        def _start_or_append(existing_prompt: str, new_prompt: str) -> str:
            if existing_prompt:
                return f"{existing_prompt}\n\nHere are additional instructions:\n\n{new_prompt}"
            return new_prompt

        system_prompt = ""
        # Add codegate system prompt if secrets or bad packages are found at the beginning
        if should_add_codegate_sys_prompt:
            system_prompt = _start_or_append(system_prompt, self.codegate_system_prompt)

        # Add workspace system prompt if present
        if wrksp_custom_instr:
            system_prompt = _start_or_append(system_prompt, wrksp_custom_instr)

        # Add request system prompt if present
        if req_sys_prompt and "codegate" not in req_sys_prompt.lower():
            system_prompt = _start_or_append(system_prompt, req_sys_prompt)

        # Add per client system prompt
        if client and client.value in self.client_prompts:
            system_prompt = _start_or_append(system_prompt, self.client_prompts[client.value])

        return system_prompt

    async def _should_add_codegate_system_prompt(self, context: PipelineContext) -> bool:
        return context.secrets_found or context.bad_packages_found

    async def process(
        self, request: ChatCompletionRequest, context: PipelineContext
    ) -> PipelineResult:
        """
        Add system prompt if not present, otherwise prepend codegate system prompt
        to the existing system prompt
        """

        wrksp_custom_instructions = await self._get_workspace_custom_instructions()
        should_add_codegate_sys_prompt = await self._should_add_codegate_system_prompt(context)

        # Nothing to do if no secrets or bad_packages are found and we don't have a workspace
        # system prompt
        if not should_add_codegate_sys_prompt and not wrksp_custom_instructions:
            return PipelineResult(request=request, context=context)

        request_system_message = {}
        req_sys_prompt = ""
        for sysprompt in request.get_system_prompt():
            req_sys_prompt = sysprompt

        system_prompt = await self._construct_system_prompt(
            context.client,
            wrksp_custom_instructions,
            req_sys_prompt,
            should_add_codegate_sys_prompt,
        )
        context.add_alert(self.name, trigger_string=system_prompt)
        request.set_system_prompt(system_prompt)

        return PipelineResult(request=request, context=context)
