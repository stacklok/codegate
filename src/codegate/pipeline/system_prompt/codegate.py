from typing import Optional

from litellm import ChatCompletionRequest, ChatCompletionSystemMessage

from codegate.pipeline.base import (
    PipelineContext,
    PipelineResult,
    PipelineStep,
)
from codegate.workspaces.crud import WorkspaceCrud


class SystemPrompt(PipelineStep):
    """
    Pipeline step that adds a system prompt to the completion request when it detects
    the word "codegate" in the user message.
    """

    def __init__(self, system_prompt: str):
        self.codegate_system_prompt = system_prompt

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
        wrksp_custom_instr: str,
        req_sys_prompt: Optional[str],
        should_add_codegate_sys_prompt: bool,
    ) -> ChatCompletionSystemMessage:

        def _start_or_append(existing_prompt: str, new_prompt: str) -> str:
            if existing_prompt:
                return existing_prompt + "\n\nHere are additional instructions:\n\n" + new_prompt
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

        new_request = request.copy()

        if "messages" not in new_request:
            new_request["messages"] = []

        request_system_message = {}
        for message in new_request["messages"]:
            if message["role"] == "system":
                request_system_message = message
        req_sys_prompt = request_system_message.get("content")

        system_prompt = await self._construct_system_prompt(
            wrksp_custom_instructions, req_sys_prompt, should_add_codegate_sys_prompt
        )
        context.add_alert(self.name, trigger_string=system_prompt)
        if not request_system_message:
            # Insert the system prompt at the beginning of the messages
            sytem_message = ChatCompletionSystemMessage(content=system_prompt, role="system")
            new_request["messages"].insert(0, sytem_message)
        else:
            # Update the existing system prompt
            request_system_message["content"] = system_prompt

        # check if we are in kodu
        if "</kodu_action>" in new_request.get("stop", []):
            # Collect messages from the assistant matching the criteria
            relevant_contents = [
                message["content"]
                for message in new_request["messages"]
                if message["role"] == "assistant"
                and (
                    message["content"].startswith("**Warning")
                    or message["content"].startswith("<thinking>")
                )
            ]

            if relevant_contents:
                # Combine the contents into a single message
                summarized_content = (
                    "<attempt_completion><result>"
                    + "".join(relevant_contents)
                    + "</result></attempt_completion>"
                )

                # Replace the messages with a single summarized message
                new_request["messages"] = [
                    message
                    for message in new_request["messages"]
                    if not (
                        message["role"] == "assistant"
                        and (
                            message["content"].startswith("**Warning")
                            or message["content"].startswith("<thinking>")
                        )
                    )
                ]

                # Append the summarized message to the messages
                new_request["messages"].append({"role": "assistant", "content": summarized_content})

        return PipelineResult(request=new_request, context=context)
