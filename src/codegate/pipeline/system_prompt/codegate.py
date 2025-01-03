import json

from litellm import ChatCompletionRequest, ChatCompletionSystemMessage

from codegate.pipeline.base import (
    PipelineContext,
    PipelineResult,
    PipelineStep,
)


class SystemPrompt(PipelineStep):
    """
    Pipeline step that adds a system prompt to the completion request when it detects
    the word "codegate" in the user message.
    """

    def __init__(self, system_prompt: str):
        self._system_message = ChatCompletionSystemMessage(content=system_prompt, role="system")

    @property
    def name(self) -> str:
        """
        Returns the name of this pipeline step.
        """
        return "system-prompt"

    async def process(
        self, request: ChatCompletionRequest, context: PipelineContext
    ) -> PipelineResult:
        """
        Add system prompt if not present, otherwise prepend codegate system prompt
        to the existing system prompt
        """

        # Nothing to do if no secrets or bad_packages are found
        if not (context.secrets_found or context.bad_packages_found):
            return PipelineResult(request=request, context=context)

        new_request = request.copy()

        if "messages" not in new_request:
            new_request["messages"] = []

        request_system_message = None
        for message in new_request["messages"]:
            if message["role"] == "system":
                request_system_message = message

        if request_system_message is None:
            # Add system message
            context.add_alert(self.name, trigger_string=json.dumps(self._system_message))
            new_request["messages"].insert(0, self._system_message)
        # Addded Logic for Cline, which sends a list of strings
        elif (
            "content" not in request_system_message
            or not isinstance(request_system_message["content"], str)
            or "codegate" not in request_system_message["content"].lower()
        ):
            # Prepend to the system message
            original_content = request_system_message.get("content", "")
            if not isinstance(original_content, str):
                original_content = json.dumps(original_content)
            prepended_message = (
                self._system_message["content"]
                + "\n Here are additional instructions. \n "
                + original_content
            )
            context.add_alert(self.name, trigger_string=prepended_message)
            request_system_message["content"] = prepended_message

        return PipelineResult(request=new_request, context=context)
