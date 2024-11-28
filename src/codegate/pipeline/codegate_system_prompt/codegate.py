from typing import Optional

from litellm import ChatCompletionRequest, ChatCompletionSystemMessage

from codegate.pipeline.base import (
    PipelineContext,
    PipelineResult,
    PipelineStep,
)


class CodegateSystemPrompt(PipelineStep):
    """
    Pipeline step that adds a system prompt to the completion request when it detects
    the word "codegate" in the user message.
    """

    def __init__(self, system_prompt_message: Optional[str] = None):
        self._system_message = ChatCompletionSystemMessage(
            content=system_prompt_message, role="system"
        )

    @property
    def name(self) -> str:
        """
        Returns the name of this pipeline step.
        """
        return "codegate-system-prompt"

    async def process(
        self, request: ChatCompletionRequest, context: PipelineContext
    ) -> PipelineResult:
        """
        Process the completion request and add a system prompt if the user message contains
        the word "codegate".
        """
        # no prompt configured
        if not self._system_message["content"]:
            return PipelineResult(request=request)

        last_user_message = self.get_last_user_message(request)

        if last_user_message is not None:
            last_user_message_str, last_user_idx = last_user_message
            if "codegate" in last_user_message_str.lower():
                # Add a system prompt to the completion request
                new_request = request.copy()
                new_request["messages"].insert(last_user_idx, self._system_message)
                return PipelineResult(
                    request=new_request,
                )

        # Fall through
        return PipelineResult(request=request)
