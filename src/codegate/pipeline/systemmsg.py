import json
from typing import Optional

from codegate.pipeline.base import PipelineContext
from codegate.types.common import ChatCompletionRequest, ChatCompletionSystemMessage


def get_existing_system_message(request: ChatCompletionRequest) -> Optional[dict]:
    """
    Retrieves the existing system message from the completion request.

    Args:
        request: The original completion request.

    Returns:
        The existing system message if found, otherwise None.
    """

    for message in request.get("messages", []):
        if message["role"] == "system":
            return message
    return None


def add_or_update_system_message(
    request: ChatCompletionRequest,
    system_message: ChatCompletionSystemMessage,
    context: PipelineContext,
) -> ChatCompletionRequest:
    """
    Adds or updates the system message in the completion request.

    Args:
        request: The original completion request.
        system_message: The system message to add or update.
        context: The pipeline context for adding alerts.

    Returns:
        The updated completion request.
    """
    new_request = request.copy()

    if "messages" not in new_request:
        new_request["messages"] = []

    request_system_message = get_existing_system_message(new_request)

    if request_system_message is None:
        # Add new system message
        context.add_alert("add-system-message", trigger_string=json.dumps(system_message))
        new_request["messages"].insert(0, system_message)
    else:
        # Handle both string and list content types (needed for Cline (sends list)
        existing_content = request_system_message["content"]
        new_content = system_message["content"]

        # Convert list to string if necessary (needed for Cline (sends list)
        if isinstance(existing_content, list):
            existing_content = "\n".join(str(item) for item in existing_content)
        if isinstance(new_content, list):
            new_content = "\n".join(str(item) for item in new_content)

        # Update existing system message
        updated_content = existing_content + "\n\n" + new_content
        context.add_alert("update-system-message", trigger_string=updated_content)
        request_system_message["content"] = updated_content

    return new_request
