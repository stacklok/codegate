import asyncio
import json
import re
from typing import Any, List, Optional, Tuple, Union

import structlog

from codegate.dashboard.request_models import (
    AlertConversation,
    ChatMessage,
    Conversation,
    PartialConversation,
    QuestionAnswer,
)
from codegate.db.queries import GetAlertsWithPromptAndOutputRow, GetPromptWithOutputsRow

logger = structlog.get_logger("codegate")


SYSTEM_PROMPTS = [
    "Given the following... please reply with a short summary that is 4-12 words in length, "
    "you should summarize what the user is asking for OR what the user is trying to accomplish. "
    "You should only respond with the summary, no additional text or explanation, "
    "you don't need ending punctuation.",
]

MEANINGFUL_MESSAGE_LENGTH = 50


async def _is_system_prompt(message: str) -> bool:
    """
    Check if the message is a system prompt.
    """
    for prompt in SYSTEM_PROMPTS:
        if prompt in message or message in prompt:
            return True
    return False


async def parse_request(request_str: str) -> Optional[str]:
    """
    Parse the request string from the pipeline and return the message.
    """
    try:
        request = json.loads(request_str)
    except Exception as e:
        logger.warning(f"Error parsing request: {request_str}. {e}")
        return None

    messages = []
    for message in request.get("messages", []):
        role = message.get("role")
        if not role == "user":
            continue
        content = message.get("content")

        message_str = ""
        if isinstance(content, str):
            message_str = content
        elif isinstance(content, list):
            for content_part in content:
                if isinstance(content_part, dict) and content_part.get("type") == "text":
                    message_str = content_part.get("text")

        if message_str and not await _is_system_prompt(message_str):
            messages.append(message_str)

    # We couldn't get anything from the messages, try the prompt
    if not messages:
        message_prompt = request.get("prompt", "")
        if message_prompt and not await _is_system_prompt(message_prompt):
            messages.append(message_prompt)

    # If still we don't have anything, return empty string
    if not messages:
        return None

    # Only respond with the latest message
    return messages[-1]


async def parse_output(output_str: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse the output string from the pipeline and return the message and chat_id.
    """
    try:
        if output_str is None:
            return None, None

        output = json.loads(output_str)
    except Exception as e:
        logger.warning(f"Error parsing output: {output_str}. {e}")
        return None, None

    def _parse_single_output(single_output: dict) -> Tuple[Any, Any]:
        single_chat_id = single_output.get("id")
        single_output_message = ""
        for choice in single_output.get("choices", []):
            if not isinstance(choice, dict):
                continue
            content_dict = choice.get("delta", {}) or choice.get("message", {})
            single_output_message += content_dict.get("content", "")
        return single_output_message, single_chat_id

    full_output_message = ""
    chat_id = None
    if isinstance(output, list):
        for output_chunk in output:
            output_message, output_chat_id = "", None
            if isinstance(output_chunk, dict):
                output_message, output_chat_id = _parse_single_output(output_chunk)
            elif isinstance(output_chunk, str):
                try:
                    output_decoded = json.loads(output_chunk)
                    output_message, output_chat_id = _parse_single_output(output_decoded)
                except Exception:
                    logger.error(f"Error reading chunk: {output_chunk}")
            else:
                logger.warning(
                    f"Could not handle output: {output_chunk}", out_type=type(output_chunk)
                )
            chat_id = chat_id or output_chat_id
            full_output_message += output_message
    elif isinstance(output, dict):
        full_output_message, chat_id = _parse_single_output(output)

    return full_output_message, chat_id


async def _get_question_answer(
    row: Union[GetPromptWithOutputsRow, GetAlertsWithPromptAndOutputRow]
) -> Tuple[Optional[QuestionAnswer], Optional[str]]:
    """
    Parse a row from the get_prompt_with_outputs query and return a PartialConversation

    The row contains the raw request and output strings from the pipeline.
    """
    async with asyncio.TaskGroup() as tg:
        if not row.request:
            return None, None
        if not row.output:
            return None, None
        request_task = tg.create_task(parse_request(row.request))
        output_task = tg.create_task(parse_output(row.output))

    request_msg_str = request_task.result()
    output_msg_str, chat_id = output_task.result()

    # If we couldn't parse the request or output, return None
    if not request_msg_str:
        return None, None

    request_message = ChatMessage(
        message=request_msg_str,
        timestamp=row.timestamp,
        message_id=row.id,
    )
    if output_msg_str:
        if row and row.output_timestamp and row.output_id:
            output_message = ChatMessage(
                message=output_msg_str,
                timestamp=row.output_timestamp,
                message_id=row.output_id,)
    else:
        output_message = None
        chat_id = row.id
    return QuestionAnswer(question=request_message, answer=output_message), chat_id


async def parse_get_prompt_with_output(
    row: GetPromptWithOutputsRow,
) -> Optional[PartialConversation]:
    """
    Parse a row from the get_prompt_with_outputs query and return a PartialConversation

    The row contains the raw request and output strings from the pipeline.
    """
    question_answer, chat_id = await _get_question_answer(row)
    if not question_answer or not chat_id:
        return None
    return PartialConversation(
        question_answer=question_answer,
        provider=row.provider,
        type=row.type,
        chat_id=chat_id,
        request_timestamp=row.timestamp,
    )


def _parse_question_answer_question(input_text: str) -> str:
    # given a string, detect if we have a pattern of "Context: xxx \n\nQuery: xxx" and strip it
    pattern = r'^Context:.*?\n\n\s*Query:\s*(.*)$'

    # Search using the regex pattern
    match = re.search(pattern, input_text, re.DOTALL)

    # If a match is found, return the captured group after "Query:"
    if match:
        input_text = match.group(1)
        return match.group(1)
    else:
        return input_text


def _parse_question_answer_answer(input_text: str) -> str:
    # need to remove blank spaces and newlines
    return input_text.strip()


async def match_conversations(
    partial_conversations: List[Optional[PartialConversation]],
) -> List[Conversation]:
    """
    Match partial conversations to form a complete conversation.
    """
    convers = {}
    for partial_conversation in partial_conversations:
        if not partial_conversation:
            continue

        # Group by chat_id
        if partial_conversation.chat_id not in convers:
            convers[partial_conversation.chat_id] = []
        convers[partial_conversation.chat_id].append(partial_conversation)

    # Sort by timestamp
    sorted_convers = {
        chat_id: sorted(conversations, key=lambda x: x.request_timestamp)
        for chat_id, conversations in convers.items()
    }
    # Create the conversation objects
    conversations = []
    for chat_id, sorted_convers in sorted_convers.items():
        questions_answers = []
        first_partial_conversation = None
        for partial_conversation in sorted_convers:
            #  check if we have an answer, otherwise do not add it
            if partial_conversation.question_answer.answer is not None:
                partial_conversation.question_answer.answer.message = _parse_question_answer_answer(
                    partial_conversation.question_answer.answer.message)
                if partial_conversation.question_answer.answer.message:
                    first_partial_conversation = partial_conversation
                    partial_conversation.question_answer.question.message = \
                        _parse_question_answer_question(
                            partial_conversation.question_answer.question.message)
                    questions_answers.append(partial_conversation.question_answer)

        # only add conversation if we have some answers
        if len(questions_answers) > 0 and first_partial_conversation is not None:
            # if type is fim, we will only show the meaningful answers
            can_show = True
            if first_partial_conversation.type == 'fim':
                # only show it if length of the answer is meaningful
                can_show = False
                for qa in questions_answers:
                    if len(qa.answer.message) > MEANINGFUL_MESSAGE_LENGTH:
                        can_show = True
                        break
            if can_show:
                conversations.append(
                    Conversation(
                        question_answers=questions_answers,
                        provider=first_partial_conversation.provider,
                        type=first_partial_conversation.type,
                        chat_id=chat_id,
                        conversation_timestamp=sorted_convers[0].request_timestamp,
                    )
                )

    return conversations


async def parse_messages_in_conversations(
    prompts_outputs: List[GetPromptWithOutputsRow],
) -> List[Conversation]:
    """
    Get all the messages from the database and return them as a list of conversations.
    """

    # Parse the prompts and outputs in parallel
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(parse_get_prompt_with_output(row)) for row in prompts_outputs]
    partial_conversations = [task.result() for task in tasks]

    conversations = await match_conversations(partial_conversations)
    return conversations


async def parse_row_alert_conversation(
    row: GetAlertsWithPromptAndOutputRow,
) -> Optional[AlertConversation]:
    """
    Parse a row from the get_alerts_with_prompt_and_output query and return a Conversation

    The row contains the raw request and output strings from the pipeline.
    """
    question_answer, chat_id = await _get_question_answer(row)
    if not question_answer or not chat_id:
        return None

    if row and row.type:
        conversation = Conversation(
            question_answers=[question_answer],
            provider=row.provider,
            type=row.type,
            chat_id=chat_id or "chat-id-not-found",
            conversation_timestamp=row.timestamp,
        )
    code_snippet = json.loads(row.code_snippet) if row.code_snippet else None
    trigger_string = None
    if row.trigger_string:
        try:
            trigger_string = json.loads(row.trigger_string)
        except Exception:
            trigger_string = row.trigger_string

    return AlertConversation(
        conversation=conversation,
        alert_id=row.id,
        code_snippet=code_snippet,
        trigger_string=trigger_string,
        trigger_type=row.trigger_type,
        trigger_category=row.trigger_category,
        timestamp=row.timestamp,
    )


async def parse_get_alert_conversation(
    alerts_conversations: List[GetAlertsWithPromptAndOutputRow],
) -> List[AlertConversation | None]:
    """
    Parse a list of rows from the get_alerts_with_prompt_and_output query and return a list of
    AlertConversation

    The rows contain the raw request and output strings from the pipeline.
    """
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(parse_row_alert_conversation(row)) for row in alerts_conversations]
    return [task.result() for task in tasks if task.result() is not None]
