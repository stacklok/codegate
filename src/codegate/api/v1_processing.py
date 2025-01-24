import asyncio
import json
import re
from collections import defaultdict
from typing import AsyncGenerator, List, Optional, Union

import requests
import structlog

from codegate.api.v1_models import (
    AlertConversation,
    ChatMessage,
    Conversation,
    PartialQuestionAnswer,
    PartialQuestions,
    QuestionAnswer,
)
from codegate.db.connection import alert_queue
from codegate.db.models import GetAlertsWithPromptAndOutputRow, GetPromptWithOutputsRow

logger = structlog.get_logger("codegate")


SYSTEM_PROMPTS = [
    "Given the following... please reply with a short summary that is 4-12 words in length, "
    "you should summarize what the user is asking for OR what the user is trying to accomplish. "
    "You should only respond with the summary, no additional text or explanation, "
    "you don't need ending punctuation.",
]


def fetch_latest_version() -> str:
    url = "https://api.github.com/repos/stacklok/codegate/releases/latest"
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    response = requests.get(url, headers=headers, timeout=5)
    response.raise_for_status()
    data = response.json()
    return data.get("tag_name", "unknown")


async def generate_sse_events() -> AsyncGenerator[str, None]:
    """
    SSE generator from queue
    """
    while True:
        message = await alert_queue.get()
        yield f"data: {message}\n\n"


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
    return messages


async def parse_output(output_str: str) -> Optional[str]:
    """
    Parse the output string from the pipeline and return the message.
    """
    try:
        if output_str is None:
            return None

        output = json.loads(output_str)
    except Exception as e:
        logger.warning(f"Error parsing output: {output_str}. {e}")
        return None

    def _parse_single_output(single_output: dict) -> str:
        single_output_message = ""
        for choice in single_output.get("choices", []):
            if not isinstance(choice, dict):
                continue
            content_dict = choice.get("delta", {}) or choice.get("message", {})
            single_output_message += content_dict.get("content", "")
        return single_output_message

    full_output_message = ""
    if isinstance(output, list):
        for output_chunk in output:
            output_message = ""
            if isinstance(output_chunk, dict):
                output_message = _parse_single_output(output_chunk)
            elif isinstance(output_chunk, str):
                try:
                    output_decoded = json.loads(output_chunk)
                    output_message = _parse_single_output(output_decoded)
                except Exception:
                    logger.error(f"Error reading chunk: {output_chunk}")
            else:
                logger.warning(
                    f"Could not handle output: {output_chunk}", out_type=type(output_chunk)
                )
            full_output_message += output_message
    elif isinstance(output, dict):
        full_output_message = _parse_single_output(output)

    return full_output_message


async def _get_question_answer(
    row: Union[GetPromptWithOutputsRow, GetAlertsWithPromptAndOutputRow]
) -> Optional[PartialQuestionAnswer]:
    """
    Parse a row from the get_prompt_with_outputs query and return a PartialConversation

    The row contains the raw request and output strings from the pipeline.
    """
    async with asyncio.TaskGroup() as tg:
        request_task = tg.create_task(parse_request(row.request))
        output_task = tg.create_task(parse_output(row.output))

    request_user_msgs = request_task.result()
    output_msg_str = output_task.result()

    # If we couldn't parse the request, return None
    if not request_user_msgs:
        return None

    request_message = PartialQuestions(
        messages=request_user_msgs,
        timestamp=row.timestamp,
        message_id=row.id,
        provider=row.provider,
        type=row.type,
    )
    if output_msg_str:
        output_message = ChatMessage(
            message=output_msg_str,
            timestamp=row.output_timestamp,
            message_id=row.output_id,
        )
    else:
        output_message = None
    return PartialQuestionAnswer(partial_questions=request_message, answer=output_message)


def parse_question_answer(input_text: str) -> str:
    # given a string, detect if we have a pattern of "Context: xxx \n\nQuery: xxx" and strip it
    pattern = r"^Context:.*?\n\n\s*Query:\s*(.*)$"

    # Search using the regex pattern
    match = re.search(pattern, input_text, re.DOTALL)

    # If a match is found, return the captured group after "Query:"
    if match:
        return match.group(1)
    else:
        return input_text


def _group_partial_messages(pq_list: List[PartialQuestions]) -> List[List[PartialQuestions]]:
    """
    A PartialQuestion is an object that contains several user messages provided from a
    chat conversation. Example:
    - PartialQuestion(messages=["Hello"], timestamp=2022-01-01T00:00:00Z)
    - PartialQuestion(messages=["Hello", "How are you?"], timestamp=2022-01-01T00:00:01Z)
    In the above example both PartialQuestions are part of the same conversation and should be
    matched together.
    Group PartialQuestions objects such that:
      - If one PartialQuestion (pq) is a subset of another pq's messages, group them together.
      - If multiple subsets exist for the same superset, choose only the one
        closest in timestamp to the superset.
      - Leave any unpaired pq by itself.
      - Finally, sort the resulting groups by the earliest timestamp in each group.
    """
    # 1) Sort by length of messages descending (largest/most-complete first),
    #    then by timestamp ascending for stable processing.
    pq_list_sorted = sorted(pq_list, key=lambda x: (-len(x.messages), x.timestamp))

    used = set()
    groups = []

    # 2) Iterate in order of "largest messages first"
    for sup in pq_list_sorted:
        if sup.message_id in used:
            continue  # Already grouped

        # Find all potential subsets of 'sup' that are not yet used
        # (If sup's messages == sub's messages, that also counts, because sub ⊆ sup)
        possible_subsets = []
        for sub in pq_list_sorted:
            if sub.message_id == sup.message_id:
                continue
            if sub.message_id in used:
                continue
            if (
                set(sub.messages).issubset(set(sup.messages))
                and sub.provider == sup.provider
                and set(sub.messages) != set(sup.messages)
            ):
                possible_subsets.append(sub)

        # 3) If there are no subsets, this sup stands alone
        if not possible_subsets:
            groups.append([sup])
            used.add(sup.message_id)
        else:
            # 4) Group subsets by messages to discard duplicates e.g.: 2 subsets with single 'hello'
            subs_group_by_messages = defaultdict(list)
            for q in possible_subsets:
                subs_group_by_messages[tuple(q.messages)].append(q)

            new_group = [sup]
            used.add(sup.message_id)
            for subs_same_message in subs_group_by_messages.values():
                # If more than one pick the one subset closest in time to sup
                closest_subset = min(
                    subs_same_message, key=lambda s: abs(s.timestamp - sup.timestamp)
                )
                new_group.append(closest_subset)
                used.add(closest_subset.message_id)
            groups.append(new_group)

    # 5) Sort the groups by the earliest timestamp within each group
    groups.sort(key=lambda g: min(pq.timestamp for pq in g))
    return groups


def _get_question_answer_from_partial(
    partial_question_answer: PartialQuestionAnswer,
) -> QuestionAnswer:
    """
    Get a QuestionAnswer object from a PartialQuestionAnswer object.
    """
    # Get the last user message as the question
    question = ChatMessage(
        message=partial_question_answer.partial_questions.messages[-1],
        timestamp=partial_question_answer.partial_questions.timestamp,
        message_id=partial_question_answer.partial_questions.message_id,
    )

    return QuestionAnswer(question=question, answer=partial_question_answer.answer)


async def match_conversations(
    partial_question_answers: List[Optional[PartialQuestionAnswer]],
) -> List[Conversation]:
    """
    Match partial conversations to form a complete conversation.
    """
    valid_partial_qas = [
        partial_qas for partial_qas in partial_question_answers if partial_qas is not None
    ]
    grouped_partial_questions = _group_partial_messages(
        [partial_qs_a.partial_questions for partial_qs_a in valid_partial_qas]
    )

    # Create the conversation objects
    conversations = []
    for group in grouped_partial_questions:
        questions_answers = []
        first_partial_qa = None
        for partial_question in sorted(group, key=lambda x: x.timestamp):
            # Partial questions don't contain the answer, so we need to find the corresponding
            selected_partial_qa = None
            for partial_qa in valid_partial_qas:
                if partial_question.message_id == partial_qa.partial_questions.message_id:
                    selected_partial_qa = partial_qa
                    break

            #  check if we have an answer, otherwise do not add it
            if selected_partial_qa.answer is not None:
                # if we don't have a first question, set it
                first_partial_qa = first_partial_qa or selected_partial_qa
                question_answer = _get_question_answer_from_partial(selected_partial_qa)
                question_answer.question.message = parse_question_answer(
                    question_answer.question.message
                )
                questions_answers.append(question_answer)

        # only add conversation if we have some answers
        if len(questions_answers) > 0 and first_partial_qa is not None:
            conversations.append(
                Conversation(
                    question_answers=questions_answers,
                    provider=first_partial_qa.partial_questions.provider,
                    type=first_partial_qa.partial_questions.type,
                    chat_id=first_partial_qa.partial_questions.message_id,
                    conversation_timestamp=first_partial_qa.partial_questions.timestamp,
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
        tasks = [tg.create_task(_get_question_answer(row)) for row in prompts_outputs]
    partial_question_answers = [task.result() for task in tasks]

    conversations = await match_conversations(partial_question_answers)
    return conversations


async def parse_row_alert_conversation(
    row: GetAlertsWithPromptAndOutputRow,
) -> Optional[AlertConversation]:
    """
    Parse a row from the get_alerts_with_prompt_and_output query and return a Conversation

    The row contains the raw request and output strings from the pipeline.
    """
    partial_qa = await _get_question_answer(row)
    if not partial_qa:
        return None

    question_answer = _get_question_answer_from_partial(partial_qa)

    conversation = Conversation(
        question_answers=[question_answer],
        provider=row.provider,
        type=row.type,
        chat_id=row.id,
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
) -> List[AlertConversation]:
    """
    Parse a list of rows from the get_alerts_with_prompt_and_output query and return a list of
    AlertConversation

    The rows contain the raw request and output strings from the pipeline.
    """
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(parse_row_alert_conversation(row)) for row in alerts_conversations]
    return [task.result() for task in tasks if task.result() is not None]
