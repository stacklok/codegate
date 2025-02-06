import asyncio
import json
import re
from collections import defaultdict
from typing import AsyncGenerator, Dict, List, Optional, Tuple

import cachetools.func
import requests
import structlog

from codegate.api import v1_models
from codegate.api.v1_models import (
    AlertConversation,
    ChatMessage,
    Conversation,
    PartialQuestionAnswer,
    PartialQuestions,
    QuestionAnswer,
    TokenUsageAggregate,
    TokenUsageByModel,
)
from codegate.db.connection import alert_queue
from codegate.db.models import Alert, GetPromptWithOutputsRow, TokenUsage

logger = structlog.get_logger("codegate")


SYSTEM_PROMPTS = [
    "Given the following... please reply with a short summary that is 4-12 words in length, "
    "you should summarize what the user is asking for OR what the user is trying to accomplish. "
    "You should only respond with the summary, no additional text or explanation, "
    "you don't need ending punctuation.",
]


@cachetools.func.ttl_cache(maxsize=128, ttl=20 * 60)
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


async def parse_request(request_str: str) -> Tuple[Optional[List[str]], str]:
    """
    Parse the request string from the pipeline and return the message and the model.
    """
    try:
        request = json.loads(request_str)
    except Exception as e:
        logger.warning(f"Error parsing request: {request_str}. {e}")
        return None, ""

    model = request.get("model", "")
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

    # If still we don't have anything, return None string
    if not messages:
        return None, model

    # Respond with the messages and the model
    return messages, model


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


async def _get_partial_question_answer(
    row: GetPromptWithOutputsRow,
) -> Optional[PartialQuestionAnswer]:
    """
    Parse a row from the get_prompt_with_outputs query and return a PartialConversation

    The row contains the raw request and output strings from the pipeline.
    """
    async with asyncio.TaskGroup() as tg:
        request_task = tg.create_task(parse_request(row.request))
        output_task = tg.create_task(parse_output(row.output))

    request_user_msgs, model = request_task.result()
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

    token_usage = TokenUsage.from_db(
        input_cost=row.input_cost,
        input_tokens=row.input_tokens,
        output_tokens=row.output_tokens,
        output_cost=row.output_cost,
    )
    # Use the model to update the token cost
    provider = row.provider
    # TODO: This should come from the database. For now, we are manually changing copilot to openai
    # Change copilot provider to openai
    if provider == "copilot":
        provider = "openai"
    model_token_usage = TokenUsageByModel(
        model=model, token_usage=token_usage, provider_type=provider
    )

    alerts: List[v1_models.Alert] = [
        v1_models.Alert.from_db_model(db_alert) for db_alert in row.alerts
    ]

    return PartialQuestionAnswer(
        partial_questions=request_message,
        answer=output_message,
        model_token_usage=model_token_usage,
        alerts=alerts,
    )


def parse_question_answer(input_text: str) -> str:
    # Remove the <environment_details>...</environment_details> pattern if present
    env_details_pattern = r"\n<environment_details>.*?</environment_details>"
    input_text = re.sub(env_details_pattern, "", input_text, flags=re.DOTALL).strip()

    # Check for the <task>...</task> pattern first
    task_pattern = r"^<task>(.*?)</task>"
    task_match = re.search(task_pattern, input_text, re.DOTALL)

    if task_match:
        return task_match.group(1).strip()

    # If no <task>...</task>, check for "Context: xxx \n\nQuery: xxx"
    context_query_pattern = r"^Context:.*?\n\n\s*Query:\s*(.*)$"
    context_query_match = re.search(context_query_pattern, input_text, re.DOTALL)

    if context_query_match:
        return context_query_match.group(1).strip()

    # If no pattern matches, return the original input text
    return input_text


def _clean_secrets_from_message(message: str) -> str:
    pattern = re.compile(r"REDACTED<(\$?[^>]+)>")
    return pattern.sub("REDACTED_SECRET", message)


def _group_partial_messages(  # noqa: C901
    pq_list: List[PartialQuestions],
) -> List[List[PartialQuestions]]:
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
    # 0) Clean secrets from messages
    for pq in pq_list:
        pq.messages = [_clean_secrets_from_message(msg) for msg in pq.messages]

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
        possible_subsets: List[PartialQuestions] = []
        for sub in pq_list_sorted:
            if sub.message_id == sup.message_id or sub.message_id in used:
                continue
            if (
                set(sub.messages).issubset(set(sup.messages))
                and sub.provider == sup.provider
                and set(sub.messages) != set(sup.messages)
            ):
                possible_subsets.append(sub)

        # 3) If there are no subsets, check for time-based grouping
        if not possible_subsets:
            new_group = [sup]
            used.add(sup.message_id)

            for other in pq_list_sorted:
                if other.message_id in used or other.message_id == sup.message_id:
                    continue
                if abs((other.timestamp - sup.timestamp).total_seconds()) <= 5 and set(
                    other.messages
                ) & set(
                    sup.messages
                ):  # At least one message in common
                    new_group.append(other)
                    used.add(other.message_id)

            groups.append(new_group)
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
    Get a QuestionAnswer object from a PartialQuestionAnswer object. PartialQuestionAnswer
    contains a list of messages as question. QuestionAnswer contains a single message as question.
    """
    # Get the last user message as the question
    message_str = partial_question_answer.partial_questions.messages[-1]
    if (
        partial_question_answer.partial_questions.provider == "copilot"
        and partial_question_answer.partial_questions.type == "chat"
    ):
        message_str = "\n".join(partial_question_answer.partial_questions.messages)

    # sanitize answer from reserved words
    if partial_question_answer.answer and partial_question_answer.answer.message:
        partial_question_answer.answer.message = re.sub(
            r"(question_about_specific_files|question_about_specific_code|unknown)\s*",
            "",
            partial_question_answer.answer.message,
        ).strip()
    question = ChatMessage(
        message=message_str,
        timestamp=partial_question_answer.partial_questions.timestamp,
        message_id=partial_question_answer.partial_questions.message_id,
    )

    return QuestionAnswer(question=question, answer=partial_question_answer.answer)


async def match_conversations(
    partial_question_answers: List[Optional[PartialQuestionAnswer]],
) -> Tuple[List[Conversation], Dict[str, Conversation]]:
    """
    Match partial conversations to form a complete conversation.
    """
    grouped_partial_questions = _group_partial_messages(
        [partial_qs_a.partial_questions for partial_qs_a in partial_question_answers]
    )

    # Create the conversation objects
    conversations = []
    map_q_id_to_conversation = {}
    for group in grouped_partial_questions:
        questions_answers: List[QuestionAnswer] = []
        token_usage_agg = TokenUsageAggregate(tokens_by_model={}, token_usage=TokenUsage())
        alerts: List[v1_models.Alert] = []
        first_partial_qa = None
        for partial_question in sorted(group, key=lambda x: x.timestamp):
            # Partial questions don't contain the answer, so we need to find the corresponding
            # valid partial question answer
            selected_partial_qa = None
            for partial_qa in partial_question_answers:
                if partial_question.message_id == partial_qa.partial_questions.message_id:
                    selected_partial_qa = partial_qa
                    break

            #  check if we have a question and answer, otherwise do not add it
            if selected_partial_qa and selected_partial_qa.answer is not None:
                # if we don't have a first question, set it. We will use it
                # to set the conversation timestamp and provider
                first_partial_qa = first_partial_qa or selected_partial_qa
                qa = _get_question_answer_from_partial(selected_partial_qa)
                qa.question.message = parse_question_answer(qa.question.message)
                questions_answers.append(qa)
                alerts.extend(selected_partial_qa.alerts)
                token_usage_agg.add_model_token_usage(selected_partial_qa.model_token_usage)

        # only add conversation if we have some answers
        if len(questions_answers) > 0 and first_partial_qa is not None:
            if token_usage_agg.token_usage.input_tokens == 0:
                token_usage_agg = None
            conversation = Conversation(
                question_answers=questions_answers,
                provider=first_partial_qa.partial_questions.provider,
                type=first_partial_qa.partial_questions.type,
                chat_id=first_partial_qa.partial_questions.message_id,
                conversation_timestamp=first_partial_qa.partial_questions.timestamp,
                token_usage_agg=token_usage_agg,
                alerts=alerts,
            )
            for qa in questions_answers:
                map_q_id_to_conversation[qa.question.message_id] = conversation
            conversations.append(conversation)

    return conversations, map_q_id_to_conversation


async def _process_prompt_output_to_partial_qa(
    prompts_outputs: List[GetPromptWithOutputsRow],
) -> List[PartialQuestionAnswer]:
    """
    Process the prompts and outputs to PartialQuestionAnswer objects.
    """
    # Parse the prompts and outputs in parallel
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(_get_partial_question_answer(row)) for row in prompts_outputs]
    return [task.result() for task in tasks if task.result() is not None]


async def parse_messages_in_conversations(
    prompts_outputs: List[GetPromptWithOutputsRow],
) -> Tuple[List[Conversation], Dict[str, Conversation]]:
    """
    Get all the messages from the database and return them as a list of conversations.
    """
    partial_question_answers = await _process_prompt_output_to_partial_qa(prompts_outputs)

    conversations, map_q_id_to_conversation = await match_conversations(partial_question_answers)
    return conversations, map_q_id_to_conversation


async def parse_row_alert_conversation(
    row: Alert, map_q_id_to_conversation: Dict[str, Conversation]
) -> Optional[AlertConversation]:
    """
    Parse a row from the get_alerts_with_prompt_and_output query and return a Conversation

    The row contains the raw request and output strings from the pipeline.
    """
    conversation = map_q_id_to_conversation.get(row.prompt_id)
    if conversation is None:
        return None
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
    alerts: List[Alert],
    prompts_outputs: List[GetPromptWithOutputsRow],
) -> List[AlertConversation]:
    """
    Parse a list of rows from the get_alerts_with_prompt_and_output query and return a list of
    AlertConversation

    The rows contain the raw request and output strings from the pipeline.
    """
    _, map_q_id_to_conversation = await parse_messages_in_conversations(prompts_outputs)
    async with asyncio.TaskGroup() as tg:
        tasks = [
            tg.create_task(parse_row_alert_conversation(row, map_q_id_to_conversation))
            for row in alerts
        ]
    return [task.result() for task in tasks if task.result() is not None]


async def parse_workspace_token_usage(
    prompts_outputs: List[GetPromptWithOutputsRow],
) -> TokenUsageAggregate:
    """
    Parse the token usage from the workspace.
    """
    partial_question_answers = await _process_prompt_output_to_partial_qa(prompts_outputs)
    token_usage_agg = TokenUsageAggregate(tokens_by_model={}, token_usage=TokenUsage())
    for p_qa in partial_question_answers:
        token_usage_agg.add_model_token_usage(p_qa.model_token_usage)
    return token_usage_agg
