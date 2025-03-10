import json
import random
import string
import time
from typing import AsyncIterable, Callable, Iterable, List, Literal, Union

import codegate.types.ollama as ollama
import codegate.types.openai as openai


def _convert_format(response_format: openai.ResponseFormat) -> dict | Literal["json"] | None:
    """
    Safely convert OpenAI response format to Ollama format structure
    """
    if not response_format:
        return None

    if response_format.type == "json_object":
        return "json"

    if response_format.type != "json_schema":
        return None

    if not response_format.json_schema or not response_format.json_schema.schema:
        return None

    return response_format.json_schema.schema


def _process_options(request: openai.ChatCompletionRequest) -> dict:
    """
    Convert OpenAI request parameters to Ollama options
    """
    options = {}

    # do we need to for chat?
    if request.stop:
        if isinstance(request.stop, str):
            options["stop"] = [request.stop]
        elif isinstance(request.stop, list):
            options["stop"] = request.stop

    if request.max_tokens:
        options["num_predict"] = request.max_tokens
    elif request.max_completion_tokens:
        options["num_predict"] = request.max_completion_tokens

    if request.temperature is not None:
        options["temperature"] = request.temperature

    if request.seed is not None:
        options["seed"] = request.seed

    if request.frequency_penalty is not None:
        options["frequency_penalty"] = request.frequency_penalty

    if request.presence_penalty is not None:
        options["presence_penalty"] = request.presence_penalty

    if request.top_p is not None:
        options["top_p"] = request.top_p

    return options


def _extract_text_content(message: openai.Message) -> str:
    """
    Extract and join text content from a message's content items
    """
    text_parts = []
    for content in message.get_content():
        if text := content.get_text():
            text_parts.append(text)
    return " ".join(text_parts)


def _convert_tool_calls(tool_calls: List[openai.ToolCall] | None) -> List[ollama.ToolCall]:
    res_tool_calls = []
    if not tool_calls:
        return res_tool_calls
    for tool_call in tool_calls:
        res_tool_calls.append(
            ollama.ToolCall(
                function=ollama.Function(
                    name=tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments),
                )
            )
        )
    return res_tool_calls


def _convert_message(message: openai.Message) -> ollama.Message:
    """
    Convert OpenAI message to Ollama message format using pattern matching
    """
    text_content = _extract_text_content(message)

    match message:
        case openai.UserMessage():
            return ollama.UserMessage(role="user", content=text_content)
        case openai.SystemMessage() | openai.DeveloperMessage():  # Handle both as system messages
            return ollama.SystemMessage(role="system", content=text_content)
        case openai.AssistantMessage():
            return ollama.AssistantMessage(
                role="assistant",
                content=text_content,
                tool_calls=_convert_tool_calls(message.tool_calls),
            )
        case openai.ToolMessage():
            return ollama.ToolMessage(role="tool", content=text_content)
        case _:
            raise ValueError(f"Unsupported message type: {type(message)}")


def _convert_tools(tools: List[openai.ToolDef] | None) -> List[ollama.ToolDef] | None:
    """
    Convert OpenAI tools to Ollama format
    """
    if not tools:
        return None

    ollama_tools = []
    for tool in tools:
        # Convert the parameters format if needed
        parameters = None
        if tool.function.parameters:
            # OpenAI parameters are a dict, need to convert to Ollama Parameters object
            # This conversion depends on the exact structure expected by Ollama
            properties = {}
            for prop_name, prop_data in tool.function.parameters.get("properties", {}).items():
                properties[prop_name] = ollama.Property(
                    type=prop_data.get("type"), description=prop_data.get("description")
                )

            parameters = ollama.Parameters(
                type="object",
                required=tool.function.parameters.get("required"),
                properties=properties,
            )

        # Create the Ollama function definition
        function_def = ollama.FunctionDef(
            name=tool.function.name, description=tool.function.description, parameters=parameters
        )

        # Create the Ollama tool definition
        ollama_tools.append(ollama.ToolDef(type="function", function=function_def))

    return ollama_tools


def ollama_chat_from_openai(request: openai.ChatCompletionRequest) -> ollama.ChatRequest:
    """
    Convert OpenAI chat completion request to Ollama chat request
    """
    messages = [_convert_message(msg) for msg in request.get_messages()]
    options = _process_options(request)
    tools = _convert_tools(request.tools)

    req = ollama.ChatRequest(
        model=request.model,  # to be rewritten later
        messages=messages,
        # ollama has a different default
        stream=request.stream if request.stream is not None else True,
        tools=tools,
        format=_convert_format(request.response_format) if request.response_format else None,
        options=options,
    )
    return req


def ollama_generate_from_openai(
    request: openai.ChatCompletionRequest,
) -> ollama.GenerateRequest:
    """
    Convert OpenAI completion request to Ollama generate request
    """
    options = {}

    if request.stop:
        if isinstance(request.stop, str):
            options["stop"] = [request.stop]
        elif isinstance(request.stop, list):
            options["stop"] = request.stop

    if request.max_tokens:
        options["num_predict"] = request.max_tokens

    if request.temperature is not None:
        options["temperature"] = request.temperature

    if request.seed is not None:
        options["seed"] = request.seed

    if request.frequency_penalty is not None:
        options["frequency_penalty"] = request.frequency_penalty
    if request.presence_penalty is not None:
        options["presence_penalty"] = request.presence_penalty

    if request.top_p is not None:
        options["top_p"] = request.top_p

    user_message = request.last_user_message()

    # todo: when converting from the legacy format we would have to handle the suffix
    # what format is sent depends on the client though
    return ollama.GenerateRequest(
        model=request.model,  # to be rewritten later
        prompt=user_message[0].get_text() if user_message else "",
        stream=request.stream if request.stream is not None else True,
        options=options,
    )


def _gen_tool_call_id():
    letter_bytes = string.ascii_lowercase + string.digits
    b = [letter_bytes[random.randint(0, len(letter_bytes) - 1)] for _ in range(8)]
    return "call_" + "".join(b).lower()


def _openai_tool_calls_from_ollama(
    tool_calls: Iterable[ollama.ToolCall],
) -> Iterable[openai.ToolCall] | None:
    if not tool_calls:
        return None
    openai_tool_calls = []
    for tool_call in tool_calls:
        json_args = json.dumps(tool_call.function.arguments)

        openai_tool_calls.append(
            openai.ToolCall(
                id=_gen_tool_call_id(),
                type="function",
                function=openai.FunctionCall(
                    name=tool_call.function.name,
                    arguments=json_args,
                ),
            )
        )

    return openai_tool_calls


def openai_chunk_from_ollama_chat(
    ollama_chunk: ollama.StreamingChatCompletion,
) -> openai.StreamingChatCompletion:
    tool_calls = _openai_tool_calls_from_ollama(ollama_chunk.message.tool_calls)

    finish_reason = None
    if ollama_chunk.done_reason:
        finish_reason = ollama_chunk.done_reason
        if tool_calls:
            finish_reason = "tool_calls"

    return openai.StreamingChatCompletion(
        id="codegate-id",  # TODO: generate a random one?
        created=int(time.time()),
        model=ollama_chunk.model,
        choices=[
            openai.ChoiceDelta(
                index=0,
                finish_reason=finish_reason,
                delta=openai.MessageDelta(
                    content=ollama_chunk.message.content,
                    tool_calls=tool_calls,
                    role="assistant",
                ),
            ),
        ],
        usage=openai.Usage(
            prompt_tokens=ollama_chunk.prompt_eval_count if ollama_chunk.prompt_eval_count else 0,
            completion_tokens=ollama_chunk.eval_count if ollama_chunk.eval_count else 0,
            total_tokens=(
                ollama_chunk.prompt_eval_count
                if ollama_chunk.prompt_eval_count
                else 0 + ollama_chunk.eval_count if ollama_chunk.eval_count else 0
            ),
        ),
    )


def openai_chunk_from_ollama_generate(
    ollama_chunk: ollama.StreamingGenerateCompletion,
) -> openai.StreamingChatCompletion:
    return openai.StreamingChatCompletion(
        id="codegate-id",  # TODO: generate a random one?
        created=int(time.time()),
        model=ollama_chunk.model,
        choices=[
            openai.ChoiceDelta(
                index=0,
                finish_reason=ollama_chunk.done_reason,
                delta=openai.MessageDelta(
                    content=ollama_chunk.response,
                    role="assistant",
                ),
            ),
        ],
        usage=openai.Usage(
            prompt_tokens=ollama_chunk.prompt_eval_count if ollama_chunk.prompt_eval_count else 0,
            completion_tokens=ollama_chunk.eval_count if ollama_chunk.eval_count else 0,
            total_tokens=(
                ollama_chunk.prompt_eval_count
                if ollama_chunk.prompt_eval_count
                else 0 + ollama_chunk.eval_count if ollama_chunk.eval_count else 0
            ),
        ),
    )


async def ollama_stream_to_openai_stream(
    stream: AsyncIterable[
        Union[
            ollama.StreamingChatCompletion,
            ollama.StreamingGenerateCompletion,
        ]
    ],
    convert_fn: Callable,
) -> AsyncIterable[openai.StreamingChatCompletion]:
    """
    Convert a stream of Ollama streaming completions to OpenAI streaming completions
    """
    async for chunk in stream:
        converted_chunk = convert_fn(chunk)
        yield converted_chunk


async def ollama_chat_stream_to_openai_stream(
    stream: AsyncIterable[ollama.StreamingChatCompletion],
) -> AsyncIterable[openai.StreamingChatCompletion]:
    async for chunk in stream:
        converted_chunk = openai_chunk_from_ollama_chat(chunk)
        yield converted_chunk


async def ollama_generate_stream_to_openai_stream(
    stream: AsyncIterable[ollama.StreamingGenerateCompletion],
) -> AsyncIterable[openai.StreamingChatCompletion]:
    async for chunk in stream:
        converted_chunk = openai_chunk_from_ollama_generate(chunk)
        yield converted_chunk
