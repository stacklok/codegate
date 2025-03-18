import json
import time

from codegate.types import anthropic, openai


def anthropic_from_openai(request: openai.ChatCompletionRequest):
    res = anthropic.ChatCompletionRequest(
        max_tokens=map_max_tokens(request.max_tokens, request.max_completion_tokens),
        messages=map_messages(request.messages),
        model=map_model(request.model),
        # Anthropic only supports "user" metadata
        metadata={"user_id": request.user} if request.user else None,
        # OpenAI stop parameter might be a string
        stop_sequences=map_stop_sequences(request.stop),
        # OpenAI stream parameter might be None
        stream=request.stream if request.stream else False,
        system=map_system_messages(request.messages),
        # Anthropic range is [0,1], OpenAI's is [0,2]
        temperature=request.temperature / 2.0 if request.temperature else None,
        thinking=map_reasoning_effort(request.reasoning_effort),
        # simple default for now
        tools=map_tools(request.tools, request.functions),
        # this might be OpenAI's logit_bias, but I'm not sure
        top_k=None,
        top_p=request.top_p,
    )

    if request.tool_choice is not None and request.tools is not None:
        res.tool_choice = map_tool_choice(request.tool_choice)

    return res


def anthropic_from_legacy_openai(request: openai.LegacyCompletionRequest):
    res = anthropic.ChatCompletionRequest(
        max_tokens=request.max_tokens if request.max_tokens else 4096,
        messages=[
            anthropic.UserMessage(
                role="user",
                content=[
                    anthropic.TextContent(
                        type="text",
                        # We default to empty string when prompt is
                        # null since `text` field is mandatory for
                        # Anthropic.
                        text=request.prompt if request.prompt else "",
                    ),
                ],
            ),
        ],
        model=map_model(request.model),
        # OpenAI stop parameter might be a string
        stop_sequences=map_stop_sequences(request.stop),
        # OpenAI stream parameter might be None
        stream=request.stream if request.stream else False,
        # Anthropic range is [0,1], OpenAI's is [0,2]
        temperature=request.temperature / 2.0 if request.temperature else None,
        # this might be OpenAI's logit_bias, but I'm not sure
        top_k=None,
        top_p=request.top_p,
    )

    return res


def map_stop_sequences(stop_sequences):
    if not stop_sequences:
        return None
    if isinstance(stop_sequences, list):
        return stop_sequences
    return [stop_sequences]


def map_max_tokens(max_tokens, max_completion_tokens):
    if max_tokens:
        return max_tokens
    if max_completion_tokens:
        return max_completion_tokens
    return 4096


def map_model(openai_model):
    """Map OpenAI model names to Anthropic equivalents"""
    # This is a simplified mapping and would need to be expanded
    model_mapping = {
        "gpt-4": "claude-3-opus-20240229",
        "gpt-4-turbo": "claude-3-7-sonnet-20250219",
        "gpt-3.5-turbo": "claude-3-haiku-20240307",
        # Add more mappings as needed
    }
    return model_mapping.get(openai_model, "claude-3-7-sonnet-20250219")  # Default fallback


def map_reasoning_effort(openai_reasoning_effort):
    """Map OpenAI reasoning_effort to Anthropic thinking configuration"""
    # Map low/medium/high to Anthropic's thinking mode
    match openai_reasoning_effort:
        case "low":
            return anthropic.ThinkingEnabled(
                type="enabled",
                budget_tokens=1024,
            )
        case "medium":
            return anthropic.ThinkingEnabled(
                type="enabled",
                budget_tokens=1024,
            )
        case "high":
            return anthropic.ThinkingEnabled(
                type="enabled",
                budget_tokens=1024,
            )
        case _:
            return None


def map_tool_choice(openai_tool_choice):
    """Map OpenAI tool_choice to Anthropic tool_choice"""
    # Map OpenAI tool_choice to Anthropic tool_choice
    if openai_tool_choice is None:
        return None

    match openai_tool_choice:
        case "none":
            return anthropic.ToolChoice(type="none")
        case "auto":
            return anthropic.ToolChoice(type="auto")
        case "required":
            return anthropic.ToolChoice(type="any")
        case openai.ToolChoice(type="function", function=func):
            return anthropic.ToolChoice(type="tool", name=func.name)
        case _:
            return anthropic.ToolChoice(type="auto")


def map_tools(openai_tools, openai_functions):
    """Map OpenAI tools to Anthropic tools"""
    # This is a simplified mapping and would need to be expanded
    if openai_tools is None and openai_functions is None:
        return None

    anthropic_tools = []
    if openai_tools is not None:
        anthropic_tools.extend(
            anthropic.ToolDef(
                name=tool.function.name,
                description=tool.function.description,
                input_schema=tool.function.parameters,
            )
            for tool in openai_tools
        )

    # Handle deprecated OpenAI functions
    if openai_functions is not None:
        anthropic_tools.extend(
            anthropic.ToolDef(
                name=func.name,
                description=func.description,
                input_schema=func.parameters,
            )
            for func in openai_functions
        )

    return anthropic_tools


def map_messages(openai_messages):
    # Map OpenAI messages to Anthropic messages
    # This is a simplified mapping and would need to be expanded
    anthropic_messages = []
    for msg in openai_messages:
        match msg:
            # user messages
            case openai.UserMessage(content=content) if content is not None:
                anthropic_content = map_content(content)
                anthropic_messages.append(
                    anthropic.UserMessage(role="user", content=anthropic_content),
                )

            # assistant messages
            case openai.AssistantMessage(content=content) if content is not None:
                anthropic_content = map_content(content)
                anthropic_messages.append(
                    anthropic.AssistantMessage(role="assistant", content=anthropic_content),
                )
            case openai.AssistantMessage(content="", tool_calls=[calls], function_call=funcall):
                anthropic_content = [
                    anthropic.ToolUseContent(
                        id=call.id,
                        name=call.function.name,
                        input=json.loads(call.function.arguments),
                    )
                    for call in calls
                ]

                if funcall:
                    anthropic_content.append(
                        anthropic.ToolUseContent(
                            id=funcall.id,
                            name=funcall.function.name,
                            input=json.loads(funcall.function.arguments),
                        )
                    )
                anthropic_messages.append(
                    anthropic.AssistantMessage(
                        role="assistant",
                        content=anthropic_content,
                    ),
                )

            # tool messages
            case openai.ToolMessage(content=content) if content is not None:
                anthropic_content = map_content(content)
                anthropic_messages.append(
                    anthropic.UserMessage(
                        role="user",
                        content=anthropic_content,
                    ),
                )
            case openai.FunctionMessage(content=content) if content is not None:
                anthropic_content = map_content(content)
                anthropic_messages.append(
                    anthropic.UserMessage(
                        role="user",
                        content=anthropic_content,
                    ),
                )

            # system messages
            case openai.DeveloperMessage(content=content):
                pass  # this is the new system message
            case openai.SystemMessage(content=content):
                pass  # this is the legacy system message

            # other, not covered cases
            case _:
                # TODO add log message
                pass

    return anthropic_messages


def map_content(openai_content):
    if isinstance(openai_content, str):
        return [anthropic.TextContent(type="text", text=openai_content)]

    anthropic_content = []
    for item in openai_content:
        match item:
            case openai.TextContent(text=text):
                anthropic_content.append(
                    anthropic.TextContent(
                        type="text",
                        text=text,
                    ),
                )
            case openai.RefusalContent(text=text):
                anthropic_content.append(
                    anthropic.TextContent(
                        type="text",
                        text=text,
                    ),
                )
            case _:
                # TODO add log message
                pass

    return anthropic_content


def map_system_messages(openai_messages):
    # Map OpenAI system messages to Anthropic system messages
    # This is a simplified mapping and would need to be expanded
    system_prompts = []
    for msg in openai_messages:
        if isinstance(msg, openai.SystemMessage) or isinstance(msg, openai.DeveloperMessage):
            if isinstance(msg.content, list):
                system_prompts.extend([c.text for c in msg.content])
            else:  # str
                system_prompts.append(msg.content)
    return "\n".join(system_prompts)


######################
## RESPONSE OBJECTS ##
######################


async def anthropic_to_openai(stream):
    last_index = -1
    id = None
    model = None
    usage_input = None
    usage_output = None

    async for item in stream:
        match item:
            case anthropic.MessageStart():
                id = item.message.id
                model = item.message.model
                usage_input = item.message.usage.input_tokens if item.message.usage else 0
                usage_output = item.message.usage.output_tokens if item.message.usage else 0

                yield openai.StreamingChatCompletion(
                    id=id,
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=model,
                    choices=[
                        openai.ChoiceDelta(
                            index=last_index,
                            delta=openai.MessageDelta(
                                role="assistant",
                                content="",
                            ),
                        ),
                    ],
                )

            case anthropic.MessageDelta():
                if item.usage is not None:
                    if usage_output is None:
                        usage_output = item.usage.output_tokens
                    else:
                        usage_output = usage_output + item.usage.output_tokens

                yield openai.StreamingChatCompletion(
                    id=id,
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=model,
                    choices=[
                        openai.ChoiceDelta(
                            index=last_index,
                            delta=openai.MessageDelta(
                                role="assistant",
                                content="",
                            ),
                        ),
                    ],
                )

            case anthropic.ContentBlockStart():
                last_index = item.index
                yield openai.StreamingChatCompletion(
                    id=id,
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=model,
                    choices=[
                        openai.ChoiceDelta(
                            index=last_index,
                            delta=openai.MessageDelta(
                                role="assistant",
                                content="",
                            ),
                        ),
                    ],
                )

            case anthropic.ContentBlockDelta():
                content = None
                match item.delta:
                    # Block containing a TEXT delta
                    case anthropic.TextDelta(text=text):
                        content = text
                    # Block containing a JSON delta
                    case anthropic.InputJsonDelta(partial_json=partial_json):
                        content = partial_json

                yield openai.StreamingChatCompletion(
                    id=id,
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=model,
                    choices=[
                        openai.ChoiceDelta(
                            index=last_index,
                            delta=openai.MessageDelta(
                                role="assistant",
                                content=content,
                            ),
                        ),
                    ],
                )

            case anthropic.ContentBlockStop():
                # There's no equivalent of content_block_stop for
                # OpenAI, but this marks the last message before the
                # index gets updated.
                continue

            case anthropic.MessageStop():
                res = openai.StreamingChatCompletion(
                    id=id,
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=model,
                    choices=[
                        openai.ChoiceDelta(
                            index=last_index,
                            delta=openai.MessageDelta(),
                            finish_reason="stop",
                        ),
                    ],
                )

                # Set usage in output message.
                if usage_input is not None or usage_output is not None:
                    total_tokens = usage_output if usage_output else 0
                    total_tokens += usage_input if usage_input else 0
                    res.usage = openai.Usage(
                        completion_tokens=usage_output if usage_output else 0,
                        prompt_tokens=usage_input if usage_input else 0,
                        total_tokens=total_tokens,
                    )

                yield res

            case anthropic.MessagePing():
                # There's no equivalent of ping messages for OpenAI.
                continue

            # TODO refine the specific error adding code based on the
            # inner error type.
            case anthropic.MessageError(error=error):
                yield openai.MessageError(
                    error=openai.ErrorDetails(
                        message=error.message,
                        code=None,
                    ),
                )

            case _:
                raise ValueError(f"case not covered: {item}")


async def anthropic_to_legacy_openai(stream):
    id = None
    model = None
    usage_input = None
    usage_output = None

    async for item in stream:
        match item:
            case anthropic.MessageStart():
                id = item.message.id
                model = item.message.model
                usage_input = item.message.usage.input_tokens if item.message.usage else 0
                usage_output = item.message.usage.output_tokens if item.message.usage else 0

                yield openai.LegacyCompletion(
                    id=id,
                    object="text_completion",
                    created=int(time.time()),
                    model=model,
                    choices=[
                        openai.LegacyMessage(
                            text="",
                        ),
                    ],
                )

            case anthropic.MessageDelta():
                if item.usage is not None:
                    if usage_output is None:
                        usage_output = item.usage.output_tokens
                    else:
                        usage_output = usage_output + item.usage.output_tokens

                yield openai.LegacyCompletion(
                    id=id,
                    object="text_completion",
                    created=int(time.time()),
                    model=model,
                    choices=[
                        openai.LegacyMessage(
                            text="",
                        ),
                    ],
                )

            case anthropic.ContentBlockStart():
                yield openai.LegacyCompletion(
                    id=id,
                    object="text_completion",
                    created=int(time.time()),
                    model=model,
                    choices=[
                        openai.LegacyMessage(
                            text="",
                        ),
                    ],
                )

            case anthropic.ContentBlockDelta():
                content = None
                match item.delta:
                    # Block containing a TEXT delta
                    case anthropic.TextDelta(text=text):
                        content = text
                    # Block containing a JSON delta. Note that this
                    # should not happen in legacy calls since it's
                    # only used in FIM.
                    case anthropic.InputJsonDelta(partial_json=partial_json):
                        content = partial_json

                yield openai.LegacyCompletion(
                    id=id,
                    object="text_completion",
                    created=int(time.time()),
                    model=model,
                    choices=[
                        openai.LegacyMessage(
                            text=content,
                        ),
                    ],
                )

            case anthropic.ContentBlockStop():
                # There's no equivalent of content_block_stop for
                # OpenAI, but this marks the last message before the
                # index gets updated.
                continue

            case anthropic.MessageStop():
                res = openai.LegacyCompletion(
                    id=id,
                    object="text_completion",
                    created=int(time.time()),
                    model=model,
                    choices=[
                        openai.LegacyMessage(
                            text="",
                            finish_reason="stop",
                        ),
                    ],
                )

                # Set usage in output message.
                if usage_input is not None or usage_output is not None:
                    total_tokens = usage_output if usage_output else 0
                    total_tokens += usage_input if usage_input else 0
                    res.usage = openai.Usage(
                        completion_tokens=usage_output if usage_output else 0,
                        prompt_tokens=usage_input if usage_input else 0,
                        total_tokens=total_tokens,
                    )

                yield res

            case anthropic.MessagePing():
                # There's no equivalent of ping messages for OpenAI.
                continue

            # TODO refine the specific error adding code based on the
            # inner error type.
            case anthropic.MessageError(error=error):
                yield openai.MessageError(
                    error=openai.ErrorDetails(
                        message=error.message,
                        code=None,
                    ),
                )

            case _:
                raise ValueError(f"case not covered: {item}")
