import asyncio
import json
import os
import pathlib

import pytest

from codegate.types.anthropic import (
    # generators
    message_wrapper,
    stream_generator,
    # request objects
    ChatCompletionRequest,
    # response objects
    ApiError,
    AuthenticationError,
    ContentBlockDelta,
    ContentBlockStart,
    ContentBlockStop,
    InvalidRequestError,
    MessageDelta,
    MessageError,
    MessagePing,
    MessageStart,
    MessageStop,
    NotFoundError,
    OverloadedError,
    PermissionError,
    RateLimitError,
    RequestTooLargeError,
)


pytest_plugins = ("pytest_asyncio",)


def read_file(fname):
    with open(fname, "rb") as fd:
        return fd.read().decode("utf-8")


@pytest.fixture(scope="session")
def tools_request():
    fname = os.path.join(pathlib.Path(__file__).parent, "tools_request.json")
    return read_file(fname)


@pytest.fixture(scope="session")
def streaming_messages():
    fname = os.path.join(pathlib.Path(__file__).parent, "streaming_messages.txt")
    return read_file(fname)


@pytest.fixture(scope="session")
def streaming_messages_error():
    fname = os.path.join(pathlib.Path(__file__).parent, "streaming_messages_error.txt")
    return read_file(fname)


@pytest.fixture(scope="session")
def streaming_messages_simple():
    fname = os.path.join(pathlib.Path(__file__).parent, "streaming_messages_simple.txt")
    return read_file(fname)


def test_chat_completion_request_serde_anthropic(tools_request):
    req = ChatCompletionRequest.model_validate_json(tools_request)
    assert req.max_tokens == 4096
    assert req.model == "claude-3-5-sonnet-20241022"
    assert req.metadata is None
    assert req.stop_sequences is None
    assert req.stream  # is True
    assert req.system.startswith("When generating new code:")
    assert req.temperature is None
    assert req.tool_choice is None
    assert req.top_k is None
    assert req.top_p is None

    assert len(req.messages) == 1
    assert req.messages[0].role == "user"
    assert req.messages[0].content == "Please, read the content of file FUBAR.txt."

    assert len(req.tools) == 9
    assert req.tools[0].name == "builtin_read_file"
    assert (
        req.tools[0].description
        == "Use this tool whenever you need to view the contents of a file."
    )


@pytest.mark.asyncio
async def test_message_wrapper(streaming_messages):
    async def _line_iterator(data):
        for line in data.splitlines():
            yield line

    async for item in message_wrapper(_line_iterator(streaming_messages)):
        assert item.__class__ in [
            ApiError,
            AuthenticationError,
            ContentBlockDelta,
            ContentBlockStart,
            ContentBlockStop,
            InvalidRequestError,
            MessageDelta,
            MessageError,
            MessagePing,
            MessageStart,
            MessageStop,
            NotFoundError,
            OverloadedError,
            PermissionError,
            RateLimitError,
            RequestTooLargeError,
        ]


@pytest.mark.asyncio
async def test_message_wrapper_error(streaming_messages_error):
    async def _line_iterator(data):
        for line in data.splitlines():
            yield line

    async for item in message_wrapper(_line_iterator(streaming_messages_error)):
        assert item.__class__ in [
            ApiError,
            AuthenticationError,
            ContentBlockDelta,
            ContentBlockStart,
            ContentBlockStop,
            InvalidRequestError,
            MessageDelta,
            MessageError,
            MessagePing,
            MessageStart,
            MessageStop,
            NotFoundError,
            OverloadedError,
            PermissionError,
            RateLimitError,
            RequestTooLargeError,
        ]


@pytest.mark.asyncio
async def test_message_wrapper(streaming_messages_simple):
    async def _line_iterator(data):
        for line in data.splitlines():
            yield line

    gen = message_wrapper(_line_iterator(streaming_messages_simple))
    event = await anext(gen)
    assert event.type == "message_start"
    assert event.message.id == "msg_014p7gG3wDgGV9EUtLvnow3U"
    assert event.message.role == "assistant"
    assert event.message.model == "claude-3-haiku-20240307"

    event = await anext(gen)
    assert event.type == "content_block_start"
    assert event.index == 0
    assert event.content_block.type == "text"
    assert event.content_block.text == "some random text"

    event = await anext(gen)
    assert event.type == "ping"

    event = await anext(gen)
    assert event.type == "content_block_delta"
    assert event.index == 0
    assert event.delta.type == "text_delta"
    assert event.delta.text == "delta 1"

    event = await anext(gen)
    assert event.type == "content_block_delta"
    assert event.index == 0
    assert event.delta.type == "text_delta"
    assert event.delta.text == "delta 2"

    event = await anext(gen)
    assert event.type == "content_block_stop"
    assert event.index == 0

    event = await anext(gen)
    assert event.type == "content_block_start"
    assert event.index == 1
    assert event.content_block.type == "tool_use"
    assert event.content_block.id == "toolu_01T1x1fJ34qAmk2tNTrN7Up6"
    assert event.content_block.name == "get_weather"

    payload_chunks = []
    event = await anext(gen)
    assert event.type == "content_block_delta"
    assert event.index == 1
    assert event.delta.type == "input_json_delta"
    payload_chunks.append(event.delta.partial_json)

    event = await anext(gen)
    assert event.type == "content_block_delta"
    assert event.index == 1
    assert event.delta.type == "input_json_delta"
    payload_chunks.append(event.delta.partial_json)

    event = await anext(gen)
    assert event.type == "content_block_delta"
    assert event.index == 1
    assert event.delta.type == "input_json_delta"
    payload_chunks.append(event.delta.partial_json)

    event = await anext(gen)
    assert event.type == "content_block_delta"
    assert event.index == 1
    assert event.delta.type == "input_json_delta"
    payload_chunks.append(event.delta.partial_json)

    assert {"foo": "bar"} == json.loads("".join(payload_chunks))

    event = await anext(gen)
    assert event.type == "content_block_stop"
    assert event.index == 1

    event = await anext(gen)
    assert event.type == "message_delta"
    assert event.delta.stop_reason == "tool_use"
    assert event.delta.stop_sequence is None

    event = await anext(gen)
    assert event.type == "message_stop"


@pytest.mark.asyncio
async def test_message_wrapper_broken_protocol():
    async def _iterator():
        yield "event: content_block_stop"
        yield "data: {}"
        yield ""

    gen = message_wrapper(_iterator())
    with pytest.raises(ValueError):
        event = await anext(gen)


@pytest.mark.asyncio
async def test_message_wrapper_error_short_circuits():
    async def _iterator():
        yield "event: error"
        yield 'data: {"type": "error", "error": {"type": "api_error", "message": "boom"}}'
        yield ""

    gen = message_wrapper(_iterator())
    event = await anext(gen)
    assert event.type == "error"
    assert event.error.type == "api_error"
    assert event.error.message == "boom"

    with pytest.raises(StopAsyncIteration):
        await anext(gen)


@pytest.mark.asyncio
async def test_message_wrapper_message_stop_short_circuits():
    async def _iterator():
        yield "event: message_start"
        yield 'data: {"type":"message_start","message":{"id":"msg_014p7gG3wDgGV9EUtLvnow3U","type":"message","role":"assistant","model":"claude-3-haiku-20240307","stop_sequence":null,"usage":{"input_tokens":472,"output_tokens":2},"content":[],"stop_reason":null}}'
        yield ""
        yield "event: message_stop"
        yield 'data: {"type":"message_stop"}'
        yield ""

    gen = message_wrapper(_iterator())
    event = await anext(gen)
    assert event.type == "message_start"

    event = await anext(gen)
    assert event.type == "message_stop"

    with pytest.raises(StopAsyncIteration):
        await anext(gen)


@pytest.mark.asyncio
async def test_message_wrapper_unknown_type():
    async def _iterator():
        yield "event: message_start"
        yield 'data: {"type":"message_start","message":{"id":"msg_014p7gG3wDgGV9EUtLvnow3U","type":"message","role":"assistant","model":"claude-3-haiku-20240307","stop_sequence":null,"usage":{"input_tokens":472,"output_tokens":2},"content":[],"stop_reason":null}}'
        yield ""
        yield "event: unknown_type"
        yield 'data: {"type":"unknown_type"}'
        yield ""

    gen = message_wrapper(_iterator())
    await anext(gen)
    with pytest.raises(ValueError):
        await anext(gen)


@pytest.mark.asyncio
async def test_stream_generator(streaming_messages_simple):
    async def _line_iterator(data):
        for line in data.splitlines():
            yield line

    gen = message_wrapper(_line_iterator(streaming_messages_simple))
    gen = stream_generator(gen)

    event = await anext(gen)
    assert event.startswith("event: message_start")
    assert "data: " in event
    assert event.endswith("\n\n")

    event = await anext(gen)
    assert event.startswith("event: content_block_start")
    assert "data: " in event
    assert "some random text" in event
    assert event.endswith("\n\n")

    event = await anext(gen)
    assert event.startswith("event: ping")
    assert "data: " in event
    assert event.endswith("\n\n")

    event = await anext(gen)
    assert event.startswith("event: content_block_delta")
    assert "data: " in event
    assert "text_delta" in event
    assert "delta 1" in event
    assert event.endswith("\n\n")

    event = await anext(gen)
    assert event.startswith("event: content_block_delta")
    assert "data: " in event
    assert "text_delta" in event
    assert "delta 2" in event
    assert event.endswith("\n\n")

    event = await anext(gen)
    assert event.startswith("event: content_block_stop")
    assert "data: " in event
    assert event.endswith("\n\n")

    event = await anext(gen)
    assert event.startswith("event: content_block_start")
    assert "data: " in event
    assert "tool_use" in event
    assert "toolu_01T1x1fJ34qAmk2tNTrN7Up6" in event
    assert "get_weather" in event
    assert event.endswith("\n\n")

    event = await anext(gen)
    assert event.startswith("event: content_block_delta")
    assert "data: " in event
    assert "input_json_delta" in event
    assert event.endswith("\n\n")

    event = await anext(gen)
    assert event.startswith("event: content_block_delta")
    assert "data: " in event
    assert "input_json_delta" in event
    assert event.endswith("\n\n")

    event = await anext(gen)
    assert event.startswith("event: content_block_delta")
    assert "data: " in event
    assert "input_json_delta" in event
    assert event.endswith("\n\n")

    event = await anext(gen)
    assert event.startswith("event: content_block_delta")
    assert "data: " in event
    assert "input_json_delta" in event
    assert event.endswith("\n\n")

    event = await anext(gen)
    assert event.startswith("event: content_block_stop")
    assert "data: " in event
    assert event.endswith("\n\n")

    event = await anext(gen)
    assert event.startswith("event: message_delta")
    assert "data: " in event
    assert "tool_use" in event
    assert event.endswith("\n\n")

    event = await anext(gen)
    assert event.startswith("event: message_stop")
    assert "data: " in event
    assert event.endswith("\n\n")


@pytest.mark.asyncio
async def test_stream_generator_payload_error():
    async def _iterator():
        yield "Ceci n'est pas une classe"

    gen = stream_generator(_iterator())

    event = await anext(gen)
    assert event.startswith("event: error")


@pytest.mark.asyncio
async def test_stream_generator_generator_error():
    async def _iterator():
        raise ValueError("boom")

    gen = stream_generator(_iterator())

    event = await anext(gen)
    assert event.startswith("event: error")
