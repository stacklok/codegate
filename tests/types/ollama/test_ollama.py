import asyncio
import json
import os
import pathlib

import pytest

from codegate.types.ollama import (
    # generators
    message_wrapper,
    stream_generator,
    # request objects
    # response objects
    MessageError,
    StreamingChatCompletion,
    StreamingGenerateCompletion,
)


pytest_plugins = ('pytest_asyncio',)


def read_file(fname):
    with open(fname, "rb") as fd:
        return fd.read().decode("utf-8")


@pytest.fixture(scope="session")
def streaming_messages():
    fname = os.path.join(pathlib.Path(__file__).parent, "streaming_messages.txt")
    return read_file(fname)


@pytest.fixture(scope="session")
def streaming_generate():
    fname = os.path.join(pathlib.Path(__file__).parent, "streaming_generate.txt")
    return read_file(fname)


@pytest.mark.asyncio
async def test_message_wrapper_chat(streaming_messages):
    async def _line_iterator(data):
        for line in data.splitlines():
            yield line

    gen = message_wrapper(StreamingChatCompletion, _line_iterator(streaming_messages))
    event = await anext(gen)
    assert event.model == "deepseek-r1:7b"
    assert event.message.content == "content 1"
    assert not event.done
    assert event.done_reason is None

    event = await anext(gen)
    assert event.model == "deepseek-r1:7b"
    assert event.message.content == "content 2"
    assert not event.done
    assert event.done_reason is None

    event = await anext(gen)
    assert event.model == "deepseek-r1:7b"
    assert event.message.content == "content 3"
    assert event.done
    assert event.done_reason == "stop"

    with pytest.raises(StopAsyncIteration):
        await anext(gen)


@pytest.mark.asyncio
async def test_stream_generator(streaming_messages):
    async def _line_iterator(data):
        for line in data.splitlines():
            yield line

    gen = message_wrapper(StreamingChatCompletion, _line_iterator(streaming_messages))
    gen = stream_generator(gen)

    event = await anext(gen)
    assert event.startswith("{")
    assert event.endswith("}\n")


@pytest.mark.asyncio
async def test_stream_generator(streaming_generate):
    async def _line_iterator(data):
        for line in data.splitlines():
            yield line

    gen = message_wrapper(StreamingGenerateCompletion, _line_iterator(streaming_generate))
    gen = stream_generator(gen)

    events = [event async for event in gen]
    assert len(events) == 47
    first = events[0]
    assert '"done":false' in first
    last = events[-1]
    assert '"done":true' in last


@pytest.mark.asyncio
async def test_stream_generator_payload_error():
    async def _iterator():
        yield "Ceci n'est pas une classe"

    gen = stream_generator(_iterator())

    event = await anext(gen)
    assert event.startswith('{"error":')


@pytest.mark.asyncio
async def test_stream_generator_generator_error():
    async def _iterator():
        raise ValueError("boom")

    gen = stream_generator(_iterator())

    event = await anext(gen)
    assert event.startswith('{"error":')
