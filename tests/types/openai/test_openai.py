import asyncio
import json
import os
import pathlib

import pytest

from codegate.types.openai import (
    # generators
    message_wrapper,
    stream_generator,
    # request objects
    # response objects
    MessageError,
    StreamingChatCompletion,
)


pytest_plugins = ("pytest_asyncio",)


def read_file(fname):
    with open(fname, "rb") as fd:
        return fd.read().decode("utf-8")


@pytest.fixture(scope="session")
def streaming_messages():
    fname = os.path.join(pathlib.Path(__file__).parent, "streaming_messages.txt")
    return read_file(fname)


@pytest.mark.asyncio
async def test_message_wrapper_chat(streaming_messages):
    async def _line_iterator(data):
        for line in data.splitlines():
            yield line

    gen = message_wrapper(_line_iterator(streaming_messages))
    event = await anext(gen)
    assert event.model == "gpt-4o-2024-08-06"
    assert event.choices[0].delta.content == "content 1"

    event = await anext(gen)
    assert event.model == "gpt-4o-2024-08-06"
    assert event.choices[0].delta.content == "content 2"

    event = await anext(gen)
    assert event.model == "gpt-4o-2024-08-06"
    assert len(event.choices) == 0
    assert event.usage is not None

    with pytest.raises(StopAsyncIteration):
        await anext(gen)


@pytest.mark.asyncio
async def test_stream_generator(streaming_messages):
    async def _line_iterator(data):
        for line in data.splitlines():
            yield line

    gen = message_wrapper(_line_iterator(streaming_messages))
    gen = stream_generator(gen)

    event = await anext(gen)
    assert event.startswith("data: {")
    assert event.endswith("}\n\n")


@pytest.mark.asyncio
async def test_stream_generator_payload_error():
    async def _iterator():
        yield "Ceci n'est pas une classe"

    gen = stream_generator(_iterator())

    event = await anext(gen)
    assert event.startswith('data: {"error":')


@pytest.mark.asyncio
async def test_stream_generator_generator_error():
    async def _iterator():
        raise ValueError("boom")

    gen = stream_generator(_iterator())

    event = await anext(gen)
    assert event.startswith('data: {"error":')
