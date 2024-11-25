from typing import AsyncIterator, Dict, List, Union

import pytest
from litellm import ModelResponse
from litellm.adapters.anthropic_adapter import AnthropicStreamWrapper
from litellm.types.llms.anthropic import (
    ContentBlockDelta,
    ContentBlockStart,
    ContentTextBlockDelta,
    MessageChunk,
    MessageStartBlock,
)
from litellm.types.utils import Delta, StreamingChoices

from codegate.providers.anthropic.adapter import AnthropicAdapter


@pytest.fixture
def adapter():
    return AnthropicAdapter()

def test_translate_completion_input_params(adapter):
    # Test input data
    completion_request = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 1024,
        "stream": True,
        "messages": [
         {
           "role": "user",
           "system": "You are an expert code reviewer",
           "content": [
             {
               "type": "text",
               "text": "Review this code"
             }
           ]
         }
       ]
    }
    expected = {
        'max_tokens': 1024,
        'messages': [
            {'content': [{'text': 'Review this code', 'type': 'text'}], 'role': 'user'}
        ],
        'model': 'claude-3-haiku-20240307',
        'stream': True
    }

    # Get translation
    result = adapter.translate_completion_input_params(completion_request)
    assert result == expected

@pytest.mark.asyncio
async def test_translate_completion_output_params_streaming(adapter):
    # Test stream data
    async def mock_stream():
        messages = [
            ModelResponse(
                id="test_id_1",
                choices=[
                    StreamingChoices(
                        finish_reason=None,
                        index=0,
                        delta=Delta(content="Hello", role="assistant")),
                ],
                model="claude-3-haiku-20240307",
            ),
            ModelResponse(
                id="test_id_2",
                choices=[
                    StreamingChoices(finish_reason=None,
                                     index=0,
                                     delta=Delta(content="world", role="assistant")),
                ],
                model="claude-3-haiku-20240307",
            ),
            ModelResponse(
                id="test_id_2",
                choices=[
                    StreamingChoices(finish_reason=None,
                                     index=0,
                                     delta=Delta(content="!", role="assistant")),
                ],
                model="claude-3-haiku-20240307",
            ),
        ]
        for msg in messages:
            yield msg

    expected: List[Union[MessageStartBlock,ContentBlockStart,ContentBlockDelta]] = [
        MessageStartBlock(
            type="message_start",
            message=MessageChunk(
                id="msg_1nZdL29xx5MUA1yADyHTEsnR8uuvGzszyY",
                type="message",
                role="assistant",
                content=[],
                # litellm makes up a message start block with hardcoded values
                model="claude-3-5-sonnet-20240620",
                stop_reason=None,
                stop_sequence=None,
                usage={"input_tokens": 25, "output_tokens": 1},
            ),
        ),
        ContentBlockStart(
            type="content_block_start",
            index=0,
            content_block={"type": "text", "text": ""},
        ),
        ContentBlockDelta(
            type="content_block_delta",
            index=0,
            delta=ContentTextBlockDelta(type="text_delta", text="Hello"),
        ),
        ContentBlockDelta(
            type="content_block_delta",
            index=0,
            delta=ContentTextBlockDelta(type="text_delta", text="world"),
        ),
        ContentBlockDelta(
            type="content_block_delta",
            index=0,
            delta=ContentTextBlockDelta(type="text_delta", text="!"),
        ),
        # litellm doesn't seem to have a type for message stop
        dict(type="message_stop"),
    ]

    stream = adapter.translate_completion_output_params_streaming(mock_stream())
    assert isinstance(stream, AnthropicStreamWrapper)

    # just so that we can zip over the expected chunks
    stream_list = [chunk async for chunk in stream]
    # Verify we got all chunks
    assert len(stream_list) == 6

    for chunk, expected_chunk in zip(stream_list, expected):
        assert chunk == expected_chunk


def test_stream_generator_initialization(adapter):
    # Verify the default stream generator is set
    from codegate.providers.litellmshim import anthropic_stream_generator
    assert adapter.stream_generator == anthropic_stream_generator

def test_custom_stream_generator():
    # Test that we can inject a custom stream generator
    async def custom_generator(stream: AsyncIterator[Dict]) -> AsyncIterator[str]:
        async for chunk in stream:
            yield "custom: " + str(chunk)

    adapter = AnthropicAdapter(stream_generator=custom_generator)
    assert adapter.stream_generator == custom_generator
