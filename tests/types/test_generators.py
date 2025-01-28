import pytest

from codegate.types.generators import (
    CodegateFunction,
    CodegateChatCompletionDeltaToolCall,
    CodegateDelta,
    CodegateStreamingChoices,
    CodegateModelResponseStream,
)

@pytest.mark.parametrize(
    "payload",
    [
        {"name": "foo"},
        {"arguments": "bar"},
    ],
)
def test_codegate_function(payload):
    try:
        obj = CodegateFunction(**payload)
    except Exception as e:
        print(repr(e))
        raise e


@pytest.mark.parametrize(
    "payload",
    [
        {"id": "myid", "function": CodegateFunction(), "type": "function", "index": 1},
        {"function": CodegateFunction(), "type": "function", "index": 1},
        {"function": CodegateFunction(), "type": "function", "index": 1},
        {"function": CodegateFunction(), "type": "function"},
    ],
)
def test_codegate_chat_completion_delta_tool_call(payload):
    try:
        obj = CodegateChatCompletionDeltaToolCall(**payload)
    except Exception as e:
        print(repr(e))
        raise e
        

@pytest.mark.parametrize(
    "payload",
    [
        {"role": "assistant", "content": "text"},
        {"role": "assistant", "tool_calls": "tool_calls"},
        {"role": "assistant",},
    ],
)
def test_codegate_chat_completion_delta_tool_call(payload):
    try:
        obj = CodegateChatCompletionDeltaToolCall(**payload)
    except Exception as e:
        print(repr(e))
        raise e
