import json
import os
import pathlib

import pytest

from codegate.types.anthropic import (
    CodegateChatCompletionRequest,
)


@pytest.fixture(scope="session")  
def tools_request():
    fname = os.path.join(pathlib.Path(__file__).parent, "tools_request.json")
    with open(fname, "rb") as fd:
        return json.load(fd)


def test_chat_completion_request_serde_anthropic(tools_request):
    req = CodegateChatCompletionRequest(**tools_request)
    assert req.get("model") == 'claude-3-5-sonnet-20241022'
    assert req.get("stream") == True
    assert len(req.get("messages")) == 1
    assert req.get("messages")[0].get("role") == "user"
    assert type(req.get("messages")[0].get("content")) == str
