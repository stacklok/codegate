from enum import Enum
from typing import (
    Dict,
    List,
    Optional,
)

from pydantic import BaseModel


class CodegateFunction(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class CodegateChatCompletionDeltaToolCall(BaseModel):
    id: Optional[str] = None
    function: CodegateFunction
    type: str
    index: Optional[int] = None


class CodegateDelta(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[CodegateChatCompletionDeltaToolCall]] = None


class CodegateStreamingChoices(BaseModel):
    delta: CodegateDelta
    index: Optional[int] = None
    finish_reason: Optional[str] = None


class CodegateModelResponseStream(BaseModel):
    id: Optional[str] = None
    created: Optional[int] = None
    model: str
    object: str
    choices: Optional[List[CodegateStreamingChoices]] = None
    payload: Optional[Dict] = None


class MessageTypeFilter(Enum):
    """
    Enum of supported message type filters
    """

    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    USER = "user"
