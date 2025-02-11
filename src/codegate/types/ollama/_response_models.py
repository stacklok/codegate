from typing import (
    Literal,
    Union,
)

import pydantic


Role = Union[
    Literal["user"],
    Literal["assistant"],
    Literal["system"],
    Literal["tool"],
]


class Message(pydantic.BaseModel):
    role: Role
    content: str


class StreamingChatCompletion(pydantic.BaseModel):
    model: str
    created_at: int | str
    message: Message
    done: bool
    done_reason: str | None = None # either `load`, `unload`, or `stop`
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None


class MessageError(pydantic.BaseModel):
    error: str
