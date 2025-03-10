from typing import (
    Any,
    Iterable,
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


class ToolCallFunction(pydantic.BaseModel):
    name: str
    index: int | None = None
    arguments: Any | None = None


class ToolCall(pydantic.BaseModel):
    function: ToolCallFunction


class Message(pydantic.BaseModel):
    role: Role
    content: str
    images: Iterable[bytes] | None = None
    tool_calls: Iterable[ToolCall] | None = None

    def get_text(self):
        return self.content

    def set_text(self, text):
        self.content = text


class StreamingChatCompletion(pydantic.BaseModel):
    model: str
    created_at: int | str
    message: Message
    done: bool
    done_reason: str | None = None  # either `load`, `unload`, `length`, or `stop`
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None

    def get_content(self) -> Iterable[Message]:
        yield self.message

    # This should be abstracted better in the output pipeline
    def set_text(self, text) -> None:
        self.message.set_text(text)


class StreamingGenerateCompletion(pydantic.BaseModel):
    model: str
    created_at: int | str
    response: str
    done: bool
    done_reason: str | None = None  # either `load`, `unload`, `length`, or `stop`
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None

    def get_content(self):
        yield self

    def get_text(self):
        return self.response

    def set_text(self, text):
        self.response = text


class MessageError(pydantic.BaseModel):
    error: str

    def get_content(self) -> Iterable[Any]:
        return iter(())  # empty generator
