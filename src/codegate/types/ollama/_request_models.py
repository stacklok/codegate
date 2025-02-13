from typing import (
    Any,
    Iterable,
    List,
    Literal,
    Mapping,
    Union,
)

import pydantic


class Property(pydantic.BaseModel):
    type: str | None = None
    description: str | None = None


class Parameters(pydantic.BaseModel):
    type: Literal["object"] | None = "object"
    required: List[str] | None = None
    properties: Mapping[str, Property] | None = None


class FunctionDef(pydantic.BaseModel):
    name: str | None = None
    description: str | None = None
    parameters: Parameters | None = None


class ToolDef(pydantic.BaseModel):
    type: Literal["function"] | None = "function"
    function: FunctionDef | None = None


class Function(pydantic.BaseModel):
    name: str
    arguments: dict


class ToolCall(pydantic.BaseModel):
    function: Function


class UserMessage(pydantic.BaseModel):
    role: Literal["user"]
    content: str | None = None
    images: List[bytes] | None = None
    tool_calls: List[ToolCall] | None = None

    def get_content(self) -> Iterable[Any]:
        yield self

    def get_text(self) -> str | None:
        return self.content

    def set_text(self, text) -> None:
        self.content = text


class AssistantMessage(pydantic.BaseModel):
    role: Literal["assistant"]
    content: str | None = None
    images: List[bytes] | None = None
    tool_calls: List[ToolCall] | None = None

    def get_content(self) -> Iterable[Any]:
        yield self

    def get_text(self) -> str | None:
        return self.content

    def set_text(self, text) -> None:
        self.content = text


class SystemMessage(pydantic.BaseModel):
    role: Literal["system"]
    content: str | None = None
    images: List[bytes] | None = None
    tool_calls: List[ToolCall] | None = None

    def get_content(self) -> Iterable[Any]:
        yield self

    def get_text(self) -> str | None:
        return self.content

    def set_text(self, text) -> None:
        self.content = text


class ToolMessage(pydantic.BaseModel):
    role: Literal["tool"]
    content: str | None = None
    images: List[bytes] | None = None
    tool_calls: List[ToolCall] | None = None

    def get_content(self) -> Iterable[Any]:
        yield self

    def get_text(self) -> str | None:
        return self.content

    def set_text(self, text) -> None:
        self.content = text


Message = Union[
    UserMessage,
    AssistantMessage,
    SystemMessage,
    ToolMessage,
]


class ChatRequest(pydantic.BaseModel):
    model: str
    messages: List[Message]
    stream: bool | None = True # see here https://github.com/ollama/ollama/blob/main/server/routes.go#L1529
    format: dict | None = None
    keep_alive: int | str | None = None
    tools: List[ToolDef] | None = None
    options: dict

    def get_stream(self) -> bool:
        return self.stream

    def get_model(self) -> str:
        return self.model

    def get_messages(self) -> Iterable[Message]:
        for msg in self.messages:
            yield msg

    def first_message(self) -> Message | None:
        return self.messages[0]

    def last_user_message(self) -> tuple[Message, int] | None:
        for idx, msg in enumerate(reversed(self.messages)):
            if isinstance(msg, UserMessage):
                return msg, len(self.messages) - 1 - idx

    def last_user_block(self) -> Iterable[tuple[Message, int]]:
        for idx, msg in enumerate(reversed(self.messages)):
            if isinstance(msg, UserMessage):
                yield msg, len(self.messages) - 1 - idx
                break

    def get_system_prompt(self) -> Iterable[str]:
        for msg in self.messages:
            if isinstance(msg, SystemMessage):
                yield msg.get_text()
                break # TODO this must be changed

    def set_system_prompt(self, text) -> None:
        for msg in self.messages:
            if isinstance(msg, SystemMessage):
                msg.set_text(text)
                break # TODO this does not make sense on multiple messages

    def add_system_prompt(self, text, sep="\n") -> None:
        self.messages.append(
            SystemMessage(
                role="system",
                content=text,
                name="codegate",
            )
        )

    def get_prompt(self, default=None):
        for message in self.messages:
            for content in message.get_content():
                for txt in content.get_text():
                    return txt
        return default


class GenerateRequest(pydantic.BaseModel):
    model: str
    prompt: str
    suffix: str | None = None
    system: str | None = None
    template: str | None = None
    context: List[int] | None = None
    stream: bool | None = True # see here https://github.com/ollama/ollama/blob/main/server/routes.go#L339
    raw: bool | None = None
    format: dict | None = None
    keep_alive: int | str | None = None
    images: List[bytes] | None = None
    options: dict

    def get_stream(self) -> bool:
        return self.stream

    def get_model(self) -> str:
        return self.model

    def get_messages(self) -> Iterable[Message]:
        yield self

    def get_content(self):
        yield self

    def get_text(self):
        return self.prompt

    def set_text(self, text):
        self.prompt = text

    def first_message(self) -> Message | None:
        return self

    def last_user_message(self) -> tuple[Message, int] | None:
        return self, 0

    def last_user_block(self) -> Iterable[tuple[Message, int]]:
        yield self, 0

    def get_system_prompt(self) -> Iterable[str]:
        yield self.system

    def set_system_prompt(self, text) -> None:
        self.system = text

    def add_system_prompt(self, text, sep="\n") -> None:
        self.system = f"{self.system}{sep}{text}"

    def get_prompt(self, default=None):
        if self.prompt is not None:
            return self.prompt
        return default
