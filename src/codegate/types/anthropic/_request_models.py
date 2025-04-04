from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Union,
)

import pydantic

from codegate.types.common import MessageTypeFilter


class CacheControl(pydantic.BaseModel):
    type: Literal["ephemeral"]


class TextContent(pydantic.BaseModel):
    type: Literal["text"]
    text: str
    cache_control: CacheControl | None = None

    def get_text(self) -> str | None:
        return self.text

    def set_text(self, text) -> None:
        self.text = text


class ToolUseContent(pydantic.BaseModel):
    id: str
    input: dict
    name: str
    type: Literal["tool_use"]
    cache_control: CacheControl | None = None

    def get_text(self) -> str | None:
        return None

    def set_text(self, text) -> None:
        pass


class ToolResultContent(pydantic.BaseModel):
    tool_use_id: str
    type: Literal["tool_result"]
    content: str
    is_error: bool | None = False
    cache_control: CacheControl | None = None

    def get_text(self) -> str | None:
        return self.content

    def set_text(self, text) -> None:
        self.content = text


MessageContent = Union[
    TextContent,
    ToolUseContent,
    ToolResultContent,
]


class UserMessage(pydantic.BaseModel):
    role: Literal["user"]
    content: str | List[MessageContent]

    def get_text(self) -> Iterable[str]:
        if isinstance(self.content, str):
            return self.content

    def set_text(self, txt: str) -> None:
        if isinstance(self.content, str):
            self.content = txt
            return

        # should have been called on the content
        raise ValueError("Cannot set text on a list of content")

    def get_content(self) -> Iterable[MessageContent]:
        if isinstance(self.content, str):
            yield self
        else:  # list
            for content in self.content:
                yield content


class AssistantMessage(pydantic.BaseModel):
    role: Literal["assistant"]
    content: str | List[MessageContent]

    def get_text(self) -> Iterable[str]:
        if isinstance(self.content, str):
            return self.content

    def set_text(self, text) -> None:
        if isinstance(self.content, str):
            self.content = text
            return

        # should have been called on the content
        raise ValueError("Cannot set text on a list of content")

    def get_content(self) -> Iterable[MessageContent]:
        if isinstance(self.content, str):
            yield self
        else:  # list
            for content in self.content:
                yield content


Message = Union[
    UserMessage,
    AssistantMessage,
]


class ResponseFormatText(pydantic.BaseModel):
    type: str = "text"


class ResponseFormatJSON(pydantic.BaseModel):
    type: str = "json_object"


class ResponseFormatJSONSchema(pydantic.BaseModel):
    json_schema: Any
    type: str = "json_schema"


ResponseFormat = Union[
    ResponseFormatText,
    ResponseFormatJSON,
    ResponseFormatJSONSchema,
]


class SystemPrompt(pydantic.BaseModel):
    text: str
    type: Literal["text"]
    cache_control: CacheControl | None = None


class ToolDef(pydantic.BaseModel):
    name: str
    description: str | None = None
    cache_control: CacheControl | None = None
    type: Literal["custom"] | None = "custom"
    input_schema: Any | None


ToolChoiceType = Union[
    Literal["auto"],
    Literal["any"],
    Literal["tool"],
    Literal["none"],
]


class ToolChoice(pydantic.BaseModel):
    type: ToolChoiceType = "auto"
    name: str | None = None
    disable_parallel_tool_use: bool | None = False


class ThinkingEnabled(pydantic.BaseModel):
    type: Literal["enabled"]
    budget_tokens: int


class ThinkingDisabled(pydantic.BaseModel):
    type: Literal["disabled"]


class ChatCompletionRequest(pydantic.BaseModel):
    max_tokens: int
    messages: List[Message]
    model: str
    metadata: Dict | None = None
    stop_sequences: List[str] | None = None
    stream: bool = False
    system: Union[str, List[SystemPrompt]] | None = None
    temperature: float | None = None
    thinking: ThinkingEnabled | ThinkingDisabled | None = None
    tool_choice: ToolChoice | None = None
    tools: List[ToolDef] | None = None
    top_k: int | None = None
    top_p: Union[int, float] | None = None

    def get_stream(self) -> bool:
        return self.stream

    def get_model(self) -> str:
        return self.model

    def get_messages(self, filters: List[MessageTypeFilter] | None = None) -> Iterable[Message]:
        messages = self.messages
        if filters:
            types = set()
            if MessageTypeFilter.ASSISTANT in filters:
                types.add(AssistantMessage)
            if MessageTypeFilter.SYSTEM in filters:
                # This is a weird case, as system messages are not
                # present in the list of messages for
                # Anthropic. Throughout the codebase we should only
                # rely on `get_system_prompt`, `set_system_prompt`,
                # and `add_system_prompt`.
                pass
            if MessageTypeFilter.TOOL in filters:
                types.add(AssistantMessage)
            if MessageTypeFilter.USER in filters:
                types.add(UserMessage)
            messages = filter(lambda m: isinstance(m, tuple(types)), self.messages)
        for msg in messages:
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

    def get_system_prompt(self) -> Iterable[str]:
        if isinstance(self.system, str):
            yield self.system
        if isinstance(self.system, list):
            for sp in self.system:
                yield sp.text
                break  # TODO this must be changed

    def set_system_prompt(self, text) -> None:
        if isinstance(self.system, (str, type(None))):
            self.system = text
        if isinstance(self.system, list):
            self.system[0].text = text

    def add_system_prompt(self, text, sep="\n") -> None:
        if isinstance(self.system, type(None)):
            self.system = text
        if isinstance(self.system, str):
            self.system = f"{self.system}{sep}{text}"
        if isinstance(self.system, list):
            self.system.append(
                SystemPrompt(
                    text=text,
                    type="text",
                )
            )

    def get_prompt(self, default=None):
        for message in self.messages:
            for content in message.get_content():
                for txt in content.get_text():
                    return txt
        return default
