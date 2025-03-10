from typing import (
    Any,
    Iterable,
    List,
    Literal,
    Union,
)

import pydantic

from codegate.types.common import MessageTypeFilter
from ._shared_models import ServiceTier


class FunctionCall(pydantic.BaseModel):
    name: str
    arguments: str


class ToolCall(pydantic.BaseModel):
    type: Literal["function"]
    id: str
    function: FunctionCall

    def get_text(self) -> str | None:
        return self.function.arguments

    def set_text(self, text) -> None:
        self.function.arguments = text


class LegacyFunctionDef(pydantic.BaseModel):
    name: str
    description: str | None = None
    parameters: dict | None = None


class FunctionChoice(pydantic.BaseModel):
    name: str


class ToolChoice(pydantic.BaseModel):
    type: Literal["function"]
    function: FunctionChoice


ToolChoiceStr = Union[
    Literal["none"],
    Literal["auto"],
    Literal["required"],
]


class FunctionDef(pydantic.BaseModel):
    name: str
    description: str | None = None
    parameters: dict | None = None
    strict: bool | None = False


class ToolDef(pydantic.BaseModel):
    type: Literal["function"]
    function: FunctionDef


class StreamOption(pydantic.BaseModel):
    include_usage: bool | None = None


ResponseFormatType = Union[
    Literal["text"],
    Literal["json_object"],
    Literal["json_schema"],
]


class JsonSchema(pydantic.BaseModel):
    name: str
    description: str | None = None
    schema: dict | None = None
    strict: bool | None = False


class ResponseFormat(pydantic.BaseModel):
    type: ResponseFormatType
    json_schema: JsonSchema | None = None


class TextContent(pydantic.BaseModel):
    type: str
    text: str

    def get_text(self) -> str | None:
        return self.text

    def set_text(self, text) -> None:
        self.text = text


class URL(pydantic.BaseModel):
    url: str
    detail: str | None = "auto"


class ImageContent(pydantic.BaseModel):
    type: str
    image_url: URL

    def get_text(self) -> str | None:
        return None


class InputAudio(pydantic.BaseModel):
    data: str
    format: Literal["wav"] | Literal["mp3"]


class AudioContent(pydantic.BaseModel):
    type: Literal["input_audio"]
    input_audio: InputAudio

    def get_text(self) -> str | None:
        return None


class RefusalContent(pydantic.BaseModel):
    type: Literal["refusal"]
    refusal: str

    def get_text(self) -> str | None:
        return self.refusal

    def set_text(self, text) -> None:
        self.refusal = text


Content = Union[
    TextContent,
    ImageContent,
    AudioContent,
    RefusalContent,
]


AudioVoice = Union[
    Literal["ash"],
    Literal["ballad"],
    Literal["coral"],
    Literal["sage"],
    Literal["verse"],
    Literal["alloy"],
    Literal["echo"],
    Literal["shimmer"],
]


AudioFormat = Union[
    Literal["wav"],
    Literal["mp3"],
    Literal["flac"],
    Literal["opus"],
    Literal["pcm16"],
]


class Audio(pydantic.BaseModel):
    voice: AudioVoice
    format: AudioFormat


class StaticContent(pydantic.BaseModel):
    type: str
    content: str | List[TextContent]


class DeveloperMessage(pydantic.BaseModel):
    role: Literal["developer"]
    content: str | List[Content]
    name: str | None = None

    def get_text(self) -> Iterable[str]:
        if isinstance(self.content, str):
            return self.content

    def set_text(self, text) -> None:
        if isinstance(self.content, str):
            self.content = text
        # TODO we should probably return an error otherwise

    def get_content(self):
        if isinstance(self.content, str):
            yield self
        else:  # list
            for content in self.content:
                yield content


class SystemMessage(pydantic.BaseModel):
    role: Literal["system"]
    content: str | List[Content]
    name: str | None = None

    def get_text(self) -> Iterable[str]:
        if isinstance(self.content, str):
            return self.content

    def set_text(self, text) -> None:
        if isinstance(self.content, str):
            self.content = text
        # TODO we should probably return an error otherwise

    def get_content(self):
        if isinstance(self.content, str):
            yield self
        else:  # list
            for content in self.content:
                yield content


class UserMessage(pydantic.BaseModel):
    role: Literal["user"]
    content: str | List[Content]
    name: str | None = None

    def get_text(self) -> Iterable[str]:
        if isinstance(self.content, str):
            return self.content

    def set_text(self, text) -> None:
        if isinstance(self.content, str):
            self.content = text
        # TODO we should probably return an error otherwise

    def get_content(self):
        if isinstance(self.content, str):
            yield self
        else:  # list
            for content in self.content:
                yield content


class AssistantMessage(pydantic.BaseModel):
    role: Literal["assistant"]
    content: str | List[TextContent | RefusalContent] | None = None
    refusal: str | None = None
    name: str | None = None
    audio: dict | None = None
    tool_calls: List[ToolCall] | None = None
    function_call: Any | None = None

    def get_text(self) -> Iterable[str]:
        if isinstance(self.content, str):
            return self.content

    def set_text(self, text) -> None:
        self.content = text

    def get_content(self):
        if self.content:
            if isinstance(self.content, str):
                yield self
            elif self.content:  # list
                for content in self.content:
                    yield content
        # According to OpenAI documentation, an assistant message can
        # have `tool_calls` populated _iff_ content is empty.
        elif self.tool_calls:
            for tc in self.tool_calls:
                yield tc


class ToolMessage(pydantic.BaseModel):
    role: Literal["tool"]
    content: str | List[Any]
    tool_call_id: str

    def get_text(self) -> Iterable[str]:
        if isinstance(self.content, str):
            return self.content

    def set_text(self, text) -> None:
        self.content = text

    def get_content(self):
        if isinstance(self.content, str):
            yield self
        else:  # list
            for content in self.content:
                yield content


class FunctionMessage(pydantic.BaseModel):
    role: Literal["function"]
    content: str | None
    name: str

    def get_text(self) -> Iterable[str]:
        return self.content

    def get_content(self):
        yield self


Message = Union[
    DeveloperMessage,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    FunctionMessage,
]


class ChatCompletionRequest(pydantic.BaseModel):
    messages: List[Message]
    prompt: str | None = None  # deprecated
    model: str
    store: bool | None = False
    reasoning_effort: Literal["low"] | Literal["medium"] | Literal["high"] | None = None
    metadata: dict | None = None
    frequency_penalty: float | None = 0.0
    logit_bias: dict | None = None
    logprobs: int | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    n: int | None = None
    modalities: List[str] | None = ["text"]
    prediction: StaticContent | None = None
    audio: Audio | None = None
    presence_penalty: float | None = 0.0
    response_format: ResponseFormat | None = None
    seed: int | None = None
    service_tier: ServiceTier | None = "auto"
    stop: str | List[Any] | None = None
    stream: bool | None = False
    stream_options: StreamOption | None = None
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    tools: List[ToolDef] | None = None
    tool_choice: str | ToolChoice | None = "auto"
    parallel_tool_calls: bool | None = True
    user: str | None = None
    function_call: str | FunctionChoice | None = "auto"  # deprecated
    functions: List[LegacyFunctionDef] | None = None  # deprecated
    include_reasoning: bool | None = None  # openrouter extension

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
                types.Add(SystemMessage)
            if MessageTypeFilter.TOOL in filters:
                types.add(ToolMessage)
                types.add(FunctionMessage)  # unsure about this
            if MessageTypeFilter.USER in filters:
                types.add(UserMessage)
                types.add(DeveloperMessage)  # unsure about this
            messages = filter(lambda m: isinstance(m, tuple(types)), self.messages)
        for msg in messages:
            yield msg

    def first_message(self) -> Message | None:
        return self.messages[0] if len(self.messages) > 0 else None

    def last_user_message(self) -> tuple[Message, int] | None:
        for idx, msg in enumerate(reversed(self.messages)):
            if isinstance(msg, UserMessage):
                return msg, len(self.messages) - 1 - idx

    def last_user_block(self) -> Iterable[tuple[Message, int]]:
        for idx, msg in enumerate(reversed(self.messages)):
            if isinstance(msg, (UserMessage, ToolMessage)):
                yield msg, len(self.messages) - 1 - idx
            elif isinstance(msg, (SystemMessage, DeveloperMessage)):
                # these can occur in the middle of a user block
                continue
            elif isinstance(msg, (AssistantMessage, FunctionMessage)):
                # these are LLM responses, end of user input, break on them
                break

    def get_system_prompt(self) -> Iterable[str]:
        for msg in self.messages:
            if isinstance(msg, SystemMessage):
                yield msg.get_text()
                break  # TODO this must be changed

    def set_system_prompt(self, text) -> None:
        for msg in self.messages:
            if isinstance(msg, SystemMessage):
                msg.set_text(text)

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
                return content.get_text()
        return default
