from litellm import AnthropicExperimentalPassThroughConfig
from litellm.adapters.anthropic_adapter import (
    AnthropicAdapter as LitellmAnthropicAdapter,
)
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    TypedDict,
    Union,
)

from litellm.adapters.anthropic_adapter import AnthropicStreamWrapper
from litellm.types.llms.anthropic import (
    AnthropicMessagesRequest,
    ContentBlockDelta,
    ContentBlockStart,
    ContentTextBlockDelta,
    MessageChunk,
    MessageStartBlock,
)

import pydantic

from codegate.clients.clients import ClientType


class CacheControl(pydantic.BaseModel):
    type: Literal["ephemeral"]


class TextContent(pydantic.BaseModel):
    type: Literal["text"]
    text: str
    cache_control: Optional[CacheControl] = None


class ToolUseContent(pydantic.BaseModel):
    id: str
    input: dict
    name: str
    type: Literal["tool_use"]
    cache_control: Optional[CacheControl] = None


class ToolResultContent(pydantic.BaseModel):
    tool_use_id: str
    type: Literal["tool_result"]
    content: str
    is_error: Optional[bool] = False
    cache_control: Optional[CacheControl] = None


MessageContent = Union[
    TextContent,
    ToolUseContent,
    ToolResultContent,
]


class UserMessage(pydantic.BaseModel):
    role: Literal["user"]
    content: str | List[MessageContent]

    def text(self) -> Iterable[str]:
        if isinstance(self.content, str):
            yield self.content
        else: # list
            for content in self.content:
                if isinstance(content, TextContent):
                    yield content.text
                # if isinstance(content, ToolResultContent):
                #     return content.input # this is an object
                if isinstance(content, ToolResultContent):
                    yield content.content


class AssistantMessage(pydantic.BaseModel):
    role: Literal["assistant"]
    content: str | List[MessageContent]

    def text(self) -> Iterable[str]:
        if isinstance(self.content, str):
            yield self.content
        else: # list
            for content in self.content:
                if isinstance(content, TextContent):
                    return content.text
                # if isinstance(content, ToolResultContent):
                #     return content.input # this is an object
                if isinstance(content, ToolResultContent):
                    return content.content


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
    type: Literal["text"] = "text"
    cache_control: Optional[CacheControl] = None


class InputSchema(pydantic.BaseModel):
    type: Literal["object"]
    properties: Optional[Dict] = None
    required: Optional[List[str]] = None


class ToolDef(pydantic.BaseModel):
    name: str
    description: Optional[str] = None
    cache_control: Optional[CacheControl] = None
    type: Optional[Literal["custom"]] = "custom"
    input_schema: Optional[InputSchema]


ToolChoiceType = Union[
    Literal["auto"],
    Literal["any"],
    Literal["tool"],
]


class ToolChoice(pydantic.BaseModel):
    type: ToolChoiceType = "auto"
    name: Optional[str] = None
    disable_parallel_tool_use: Optional[bool] = False


class ChatCompletionRequest(pydantic.BaseModel):
    max_tokens: int
    messages: List[Message]
    model: str
    metadata: Optional[Dict] = None
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    system: Optional[Union[str, SystemPrompt]] = None
    temperature: Optional[float] = None
    tool_choice: Optional[ToolChoice] = None
    tools: Optional[List[ToolDef]] = None
    top_k: Optional[int] = None
    top_p: Optional[Union[int, float]] = None

    def get_stream(self) -> bool:
        return self.stream

    def get_model(self) -> str:
        return self.model

    def first_message(self) -> Message | None:
        return self.messages[0]

    def last_user_message(self) -> Message | None:
        for idx, msg in enumerate(reversed(self.messages)):
            if isinstance(msg, UserMessage):
                return msg, len(self.messages) - idx

    def last_user_block(self) -> Iterable[Message] | None:
        for idx, msg in enumerate(reversed(self.messages)):
            if isinstance(msg, UserMessage):
                yield  msg, len(self.messages) - idx
            else:
                break

    def user_text(self) -> Iterable[str]:
        for msg in reversed(self.messages):
            if isinstance(msg, UserMessage):
                yield msg.text()
            else:
                return

    def all_text(self) -> Iterable[str]:
        for msg in self.messages:
            for txt in msg.text():
                yield txt

    def system_prompt(self) -> Iterable[str]:
        if isinstance(self.system, str):
            yield self.system
        if isinstance(self.system, SystemPrompt):
            yield self.system.text

    def set_system_prompt(self, text) -> None:
        if isinstance(self.system, str):
            self.system = text
        if isinstance(self.system, SystemPrompt):
            self.system.text = text

    def prompt(self, default=None):
        if default is not None:
            return default
        return None
