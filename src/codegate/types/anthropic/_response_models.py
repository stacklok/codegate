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

import pydantic


##### Batch Messages #####


class TextResponseContent(pydantic.BaseModel):
    type: Literal["text"]
    text: str

    def get_text(self):
        yield self.text


class ToolUseResponseContent(pydantic.BaseModel):
    type: Literal["tool_use"]
    id: str
    input: Any
    name: str

    def get_text(self):
        return iter(()) # empty generator


ResponseContent = Union[
    TextResponseContent,
    ToolUseResponseContent,
]


StopReason = Union[
    Literal["end_turn"],
    Literal["max_tokens"],
    Literal["stop_sequence"],
    Literal["tool_use"],
]


class Usage(pydantic.BaseModel):
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class Message(pydantic.BaseModel):
    type: Literal["message"]
    content: Iterable[ResponseContent]
    id: str
    model: str
    role: Literal["assistant"]
    stop_reason: StopReason | None
    stop_sequence: str | None
    usage: Usage

    def get_content(self):
        for content in self.content:
            yield content


##### Streaming Messages #####


class TextDelta(pydantic.BaseModel):
    # NOTE: it might be better to split these in two distinct classes
    type: Literal["text"] | Literal["text_delta"]
    text: str

    def get_text(self):
        return self.text

    def set_text(self, text):
        self.text = text


class ToolUse(pydantic.BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict

    def get_text(self):
        return ""

    def set_text(self, text):
        print(f"THIS IS WEIRD: {text}")
        pass


class InputJsonDelta(pydantic.BaseModel):
    type: Literal["input_json_delta"]
    partial_json: str

    def get_text(self):
        return self.partial_json

    def set_text(self, text):
        print(f"THIS IS WEIRD: {text}")
        pass


##### Streaming Messages: Content Blocks #####


class ContentBlockStart(pydantic.BaseModel):
    type: Literal["content_block_start"]
    index: int
    content_block: TextDelta | ToolUse

    def get_content(self):
        yield self.content_block


class ContentBlockDelta(pydantic.BaseModel):
    type: Literal["content_block_delta"]
    index: int
    delta: TextDelta | InputJsonDelta

    def get_content(self):
        yield self.delta

    def set_text(self, text):
        self.delta.set_text(text)


class ContentBlockStop(pydantic.BaseModel):
    type: Literal["content_block_stop"]
    index: int

    def get_content(self):
        return iter(()) # empty generator


ContentBlock = Union[
    ContentBlockStart,
    ContentBlockDelta,
    ContentBlockStop,
]


##### Streaming Messages: Message Types #####


class MessageStart(pydantic.BaseModel):
    type: Literal["message_start"]
    message: Message

    def get_content(self) -> Iterable[Any]:
        return self.message.get_content()


class LimitedMessage(pydantic.BaseModel):
    stop_reason: StopReason | None
    stop_sequence: str | None


class MessageDelta(pydantic.BaseModel):
    type: Literal["message_delta"]
    delta: LimitedMessage
    usage: Usage

    def get_content(self) -> Iterable[Any]:
        return iter(()) # empty generator


class MessageStop(pydantic.BaseModel):
    type: Literal["message_stop"]

    def get_content(self) -> Iterable[Any]:
        return iter(()) # empty generator


##### Streaming Messages: others #####


class MessagePing(pydantic.BaseModel):
    type: Literal["ping"]

    def get_content(self) -> Iterable[Any]:
        return iter(()) # empty generator


# Anthropic’s API is temporarily overloaded. (HTTP 529)
class OverloadedError(pydantic.BaseModel):
    type: Literal["overloaded_error"]
    message: str


# There was an issue with the format or content of your request. We
# may also use this error type for other 4XX status codes not listed
# below. (HTTP 400)
class InvalidRequestError(pydantic.BaseModel):
    type: Literal["invalid_request_error"]
    message: str


# There’s an issue with your API key. (HTTP 401)
class AuthenticationError(pydantic.BaseModel):
    type: Literal["authentication_error"]
    message: str


# Your API key does not have permission to use the specified
# resource. (HTTP 403)
class PermissionError(pydantic.BaseModel):
    type: Literal["permission_error"]
    message: str


# The requested resource was not found. (HTTP 404)
class NotFoundError(pydantic.BaseModel):
    type: Literal["not_found_error"]
    message: str


# Request exceeds the maximum allowed number of bytes. (HTTP 413)
class RequestTooLargeError(pydantic.BaseModel):
    type: Literal["request_too_large"]
    message: str


# Your account has hit a rate limit. (HTTP 429)
class RateLimitError(pydantic.BaseModel):
    type: Literal["rate_limit_error"]
    message: str


# An unexpected error has occurred internal to Anthropic’s
# systems. (HTTP 500)
class ApiError(pydantic.BaseModel):
    type: Literal["api_error"]
    message: str


Error = Union[
    OverloadedError,
    InvalidRequestError,
    AuthenticationError,
    PermissionError,
    NotFoundError,
    RequestTooLargeError,
    RateLimitError,
    ApiError,
]


class MessageError(pydantic.BaseModel):
    type: Literal["error"]
    error: Error

    def get_content(self) -> Iterable[Any]:
        return iter(()) # empty generator
