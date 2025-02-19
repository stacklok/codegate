from typing import (
    Any,
    Iterable,
    List,
    Literal,
    Optional,
    Union,
)

import pydantic

from ._shared_models import ServiceTier  # TODO: openai seems to have a different ServiceTier model


class CompletionTokenDetails(pydantic.BaseModel):
    accepted_prediction_tokens: int | None = None
    audio_tokens: int | None = None
    reasoning_tokens: int | None = None
    rejected_prediction_tokens: int | None = None


class PromptTokenDetails(pydantic.BaseModel):
    audio_tokens: int | None = None
    cached_tokens: int | None = None


class Usage(pydantic.BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    completion_tokens_details: CompletionTokenDetails | None = None
    prompt_tokens_details: PromptTokenDetails | None = None


FinishReason = Union[
    Literal["stop"],
    Literal["length"],
    Literal["content_filter"],
    Literal["tool_calls"],
    Literal["function_call"], # deprecated
]


Role = Union[
    Literal["user"],
    Literal["developer"],
    Literal["assistant"],
    Literal["system"],
    Literal["tool"],
]


class RawLogProbsContent(pydantic.BaseModel):
    token: str
    logprob: float
    bytes: Optional[List[int]] = None


class LogProbsContent(pydantic.BaseModel):
    token: str
    logprob: float
    bytes: Optional[List[int]] = None
    top_logprobs: List[RawLogProbsContent]


class LogProbs(pydantic.BaseModel):
    content: List[LogProbsContent] | None = None
    refusal: List[LogProbsContent] | None = None


class FunctionCall(pydantic.BaseModel):
    name: str | None = None
    arguments: str | None = None


class ToolCall(pydantic.BaseModel):
    id: str | None = None
    type: Literal["function"] = "function"
    function: FunctionCall | None = None


class AudioMessage(pydantic.BaseModel):
    id: str
    expires_at: int
    data: str
    transcript: str


class Message(pydantic.BaseModel):
    content: str | None
    refusal: str | None
    tool_calls: List[ToolCall] | None = None
    role: str
    function_call: FunctionCall | None = None # deprecated
    audio: AudioMessage | None


class Choice(pydantic.BaseModel):
    finish_reason: FinishReason
    index: int
    message: Message
    logprobs: LogProbs | None = None

    def get_text(self) -> str | None:
        if self.message:
            return self.message.content

    def set_text(self, text) -> None:
        self.message.content = text


class MessageDelta(pydantic.BaseModel):
    content: str | None = None
    refusal: str | None = None
    tool_calls: List[ToolCall] | None = None
    role: Role | None = None
    function_call: FunctionCall | None = None # deprecated
    reasoning: str | None = None # openrouter extension


class ChoiceDelta(pydantic.BaseModel):
    finish_reason: FinishReason | None = None
    index: int
    # TODO: Copilot FIM seems to contain a "text" field only, no delta
    delta: MessageDelta
    logprobs: LogProbs | None = None

    def get_text(self) -> str | None:
        if self.delta:
            return self.delta.content

    def set_text(self, text: str) -> None:
        self.delta.content = text


class ChatCompletion(pydantic.BaseModel):
    id: str
    choices: List[Choice]
    created: int
    model: str
    service_tier: ServiceTier | None = None
    system_fingerprint: str
    object: Literal["chat.completion"] = "chat.completion"
    usage: Usage

    def get_content(self) -> Iterable[Choice]:
        for choice in self.choices:
            yield choice


class StreamingChatCompletion(pydantic.BaseModel):
    id: str
    choices: List[ChoiceDelta]
    created: int
    model: str
    service_tier: ServiceTier | None = None
    system_fingerprint: str | None = None
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    usage: Usage | None = None

    def get_content(self) -> Iterable[ChoiceDelta]:
        for choice in self.choices:
            yield choice

    def set_text(self, text) -> None:
        if self.choices:
            self.choices[0].set_text(text)


class ErrorDetails(pydantic.BaseModel):
    message: str
    code: int | str | None

    def get_text(self) -> str | None:
        return self.message

    def set_text(self, text) -> None:
        self.message = text


class MessageError(pydantic.BaseModel):
    error: ErrorDetails

    def get_content(self) -> Iterable[Any]:
        yield self.error

    def set_text(self, text) -> None:
        self.error.message = text


class VllmMessageError(pydantic.BaseModel):
    object: str
    message: str
    code: int

    def get_content(self) -> Iterable[Any]:
        yield self

    def get_text(self) -> str | None:
        return self.message

    def set_text(self, text) -> None:
        self.message = text
