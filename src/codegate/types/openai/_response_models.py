from typing import (
    List,
    Literal,
    Union,
)

import pydantic

from ._shared_models import ServiceTier


class CompletionTokenDetails(pydantic.BaseModel):
    accepted_prediction_tokens: int
    audio_tokens: int
    reasoning_tokens: int
    rejected_prediction_tokens: int


class PromptTokenDetails(pydantic.BaseModel):
    audio_tokens: int
    cached_tokens: int


class Usage(pydantic.BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    completion_tokens_details: CompletionTokenDetails
    prompt_tokens_details: PromptTokenDetails


FinishReason = Union[
    Literal["stop"],
    Literal["length"],
    Literal["content_filter"],
    Literal["tool_calls"],
    Literal["function_call"], # deprecated
]


class RawLogProbsContent(pydantic.BaseModel):
    token: str
    logprob: float
    bytes: str | None


class LogProbsContent(pydantic.BaseModel):
    token: str
    logprob: float
    bytes: str | None
    top_logprobs: List[RawLogProbsContent]


class LogProbs(pydantic.BaseModel):
    content: List[LogProbsContent] | None = None
    refusal: List[LogProbsContent] | None = None


class FunctionCall(pydantic.BaseModel):
    name: str
    arguments: dict | None


class ToolCall(pydantic.BaseModel):
    id: str
    type: Literal["function"]
    function: FunctionCall


class AudioMessage(pydantic.BaseModel):
    id: str
    expires_at: int
    data: str
    transcript: str


class Message(pydantic.BaseModel):
    content: str | None
    refusal: str | None
    tool_calls: List[ToolCall]
    role: str
    function_call: FunctionCall # deprecated
    audio: AudioMessage | None


class Choice(pydantic.BaseModel):
    finish_reason: FinishReason
    index: int
    message: Message
    logprobs: LogProbs | None = None


class MessageDelta(pydantic.BaseModel):
    content: str | None
    refusal: str | None
    tool_calls: List[ToolCall]
    role: str
    function_call: FunctionCall # deprecated


class ChoiceDelta(pydantic.BaseModel):
    finish_reason: FinishReason
    index: int
    delta: MessageDelta
    logprobs: LogProbs


class ChatCompletion(pydantic.BaseModel):
    id: str
    choices: List[Choice]
    created: int
    model: str
    service_tier: ServiceTier | None
    system_fingerprint: str
    object: Literal["chat.completion"]
    usage: Usage


class StreamingChatCompletion(pydantic.BaseModel):
    id: str
    choices: List[ChoiceDelta]
    created: int
    model: str
    service_tier: ServiceTier | None
    system_fingerprint: str
    object: Literal["chat.completion.chunk"]
    usage: Usage
