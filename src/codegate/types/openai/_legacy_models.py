from typing import (
    Any,
    Iterable,
    List,
    Literal, Dict,
)

import pydantic

from ._response_models import (
    Usage,
)
from ._request_models import (
    StreamOption,
    Message,
)


class LegacyCompletionRequest(pydantic.BaseModel):
    prompt: str | None = None
    model: str
    best_of: int | None = 1
    echo: bool | None = False
    frequency_pentalty: float | None = 0.0
    logit_bias: dict | None = None
    logprobs: int | None = None
    max_tokens: int | None = None
    n: int | None = None
    presence_penalty: float | None = 0.0
    seed: int | None = None
    stop: str | List[Any] | None = None
    stream: bool | None = False
    stream_options: StreamOption | None = None
    suffix: str | None = None
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    user: str | None = None
    nwo: str | None = None # Copilot specific
    extra: Dict[str, Any] | None = None # Copilot specific

    def get_stream(self) -> bool:
        return self.stream

    def get_model(self) -> str:
        return self.model

    def get_messages(self) -> Iterable[Message]:
        yield self

    def get_content(self) -> Iterable[Any]:
        yield self

    def get_text(self) -> str | None:
        return self.prompt

    def set_text(self, text) -> None:
        self.prompt = text

    def first_message(self) -> Message | None:
        return self

    def last_user_message(self) -> tuple[Message, int] | None:
        return self, 0

    def last_user_block(self) -> Iterable[tuple[Message, int]]:
        yield self, 0

    def get_system_prompt(self) -> Iterable[str]:
        yield self.get_text()

    def set_system_prompt(self, text) -> None:
        self.set_text(text)

    def add_system_prompt(self, text, sep="\n") -> None:
        original = self.get_text()
        self.set_text(f"{original}{sep}{text}")

    def get_prompt(self, default=None):
        if self.prompt is not None:
            return self.get_text()
        return default


class LegacyCompletionTokenDetails(pydantic.BaseModel):
    accepted_prediction_tokens: int
    audio_tokens: int
    reasoning_tokens: int


class LegacyPromptTokenDetails(pydantic.BaseModel):
    audio_tokens: int
    cached_tokens: int


class LegacyUsage(pydantic.BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    completion_tokens_details: LegacyCompletionTokenDetails | None = None
    prompt_tokens_details: LegacyPromptTokenDetails | None = None


class LegacyLogProbs(pydantic.BaseModel):
    text_offset: List[Any]
    token_logprobs: List[Any]
    tokens: List[Any]
    top_logprobs: List[Any]


class LegacyMessage(pydantic.BaseModel):
    text: str
    finish_reason: str | None = None
    index: int = 0
    logprobs: LegacyLogProbs | None = None

    def get_text(self) -> str | None:
        return self.text

    def set_text(self, text) -> None:
        self.text = text


class LegacyCompletion(pydantic.BaseModel):
    id: str
    choices: List[LegacyMessage]
    created: int
    model: str
    system_fingerprint: str | None = None
    object: Literal["text_completion"] = "text_completion"
    usage: Usage | None = None

    def get_content(self) -> Iterable[LegacyMessage]:
        for message in self.choices:
            yield message

    def set_text(self, text) -> None:
        if self.choices:
            self.choices[0].set_text(text)
