from litellm import AnthropicExperimentalPassThroughConfig
from litellm.adapters.anthropic_adapter import (
    AnthropicAdapter as LitellmAnthropicAdapter,
)
from typing import (
    Any,
    Dict,
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


class ToolCallFunction(TypedDict):
    name: str
    arguments: str


class ToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: ToolCallFunction


class Content(TypedDict):
    type: str
    text: str


class UserMessage(TypedDict):
    role: Literal["user"]
    content: Union[str, List[Content]]


class AssistantMessage(TypedDict):
    role: Literal["assistant"]
    content: Optional[Union[str, List]] = None


Message = Union[
    UserMessage,
    AssistantMessage,
]


class ResponseFormatText(TypedDict):
    type: str = "text"


class ResponseFormatJSON(TypedDict):
    type: str = "json_object"


class ResponseFormatJSONSchema(TypedDict):
    json_schema: Any
    type: str = "json_schema"


ResponseFormat = Union[
    ResponseFormatText,
    ResponseFormatJSON,
    ResponseFormatJSONSchema,
]


class FunctionDef(TypedDict):
    name: str
    description: Optional[str]
    parameters: Optional[Dict]
    strict: Optional[bool] = False


class ToolDef(TypedDict):
    function: FunctionDef
    type: str = "function"


class ToolChoiceFunction(TypedDict):
    name: str


class ToolChoice(TypedDict):
    function: ToolChoiceFunction
    type: str = "function"


class CodegateChatCompletionRequest(TypedDict):
    max_tokens: int
    messages: List[Message]
    model: str
    metadata: Optional[Dict] = None
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    system: Optional[str] = None
    tool_choice: Optional[Dict] = None
    tools: Optional[List[Dict]] = None
    top_k: Optional[int] = None
    top_p: Optional[Union[int,float]] = None


ChatCompletionRequest = CodegateChatCompletionRequest


class FunctionCall(TypedDict):
    name: str
    arguments: str


class Delta(TypedDict):
    content: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    tool_calls: Optional[List[ToolCall]] = None
    role: Optional[str] = None
    refusal: Optional[str] = None


# TODO verify whether it's needed
# class LogProbs(TypedDict):
#     content: Optional[Iterable] = None
#     refusal: Optional[Iterable] = None


class Choice(TypedDict):
    delta: Delta
    index: int
    # logprobs: Optional[LogProbs] = None
    finish_reason: Optional[str] = None


class Usage(TypedDict):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class ChatCompletionChunk(TypedDict):
    id: str
    choices: List[Choice]
    created: int
    model: str
    system_fingerprint: str
    object: str = "chat.completion.chunk"
    service_tier: Optional[str] = None
    usage: Optional[Usage] = None
