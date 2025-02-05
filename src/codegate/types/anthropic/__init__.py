from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    TypedDict,
    Union,
)

from litellm import AnthropicExperimentalPassThroughConfig
from litellm.adapters.anthropic_adapter import (
    AnthropicAdapter as LitellmAnthropicAdapter,
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

from ._request_models import (
    AssistantMessage,
    CacheControl,
    ChatCompletionRequest,
    InputSchema,
    ResponseFormatJSON,
    ResponseFormatJSONSchema,
    ResponseFormatText,
    SystemPrompt,
    TextContent,
    ToolChoice,
    ToolDef,
    ToolResultContent,
    ToolUseContent,
    UserMessage,
)
