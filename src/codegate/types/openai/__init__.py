from ._generators import (
    completions_streaming,
    message_wrapper,
    stream_generator,
)

from ._response_models import (
    AudioMessage,
    ChatCompletion,
    Choice,
    ChoiceDelta,
    CompletionTokenDetails,
    ErrorDetails,
    FunctionCall,
    LogProbs,
    LogProbsContent,
    Message,
    MessageDelta,
    MessageError,
    PromptTokenDetails,
    RawLogProbsContent,
    StreamingChatCompletion,
    ToolCall,
    Usage,
)

from ._request_models import (
    AssistantMessage,
    Audio,
    AudioContent,
    ChatCompletionRequest,
    DeveloperMessage,
    FunctionChoice,
    FunctionDef,
    FunctionMessage,
    ImageContent,
    InputAudio,
    JsonSchema,
    LegacyFunctionDef,
    RefusalContent,
    ResponseFormat,
    StaticContent,
    StreamOption,
    SystemMessage,
    TextContent,
    ToolChoice,
    ToolDef,
    ToolMessage,
    URL,
    UserMessage,
)

from ._shared_models import (
    ServiceTier,
)

from ._legacy_models import (
    LegacyCompletionRequest,
    LegacyCompletionTokenDetails,
    LegacyPromptTokenDetails,
    LegacyUsage,
    LegacyLogProbs,
    LegacyMessage,
    LegacyCompletion,
)

from ._copilot import (
    CopilotCompletionRequest
)