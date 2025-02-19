# VLLM types and generators are mainly a repackaging of OpenAI ones,
# except for a few types. To keep things simple, we repackage all used
# structs, but retain the right (and duty) to clone the structs in
# this package at the first signal of divergence.

from codegate.types.openai import (
    # generators
    completions_streaming,
    message_wrapper,
    stream_generator,
    # types
    AudioMessage,
    ChatCompletion,
    Choice,
    ChoiceDelta,
    CompletionTokenDetails,
    FunctionCall,
    LogProbs,
    LogProbsContent,
    Message,
    MessageDelta,
    PromptTokenDetails,
    RawLogProbsContent,
    StreamingChatCompletion,
    ToolCall,
    Usage,
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
    ServiceTier,
)

from ._response_models import (
    VllmMessageError,
)
