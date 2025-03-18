import pydantic
import pytest

import codegate.types.ollama as ollama
import codegate.types.openai as openai
from codegate.muxing.ollama_mappers import ollama_chat_from_openai


@pytest.fixture
def base_request():
    return openai.ChatCompletionRequest(model="gpt-4", messages=[], stream=True)


def test_convert_user_message(base_request):
    base_request.messages = [
        openai.UserMessage(role="user", content=[openai.TextContent(type="text", text="Hello")])
    ]

    result = ollama_chat_from_openai(base_request)

    assert isinstance(result, ollama.ChatRequest)
    assert len(result.messages) == 1
    assert isinstance(result.messages[0], ollama.UserMessage)
    assert result.messages[0].role == "user"
    assert result.messages[0].content == "Hello"


def test_convert_system_message(base_request):
    base_request.messages = [
        openai.SystemMessage(
            role="system", content=[openai.TextContent(type="text", text="System prompt")]
        )
    ]

    result = ollama_chat_from_openai(base_request)

    assert isinstance(result, ollama.ChatRequest)
    assert len(result.messages) == 1
    assert isinstance(result.messages[0], ollama.SystemMessage)
    assert result.messages[0].role == "system"
    assert result.messages[0].content == "System prompt"


def test_convert_developer_message(base_request):
    base_request.messages = [
        openai.DeveloperMessage(
            role="developer", content=[openai.TextContent(type="text", text="Developer info")]
        )
    ]

    result = ollama_chat_from_openai(base_request)

    assert isinstance(result, ollama.ChatRequest)
    assert len(result.messages) == 1
    assert isinstance(result.messages[0], ollama.SystemMessage)
    assert result.messages[0].role == "system"
    assert result.messages[0].content == "Developer info"


def test_convert_assistant_message(base_request):
    base_request.messages = [
        openai.AssistantMessage(
            role="assistant", content=[openai.TextContent(type="text", text="Assistant response")]
        )
    ]

    result = ollama_chat_from_openai(base_request)

    assert isinstance(result, ollama.ChatRequest)
    assert len(result.messages) == 1
    assert isinstance(result.messages[0], ollama.AssistantMessage)
    assert result.messages[0].role == "assistant"
    assert result.messages[0].content == "Assistant response"


def test_convert_tool_message(base_request):
    base_request.messages = [
        openai.ToolMessage(
            role="tool",
            content=[openai.TextContent(type="text", text="Tool output")],
            tool_call_id="mock-tool-id",
        )
    ]

    result = ollama_chat_from_openai(base_request)

    assert isinstance(result, ollama.ChatRequest)
    assert len(result.messages) == 1
    assert isinstance(result.messages[0], ollama.ToolMessage)
    assert result.messages[0].role == "tool"
    assert result.messages[0].content == "Tool output"


def test_convert_multiple_content_items(base_request):
    base_request.messages = [
        openai.UserMessage(
            role="user",
            content=[
                openai.TextContent(type="text", text="Hello"),
                openai.TextContent(type="text", text="World"),
            ],
        )
    ]

    result = ollama_chat_from_openai(base_request)

    assert isinstance(result, ollama.ChatRequest)
    assert len(result.messages) == 1
    assert isinstance(result.messages[0], ollama.UserMessage)
    assert result.messages[0].content == "Hello World"


def test_convert_complete_conversation(base_request):
    base_request.messages = [
        openai.SystemMessage(
            role="system", content=[openai.TextContent(type="text", text="System prompt")]
        ),
        openai.UserMessage(
            role="user", content=[openai.TextContent(type="text", text="User message")]
        ),
        openai.AssistantMessage(
            role="assistant", content=[openai.TextContent(type="text", text="Assistant response")]
        ),
    ]

    result = ollama_chat_from_openai(base_request)

    assert isinstance(result, ollama.ChatRequest)
    assert result.model == "gpt-4"
    assert result.stream is True
    assert len(result.messages) == 3

    assert isinstance(result.messages[0], ollama.SystemMessage)
    assert result.messages[0].content == "System prompt"

    assert isinstance(result.messages[1], ollama.UserMessage)
    assert result.messages[1].content == "User message"

    assert isinstance(result.messages[2], ollama.AssistantMessage)
    assert result.messages[2].content == "Assistant response"


def test_convert_empty_messages(base_request):
    base_request.messages = []
    result = ollama_chat_from_openai(base_request)
    assert isinstance(result, ollama.ChatRequest)
    assert len(result.messages) == 0


def test_convert_default_stream(base_request):
    base_request.stream = None
    result = ollama_chat_from_openai(base_request)
    assert result.stream is True


def test_convert_response_format_json_object(base_request):
    base_request.response_format = openai.ResponseFormat(type="json_object")
    result = ollama_chat_from_openai(base_request)
    assert result.format == "json"


def test_convert_response_format_json_schema(base_request):
    base_request.response_format = openai.ResponseFormat(
        type="json_schema",
        json_schema=openai.JsonSchema(
            name="TestSchema",
            description="Test schema description",
            schema={"name": {"type": "string"}},
        ),
    )
    result = ollama_chat_from_openai(base_request)
    assert result.format == {"name": {"type": "string"}}


def test_convert_request_with_tools(base_request):
    base_request.tools = [
        openai.ToolDef(
            type="function",
            function=openai.FunctionDef(
                name="test_function",
                description="Test function description",
                parameters={
                    "type": "object",
                    "required": ["param1"],
                    "properties": {"param1": {"type": "string", "description": "Test parameter"}},
                },
            ),
        )
    ]

    result = ollama_chat_from_openai(base_request)

    assert result.tools is not None
    assert len(result.tools) == 1
    assert result.tools[0].type == "function"
    assert result.tools[0].function.name == "test_function"
    assert result.tools[0].function.description == "Test function description"
    assert result.tools[0].function.parameters.type == "object"
    assert result.tools[0].function.parameters.required == ["param1"]
    assert "param1" in result.tools[0].function.parameters.properties


def test_convert_request_with_options(base_request):
    base_request.max_tokens = 100
    base_request.stop = ["stop1", "stop2"]
    base_request.seed = 42

    result = ollama_chat_from_openai(base_request)

    assert result.options["num_predict"] == 100
    assert result.options["stop"] == ["stop1", "stop2"]
    assert result.options["seed"] == 42


def test_convert_request_with_single_stop(base_request):
    base_request.stop = "stop1"
    result = ollama_chat_from_openai(base_request)
    assert result.options["stop"] == ["stop1"]


def test_convert_request_with_max_completion_tokens(base_request):
    base_request.max_completion_tokens = 200
    result = ollama_chat_from_openai(base_request)
    assert result.options["num_predict"] == 200


class UnsupportedMessage(openai.Message):
    role: str = "unsupported"


def test_convert_unsupported_message_type(base_request):
    class UnsupportedMessage(pydantic.BaseModel):
        role: str = "unsupported"
        content: str

        def get_content(self):
            yield self

        def get_text(self):
            return self.content

    base_request.messages = [UnsupportedMessage(role="unsupported", content="Unsupported message")]

    with pytest.raises(ValueError, match="Unsupported message type:.*"):
        ollama_chat_from_openai(base_request)
