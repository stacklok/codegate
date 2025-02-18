from unittest.mock import AsyncMock, Mock, patch

import pytest

from codegate.clients.clients import ClientType
from codegate.extract_snippets.message_extractor import CodeSnippet
from codegate.pipeline.base import PipelineContext
from codegate.pipeline.codegate_context_retriever.codegate import CodegateContextRetriever
from codegate.storage.storage_engine import StorageEngine
from codegate.types.anthropic import AssistantMessage as AnthropicAssistantMessage
from codegate.types.anthropic import ChatCompletionRequest as AnthropicChatCompletionRequest
from codegate.types.anthropic import ToolResultContent as AnthropicToolResultContent
from codegate.types.anthropic import ToolUseContent as AnthropicToolUseContent
from codegate.types.anthropic import UserMessage as AnthropicUserMessage
from codegate.types.openai import (
    AssistantMessage as OpenaiAssistantMessage,
)
from codegate.types.openai import (
    ChatCompletionRequest as OpenaiChatCompletionRequest,
)
from codegate.types.openai import (
    ToolMessage as OpenaiToolMessage,
)
from codegate.types.openai import (
    UserMessage as OpenaiUserMessage,
)
from codegate.utils.package_extractor import PackageExtractor


class TestCodegateContextRetriever:
    @pytest.fixture
    def mock_storage_engine(self):
        return Mock(spec=StorageEngine)

    @pytest.fixture
    def mock_package_extractor(self):
        return Mock(spec=PackageExtractor)

    @pytest.fixture
    def mock_context(self):
        context = Mock(spec=PipelineContext)
        context.client = ClientType.GENERIC
        return context

    @pytest.fixture
    def mock_cline_context(self):
        context = Mock(spec=PipelineContext)
        context.client = ClientType.CLINE
        return context

    def test_init_default(self):
        """Test initialization with default dependencies"""
        retriever = CodegateContextRetriever()
        assert isinstance(retriever.storage_engine, StorageEngine)
        assert retriever.package_extractor == PackageExtractor

    def test_init_with_dependencies(self, mock_storage_engine, mock_package_extractor):
        """Test initialization with custom dependencies"""
        retriever = CodegateContextRetriever(
            storage_engine=mock_storage_engine,
            package_extractor=mock_package_extractor
        )
        assert retriever.storage_engine == mock_storage_engine
        assert retriever.package_extractor == mock_package_extractor

    def test_name_property(self):
        """Test the name property returns the correct value"""
        retriever = CodegateContextRetriever()
        assert retriever.name == "codegate-context-retriever"

    @pytest.mark.asyncio
    async def test_process_no_bad_packages(self, mock_storage_engine, mock_context):
        """Test processing when no bad packages are found"""
        retriever = CodegateContextRetriever(storage_engine=mock_storage_engine)
        mock_storage_engine.search = AsyncMock(return_value=[])

        request = OpenaiChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Test message"}]
        )

        result = await retriever.process(request, mock_context)
        assert result.request == request
        assert mock_storage_engine.search.call_count > 0

    @pytest.mark.asyncio
    async def test_process_with_code_snippets(
            self,
            mock_storage_engine,
            mock_package_extractor,
            mock_context,
    ):
        """Test processing with bad packages found in code snippets"""
        retriever = CodegateContextRetriever(
            storage_engine=mock_storage_engine,
            package_extractor=mock_package_extractor
        )

        mock_package_extractor.extract_packages = Mock(return_value=["malicious-package"])

        bad_package = {
            "properties": {
                "name": "malicious-package",
                "type": "npm",
                "status": "malicious",
                "description": "This package is bad mojo",
            }
        }

        # Mock storage engine to return bad package only on first call
        mock_search = AsyncMock()
        # First call returns bad package, subsequent calls return empty list
        mock_search.side_effect = [[bad_package], []]
        mock_storage_engine.search = mock_search

        with patch("codegate.extract_snippets.factory.MessageCodeExtractorFactory.create_snippet_extractor") as mock_factory: # noqa
            mock_extractor = Mock()
            mock_extractor.extract_snippets = Mock(return_value=[
                CodeSnippet(
                    code="const pkg = require('malicious-package')",
                    language="javascript",
                    filepath="test.js"
                )
            ])
            mock_factory.return_value = mock_extractor

            request = OpenaiChatCompletionRequest(
                model="test-model",
                messages=[{
                    "role": "user",
                    "content": "<task>Install package</task>\n```javascript\nconst pkg = require('malicious-package')\n```" # noqa
                }]
            )

            result = await retriever.process(request, mock_context)

            assert "malicious-package" in result.request.messages[0].content
            # Verify search was called at least twice (once for snippets, once for text)
            assert mock_storage_engine.search.call_count >= 2
            # Verify only one alert was added (from the snippet search only)
            assert mock_context.add_alert.call_count == 1

    @pytest.mark.asyncio
    async def test_process_with_text_matches_cline(self, mock_storage_engine, mock_cline_context):
        """Test processing with bad packages found in regular text"""
        retriever = CodegateContextRetriever(storage_engine=mock_storage_engine)

        bad_package = {
            "properties": {
                "name": "evil-package",
                "type": "pip",
                "status": "malicious",
                "description": "This package is bad mojo",
            }
        }
        mock_storage_engine.search = AsyncMock(return_value=[bad_package])

        request = OpenaiChatCompletionRequest(
            model="test-model",
            messages=[{
                "role": "user",
                "content": "<task>Should I use the evil-package package?</task>"
            }]
        )

        result = await retriever.process(request, mock_cline_context)

        assert "This package is bad mojo" in result.request.messages[0].content
        assert mock_cline_context.add_alert.call_count == 1

    @pytest.mark.asyncio
    async def test_bad_pkg_in_openai_tool_call(self, mock_storage_engine, mock_context):
        """Test that bad package is found in openai tool call"""
        retriever = CodegateContextRetriever(storage_engine=mock_storage_engine)

        bad_packages = [
            {
                "properties": {
                    "name": "mal-package-1",
                    "type": "npm",
                    "status": "malicious",
                    "description": "This package is mal-1",
                },
            },
        ]
        mock_storage_engine.search = AsyncMock(return_value=bad_packages)

        request = OpenaiChatCompletionRequest(
            model="test-model",
            messages=[
                OpenaiUserMessage(
                    content="Evaluate packages in requirements.txt",
                    role="user",
                ),
                OpenaiAssistantMessage(
                    role="assistant",
                    tool_calls=[
                        {"id": "tool-1",
                         "type": "function",
                         "index": 0,
                         "function": {
                             "name:": "read_file",
                             "args": {"file_path": "requirements.txt"}},
                         },
                    ]),
                OpenaiToolMessage(
                    role="tool",
                    content="mal-package-1",
                    tool_call_id="call_XnHqU5AiAzCzRpNY9rGrOEs4",
                ),
            ],
        )

        result = await retriever.process(request, mock_context)

        # Verify storage engine was called with the correct package name
        mock_storage_engine.search.assert_called_with(query="mal-package-1", distance=0.5, limit=100)
        # verify the tool message was augmented with the package description
        assert "This package is mal-1" in result.request.messages[2].content
        assert mock_context.add_alert.call_count == 1

    @pytest.mark.asyncio
    async def test_bad_pkg_in_anthropic_tool_call(self, mock_storage_engine, mock_context):
        """
        Test that bad package is found in anthropic tool call

        The point is really that ToolUseContent returns None for get_text
        """
        retriever = CodegateContextRetriever(storage_engine=mock_storage_engine)

        bad_packages = [
            {
                "properties": {
                    "name": "archived-package-1",
                    "type": "npm",
                    "status": "archived",
                    "description": "This package is archived-1",
                },
            },
        ]
        mock_storage_engine.search = AsyncMock(return_value=bad_packages)

        request = AnthropicChatCompletionRequest(
            model="test-model",
            max_tokens=100,
            messages=[
                AnthropicUserMessage(
                    role="user",
                    content="Evaluate packages in requirements.txt",
                ),
                AnthropicAssistantMessage(
                    role="assistant",
                    content=[
                        AnthropicToolUseContent(
                            type="tool_use",
                            id="toolu_01CPkkQC53idEC89daHDEvPt",
                            input={
                                "filepath": "requirements.txt",
                            },
                            name="builtin_read_file",
                        ),
                    ],
                ),
                AnthropicUserMessage(
                    role="user",
                    content=[
                        AnthropicToolResultContent(
                            type="tool_result",
                            tool_use_id="toolu_01CPkkQC53idEC89daHDEvPt",
                            content="archived-package-1",
                        ),
                    ],
                ),
            ],
        )

        result = await retriever.process(request, mock_context)

        # Verify storage engine was called with the correct package name
        mock_storage_engine.search.assert_called_with(query="archived-package-1", distance=0.5, limit=100)
        # verify the tool message was augmented with the package description
        assert "archived-1" in result.request.messages[2].content[0].content


    def test_generate_context_str(self, mock_storage_engine, mock_context):
        """Test context string generation"""
        retriever = CodegateContextRetriever(storage_engine=mock_storage_engine)

        bad_packages = [
            {
                "properties": {
                    "name": "bad-package-1",
                    "type": "npm",
                    "status": "malicious",
                    "description": "This package is bad-1",
                },
            },
            {
                "properties": {
                    "name": "bad-package-2",
                    "type": "pip",
                    "status": "archived",
                    "description": "This package is bad-2",
                },
            }
        ]

        context_str = retriever.generate_context_str(bad_packages, mock_context)

        assert "bad-package-1" in context_str
        assert "bad-package-2" in context_str
        assert "npm" in context_str
        assert "pip" in context_str
        assert "bad-1" in context_str
        assert "bad-2" in context_str
        assert "malicious" in context_str
        assert "archived" in context_str

        assert mock_context.add_alert.call_count == len(bad_packages)
