from unittest.mock import Mock

import pytest
from litellm.types.llms.openai import ChatCompletionRequest

from codegate.pipeline.base import PipelineContext
from codegate.pipeline.codegate_system_prompt.codegate import CodegateSystemPrompt


@pytest.mark.asyncio
class TestCodegateSystemPrompt:
    def test_init_no_system_message(self):
        """
        Test initialization with no system message
        """
        step = CodegateSystemPrompt()
        assert step._system_message["content"] is None

    def test_init_with_system_message(self):
        """
        Test initialization with a system message
        """
        test_message = "Test system prompt"
        step = CodegateSystemPrompt(system_prompt_message=test_message)
        assert step._system_message["content"] == test_message

    @pytest.mark.parametrize(
        "user_message,expected_modification",
        [
            # Test cases with different scenarios
            ("Hello CodeGate", True),
            ("CODEGATE in uppercase", True),
            ("No matching message", False),
            ("codegate with lowercase", True),
        ],
    )
    async def test_process_system_prompt_insertion(self, user_message, expected_modification):
        """
        Test system prompt insertion based on message content
        """
        # Prepare mock request with user message
        mock_request = {"messages": [{"role": "user", "content": user_message}]}
        mock_context = Mock(spec=PipelineContext)

        # Create system prompt step
        system_prompt = "Security analysis system prompt"
        step = CodegateSystemPrompt(system_prompt_message=system_prompt)

        # Mock the get_last_user_message method
        step.get_last_user_message = Mock(return_value=(user_message, 0))

        # Process the request
        result = await step.process(ChatCompletionRequest(**mock_request), mock_context)

        if expected_modification:
            # Check that system message was inserted
            assert len(result.request["messages"]) == 2
            assert result.request["messages"][0]["role"] == "system"
            assert result.request["messages"][0]["content"] == system_prompt
            assert result.request["messages"][1]["role"] == "user"
            assert result.request["messages"][1]["content"] == user_message
        else:
            # Ensure no modification occurred
            assert len(result.request["messages"]) == 1

    async def test_no_system_message_configured(self):
        """
        Test behavior when no system message is configured
        """
        mock_request = {"messages": [{"role": "user", "content": "CodeGate test"}]}
        mock_context = Mock(spec=PipelineContext)

        # Create step without system message
        step = CodegateSystemPrompt()

        # Process the request
        result = await step.process(ChatCompletionRequest(**mock_request), mock_context)

        # Verify request remains unchanged
        assert result.request == mock_request

    @pytest.mark.parametrize(
        "edge_case",
        [
            None,  # No messages
            [],  # Empty messages list
        ],
    )
    async def test_edge_cases(self, edge_case):
        """
        Test edge cases with None or empty message list
        """
        mock_request = {"messages": edge_case} if edge_case is not None else {}
        mock_context = Mock(spec=PipelineContext)

        system_prompt = "Security edge case prompt"
        step = CodegateSystemPrompt(system_prompt_message=system_prompt)

        # Mock get_last_user_message to return None
        step.get_last_user_message = Mock(return_value=None)

        # Process the request
        result = await step.process(ChatCompletionRequest(**mock_request), mock_context)

        # Verify request remains unchanged
        assert result.request == mock_request
