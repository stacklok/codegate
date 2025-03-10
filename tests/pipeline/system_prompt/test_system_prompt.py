from unittest.mock import AsyncMock, Mock

import pytest

from codegate.pipeline.base import PipelineContext
from codegate.pipeline.system_prompt.codegate import SystemPrompt
from codegate.types.openai import ChatCompletionRequest


class TestSystemPrompt:
    def test_init_with_system_message(self):
        """
        Test initialization with a system message
        """
        test_message = "Test system prompt"
        step = SystemPrompt(system_prompt=test_message, client_prompts={})
        assert step.codegate_system_prompt == test_message

    @pytest.mark.asyncio
    async def test_process_system_prompt_insertion(self):
        """
        Test system prompt insertion based on message content
        """
        # Prepare mock request with user message
        user_message = "Test user message"
        mock_request = {"model": "model", "messages": [{"role": "user", "content": user_message}]}
        mock_context = Mock(spec=PipelineContext)
        mock_context.secrets_found = False
        mock_context.pii_found = False

        # Create system prompt step
        system_prompt = "Security analysis system prompt"
        step = SystemPrompt(system_prompt=system_prompt, client_prompts={})
        step._get_workspace_custom_instructions = AsyncMock(return_value="")

        # Mock the get_last_user_message method
        step.get_last_user_message = Mock(return_value=(user_message, 0))

        # Process the request
        result = await step.process(ChatCompletionRequest(**mock_request), mock_context)

        # Check that system message was inserted
        assert len(result.request.messages) == 2
        assert result.request.messages[0].role == "user"
        assert result.request.messages[0].content == user_message
        assert result.request.messages[1].role == "system"
        assert result.request.messages[1].content == system_prompt

    @pytest.mark.asyncio
    async def test_process_system_prompt_update(self):
        """
        Test system prompt update
        """
        # Prepare mock request with user message
        request_system_message = "Existing system message"
        user_message = "Test user message"
        mock_request = {
            "model": "model",
            "messages": [
                {"role": "system", "content": request_system_message},
                {"role": "user", "content": user_message},
            ],
        }
        mock_context = Mock(spec=PipelineContext)
        mock_context.secrets_found = False
        mock_context.pii_found = False

        # Create system prompt step
        system_prompt = "Security analysis system prompt"
        step = SystemPrompt(system_prompt=system_prompt, client_prompts={})
        step._get_workspace_custom_instructions = AsyncMock(return_value="")

        # Mock the get_last_user_message method
        step.get_last_user_message = Mock(return_value=(user_message, 0))

        # Process the request
        result = await step.process(ChatCompletionRequest(**mock_request), mock_context)

        # Check that system message was inserted
        assert len(result.request.messages) == 2
        assert result.request.messages[0].role == "system"
        assert (
            result.request.messages[0].content
            == f"{system_prompt}\n\nHere are additional instructions:\n\n{request_system_message}"
        )
        assert result.request.messages[1].role == "user"
        assert result.request.messages[1].content == user_message

    @pytest.mark.asyncio
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
        mock_request = {"model": "model", "messages": edge_case if edge_case is not None else []}
        mock_context = Mock(spec=PipelineContext)
        mock_context.secrets_found = False
        mock_context.pii_found = False

        system_prompt = "Security edge case prompt"
        step = SystemPrompt(system_prompt=system_prompt, client_prompts={})
        step._get_workspace_custom_instructions = AsyncMock(return_value="")

        # Mock get_last_user_message to return None
        step.get_last_user_message = Mock(return_value=None)

        # Process the request
        result = await step.process(ChatCompletionRequest(**mock_request), mock_context)

        # Verify request remains unchanged
        assert len(result.request.messages) == 1
        # TODO this should use the abstract interface
        assert result.request.messages[0].role == "system"
        assert result.request.messages[0].content == system_prompt
