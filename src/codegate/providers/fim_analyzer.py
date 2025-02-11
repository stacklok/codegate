from typing import Dict

import structlog

logger = structlog.get_logger("codegate")


class FIMAnalyzer:

    @classmethod
    def _is_fim_request_url(cls, request_url_path: str) -> bool:
        """
        Checks the request URL to determine if a request is FIM or chat completion.
        Used by: llama.cpp
        """
        # Evaluate first a larger substring.
        if request_url_path.endswith("chat/completions"):
            return False

        # /completions is for OpenAI standard. /api/generate is for ollama.
        if request_url_path.endswith("completions") or request_url_path.endswith("api/generate"):
            return True

        return False

    @classmethod
    def _is_fim_request_body(cls, data: Dict) -> bool:
        """
        Determine from the raw incoming data if it's a FIM request.
        Used by: OpenAI and Anthropic
        """
        messages = data.get("messages", [])
        if not messages:
            return False

        first_message_content = messages[0].get("content")
        if first_message_content is None:
            return False

        fim_stop_sequences = ["</COMPLETION>", "<COMPLETION>", "</QUERY>", "<QUERY>"]
        if isinstance(first_message_content, str):
            msg_prompt = first_message_content
        elif isinstance(first_message_content, list):
            msg_prompt = first_message_content[0].get("text", "")
        else:
            logger.warning(f"Could not determine if message was FIM from data: {data}")
            return False
        return all([stop_sequence in msg_prompt for stop_sequence in fim_stop_sequences])

    @classmethod
    def is_fim_request(cls, request_url_path: str, data: Dict) -> bool:
        """
        Determine if the request is FIM by the URL or the data of the request.
        """
        # first check if we are in specific tools to discard FIM
        prompt = data.get("prompt", "")
        tools = ["cline", "kodu", "open interpreter"]
        for tool in tools:
            if tool in prompt.lower():
                #  those tools can never be FIM
                return False
        # Avoid more expensive inspection of body by just checking the URL.
        if cls._is_fim_request_url(request_url_path):
            return True

        return cls._is_fim_request_body(data)
