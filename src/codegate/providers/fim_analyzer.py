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
    def _is_fim_request_body(cls, data) -> bool:
        """
        Determine from the raw incoming data if it's a FIM request.
        Used by: OpenAI and Anthropic
        """
        fim_stop_sequences = ["</COMPLETION>", "<COMPLETION>", "</QUERY>", "<QUERY>"]
        if data.first_message() is None:
            return False
        for content in data.first_message().get_content():
            for stop_sequence in fim_stop_sequences:
                if stop_sequence not in content.get_text():
                    return False
        return True

    @classmethod
    def is_fim_request(cls, request_url_path: str, data) -> bool:
        """
        Determine if the request is FIM by the URL or the data of the request.
        """
        # first check if we are in specific tools to discard FIM
        prompt = data.get_prompt("")
        tools = ["cline", "kodu", "open interpreter"]
        for tool in tools:
            if tool in prompt.lower():
                #  those tools can never be FIM
                return False
        # Avoid more expensive inspection of body by just checking the URL.
        if cls._is_fim_request_url(request_url_path):
            return True

        return cls._is_fim_request_body(data)
