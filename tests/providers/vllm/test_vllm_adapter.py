import pytest

from codegate.providers.vllm.adapter import ChatMlInputNormalizer


class TestChatMlInputNormalizer:
    @pytest.fixture
    def normalizer(self):
        return ChatMlInputNormalizer()

    def test_str_from_message_simple_string(self):
        normalizer = ChatMlInputNormalizer()
        message = "Hello world"
        assert normalizer._str_from_message(message) == "Hello world"

    def test_str_from_message_dict_content(self):
        normalizer = ChatMlInputNormalizer()
        message = [{"type": "text", "text": "Hello world"}]
        assert normalizer._str_from_message(message) == "Hello world"

    def test_str_from_message_multiple_text_items(self):
        normalizer = ChatMlInputNormalizer()
        message = [{"type": "text", "text": "Hello"}, {"type": "text", "text": "world"}]
        assert normalizer._str_from_message(message) == "Hello world"

    def test_str_from_message_invalid_input(self):
        normalizer = ChatMlInputNormalizer()
        message = [{"type": "invalid"}]
        assert normalizer._str_from_message(message) == ""

    def test_split_chat_ml_request_single_message(self):
        normalizer = ChatMlInputNormalizer()
        request = """<|im_start|>system
You are an assistant<|im_end|>
<|im_start|>user
Hello, how are you?<|im_end|>"""

        result = normalizer.split_chat_ml_request(request)

        assert len(result) == 2
        assert result[0] == {"role": "system", "content": "You are an assistant"}
        assert result[1] == {"role": "user", "content": "Hello, how are you?"}

    def test_split_chat_ml_request_incomplete_message(self):
        normalizer = ChatMlInputNormalizer()
        request = """<|im_start|>system
You are an assistant"""

        result = normalizer.split_chat_ml_request(request)

        assert len(result) == 0

    def test_normalize_non_chat_ml_request(self, normalizer):
        input_data = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
        }

        result = normalizer.normalize(input_data)

        assert result == input_data

    def test_normalize_chat_ml_request(self, normalizer):
        input_data = {
            "messages": [
                {
                    "role": "user",
                    "content": """<|im_start|>system
You are an assistant<|im_end|>
<|im_start|>user
Hello, how are you?<|im_end|>""",
                }
            ]
        }

        result = normalizer.normalize(input_data)

        assert len(result["messages"]) == 2
        assert result["messages"][0] == {"role": "system", "content": "You are an assistant"}
        assert result["messages"][1] == {"role": "user", "content": "Hello, how are you?"}

    def test_normalize_with_additional_input_fields(self, normalizer):
        input_data = {
            "messages": [
                {
                    "role": "user",
                    "content": """<|im_start|>system
You are an assistant<|im_end|>
<|im_start|>user
Hello, how are you?<|im_end|>""",
                }
            ],
            "temperature": 0.7,
            "max_tokens": 100,
        }

        result = normalizer.normalize(input_data)

        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 100
        assert len(result["messages"]) == 2
