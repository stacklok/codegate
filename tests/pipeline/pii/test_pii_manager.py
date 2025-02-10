from unittest.mock import MagicMock, patch

import pytest

from codegate.pipeline.pii.analyzer import PiiSessionStore
from codegate.pipeline.pii.manager import PiiManager


class TestPiiManager:
    @pytest.fixture
    def mock_analyzer(self):
        with patch("codegate.pipeline.pii.analyzer.PiiAnalyzer") as mock:
            mock_instance = MagicMock()
            mock_instance.analyze = MagicMock()
            mock_instance.restore_pii = MagicMock()
            mock_instance.session_store = PiiSessionStore()
            mock.get_instance.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def manager(self, mock_analyzer):
        return PiiManager()

    def test_init(self, manager, mock_analyzer):
        assert manager.session_store is mock_analyzer.session_store
        assert manager.analyzer is mock_analyzer

    def test_analyze_no_pii(self, manager, mock_analyzer):
        text = "Hello CodeGate"
        session_store = PiiSessionStore()
        mock_analyzer.analyze.return_value = (text, [], session_store)

        anonymized_text, found_pii = manager.analyze(text)

        assert anonymized_text == text
        assert found_pii == []
        assert isinstance(manager.session_store, PiiSessionStore)

    def test_analyze_with_pii(self, manager, mock_analyzer):
        text = "My email is test@example.com"
        session_store = PiiSessionStore()
        placeholder = "<test-uuid>"
        pii_details = [
            {
                "type": "EMAIL_ADDRESS",
                "value": "test@example.com",
                "score": 0.85,
                "start": 12,
                "end": 27,
                "uuid_placeholder": placeholder,
            }
        ]
        anonymized_text = f"My email is {placeholder}"
        session_store.mappings[placeholder] = "test@example.com"
        mock_analyzer.analyze.return_value = (anonymized_text, pii_details, session_store)

        result_text, found_pii = manager.analyze(text)

        # Don't check exact UUID since it's randomly generated!!
        assert "My email is <" in result_text
        assert ">" in result_text
        assert found_pii == pii_details
        assert manager.session_store == session_store
        assert manager.session_store.mappings[placeholder] == "test@example.com"

    def test_restore_pii_no_session(self, manager):
        text = "Anonymized text"
        manager.session_store = None

        restored_text = manager.restore_pii(text)

        assert restored_text == text

    def test_restore_pii_with_session(self, manager, mock_analyzer):
        anonymized_text = "My email is <test-uuid>"
        original_text = "My email is test@example.com"
        session = PiiSessionStore()
        placeholder = "<test-uuid>"
        session.mappings[placeholder] = "test@example.com"
        manager.session_store = session

        mock_analyzer.restore_pii.return_value = original_text

        restored_text = manager.restore_pii(anonymized_text)

        assert restored_text == original_text
        mock_analyzer.restore_pii.assert_called_once_with(anonymized_text, session)

    def test_restore_pii_multiple_placeholders(self, manager, mock_analyzer):
        anonymized_text = "Email: <uuid1>, Phone: <uuid2>"
        original_text = "Email: test@example.com, Phone: 123-456-7890"
        session = PiiSessionStore()
        session.mappings["<uuid1>"] = "test@example.com"
        session.mappings["<uuid2>"] = "123-456-7890"
        manager.session_store = session

        mock_analyzer.restore_pii.return_value = original_text

        restored_text = manager.restore_pii(anonymized_text)

        assert restored_text == original_text
        mock_analyzer.restore_pii.assert_called_once_with(anonymized_text, session)
