from unittest.mock import MagicMock, patch

import pytest

from codegate.pipeline.pii.analyzer import PiiSessionStore
from codegate.pipeline.pii.manager import PiiManager


class TestPiiManager:
    @pytest.fixture
    def session_store(self):
        """Create a session store that will be shared between the mock and manager"""
        return PiiSessionStore()

    @pytest.fixture
    def mock_analyzer(self, session_store):
        """Create a mock analyzer with the shared session store"""
        mock_instance = MagicMock()
        mock_instance.analyze = MagicMock()
        mock_instance.restore_pii = MagicMock()
        mock_instance.session_store = session_store
        return mock_instance

    @pytest.fixture
    def manager(self, mock_analyzer):
        """Create a PiiManager instance with the mocked analyzer"""
        with patch("codegate.pipeline.pii.manager.PiiAnalyzer") as mock_analyzer_class:
            # Set up the mock class to return our mock instance
            mock_analyzer_class.get_instance.return_value = mock_analyzer
            # Create the manager which will use our mock
            return PiiManager()

    def test_init(self, manager, mock_analyzer):
        assert manager.session_store is mock_analyzer.session_store
        assert manager.analyzer is mock_analyzer

    def test_analyze_no_pii(self, manager, mock_analyzer):
        text = "Hello CodeGate"
        session_store = mock_analyzer.session_store
        mock_analyzer.analyze.return_value = (text, [], session_store)

        anonymized_text, found_pii = manager.analyze(text)

        assert anonymized_text == text
        assert found_pii == []
        assert manager.session_store is session_store
        mock_analyzer.analyze.assert_called_once_with(text)

    def test_analyze_with_pii(self, manager, mock_analyzer):
        text = "My email is test@example.com"
        session_store = mock_analyzer.session_store
        placeholder = "<test-uuid>"
        pii_details = [
            {
                "type": "EMAIL_ADDRESS",
                "value": "test@example.com",
                "score": 0.85,
                "start": 12,
                "end": 28,  # Fixed end position
                "uuid_placeholder": placeholder,
            }
        ]
        anonymized_text = f"My email is {placeholder}"
        session_store.mappings[placeholder] = "test@example.com"
        mock_analyzer.analyze.return_value = (anonymized_text, pii_details, session_store)

        result_text, found_pii = manager.analyze(text)

        assert "My email is <" in result_text
        assert ">" in result_text
        assert found_pii == pii_details
        assert manager.session_store is session_store
        assert manager.session_store.mappings[placeholder] == "test@example.com"
        mock_analyzer.analyze.assert_called_once_with(text)

    def test_restore_pii_no_session(self, manager, mock_analyzer):
        text = "Anonymized text"
        # Create a new session store that's None
        mock_analyzer.session_store = None

        restored_text = manager.restore_pii(text)

        assert restored_text == text

    def test_restore_pii_with_session(self, manager, mock_analyzer):
        anonymized_text = "My email is <test-uuid>"
        original_text = "My email is test@example.com"
        manager.session_store.mappings["<test-uuid>"] = "test@example.com"
        mock_analyzer.restore_pii.return_value = original_text

        restored_text = manager.restore_pii(anonymized_text)

        assert restored_text == original_text
        mock_analyzer.restore_pii.assert_called_once_with(anonymized_text, manager.session_store)

    def test_restore_pii_multiple_placeholders(self, manager, mock_analyzer):
        anonymized_text = "Email: <uuid1>, Phone: <uuid2>"
        original_text = "Email: test@example.com, Phone: 123-456-7890"
        manager.session_store.mappings["<uuid1>"] = "test@example.com"
        manager.session_store.mappings["<uuid2>"] = "123-456-7890"
        mock_analyzer.restore_pii.return_value = original_text

        restored_text = manager.restore_pii(anonymized_text)

        assert restored_text == original_text
        mock_analyzer.restore_pii.assert_called_once_with(anonymized_text, manager.session_store)
