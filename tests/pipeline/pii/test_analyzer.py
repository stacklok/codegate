from unittest.mock import MagicMock, patch

import pytest
from presidio_analyzer import RecognizerResult

from codegate.pipeline.pii.analyzer import PiiAnalyzer


class TestPiiAnalyzer:
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the singleton instance before each test"""
        PiiAnalyzer._instance = None
        yield
        PiiAnalyzer._instance = None

    @pytest.fixture
    def mock_nlp_engine(self):
        with patch("presidio_analyzer.nlp_engine.NlpEngineProvider") as mock:
            mock_nlp = MagicMock()
            # Create a mock token with the required attributes
            mock_token = MagicMock()
            mock_token.text = "test@example.com"
            mock_token.idx = 12
            mock_token.like_email = True

            # Create a mock doc with tokens and indices
            mock_doc = MagicMock()
            mock_doc.tokens = [mock_token]
            mock_doc.tokens_indices = [12]  # Start index of the token

            # Set up the process_text method to return our mock doc
            mock_nlp.process_text = MagicMock(return_value=mock_doc)

            # Make the provider return our mock NLP engine
            mock.return_value.create_engine.return_value = mock_nlp
            yield mock_nlp

    @pytest.fixture
    def mock_analyzer_engine(self):
        with patch("presidio_analyzer.AnalyzerEngine") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def mock_anonymizer_engine(self):
        with patch("presidio_anonymizer.AnonymizerEngine") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def analyzer(self, mock_nlp_engine, mock_analyzer_engine, mock_anonymizer_engine):
        with patch("os.path.dirname") as mock_dirname:
            mock_dirname.return_value = "/test/path"
            return PiiAnalyzer.get_instance()

    def test_singleton_pattern(self):
        """Test that PiiAnalyzer follows the singleton pattern"""
        # First instance
        analyzer1 = PiiAnalyzer.get_instance()
        # Second instance should be the same object
        analyzer2 = PiiAnalyzer.get_instance()
        assert analyzer1 is analyzer2
        # Direct instantiation should raise an error
        with pytest.raises(RuntimeError, match="Use PiiAnalyzer.get_instance()"):
            PiiAnalyzer()

    def test_restore_pii(self, analyzer):
        original_text = "test@example.com"
        session_id = "session-id"

        placeholder = analyzer.session_store.add_mapping(session_id, original_text)
        anonymized_text = f"My email is {placeholder}"
        restored_text = analyzer.restore_pii(session_id, anonymized_text)

        assert restored_text == f"My email is {original_text}"

    def test_restore_pii_multiple(self, analyzer):
        email = "test@example.com"
        phone = "123-456-7890"
        session_id = "session-id"
        email_placeholder = analyzer.session_store.add_mapping(session_id, email)
        phone_placeholder = analyzer.session_store.add_mapping(session_id, phone)
        anonymized_text = f"Email: {email_placeholder}, Phone: {phone_placeholder}"

        restored_text = analyzer.restore_pii(session_id, anonymized_text)

        assert restored_text == f"Email: {email}, Phone: {phone}"

    def test_restore_pii_no_placeholders(self, analyzer):
        text = "No PII here"
        session_id = "session-id"
        restored_text = analyzer.restore_pii(session_id, text)

        assert restored_text == text
