from unittest.mock import MagicMock, patch

import pytest
from presidio_analyzer import RecognizerResult

from codegate.pipeline.pii.analyzer import PiiAnalyzer
from codegate.session.session_store import SessionStore


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

    def test_analyze_no_pii(self, analyzer, mock_analyzer_engine):
        text = "Hello world"
        mock_analyzer_engine.analyze.return_value = []

        result_text, found_pii, session_store = analyzer.analyze(text)

        assert result_text == text
        assert found_pii == []
        assert isinstance(session_store, SessionStore)

    def test_analyze_with_pii(self, analyzer, mock_analyzer_engine):
        text = "My email is test@example.com"
        email_pii = RecognizerResult(
            entity_type="EMAIL_ADDRESS",
            start=12,
            end=28,
            score=1.0,  # EmailRecognizer returns a score of 1.0
        )
        mock_analyzer_engine.analyze.return_value = [email_pii]

        result_text, found_pii, session_store = analyzer.analyze(text)

        assert len(found_pii) == 1
        pii_info = found_pii[0]
        assert pii_info["type"] == "EMAIL_ADDRESS"
        assert pii_info["value"] == "test@example.com"
        assert pii_info["score"] == 1.0
        assert pii_info["start"] == 12
        assert pii_info["end"] == 28
        assert "uuid_placeholder" in pii_info
        # Verify the placeholder was used to replace the PII
        placeholder = pii_info["uuid_placeholder"]
        assert result_text == f"My email is {placeholder}"
        # Verify the mapping was stored
        assert session_store.get_pii(placeholder) == "test@example.com"

    def test_restore_pii(self, analyzer):
        session_store = SessionStore()
        original_text = "test@example.com"
        session_id = "session-id"
        placeholder = session_store.add_mapping(session_id, original_text)
        anonymized_text = f"My email is {placeholder}"

        restored_text = analyzer.restore_pii(anonymized_text, session_id)

        assert restored_text == f"My email is {original_text}"

    def test_restore_pii_multiple(self, analyzer):
        session_store = SessionStore()
        email = "test@example.com"
        phone = "123-456-7890"
        session_id = "session-id"
        email_placeholder = session_store.add_mapping(session_id, email)
        phone_placeholder = session_store.add_mapping(session_id, phone)
        anonymized_text = f"Email: {email_placeholder}, Phone: {phone_placeholder}"

        restored_text = analyzer.restore_pii(anonymized_text, session_id)

        assert restored_text == f"Email: {email}, Phone: {phone}"

    def test_restore_pii_no_placeholders(self, analyzer):
        text = "No PII here"
        session_id = "session-id"
        restored_text = analyzer.restore_pii(text, session_id)

        assert restored_text == text
