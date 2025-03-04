import json
from unittest.mock import MagicMock, patch
import pytest
from codegate.pipeline.sensitive_data.manager import SensitiveData, SensitiveDataManager
from codegate.pipeline.sensitive_data.session_store import SessionStore


class TestSensitiveDataManager:
    @pytest.fixture
    def mock_session_store(self):
        """Mock the SessionStore instance used within SensitiveDataManager."""
        return MagicMock(spec=SessionStore)

    @pytest.fixture
    def manager(self, mock_session_store):
        """Patch SensitiveDataManager to use the mocked SessionStore."""
        with patch.object(SensitiveDataManager, "__init__", lambda self: None):
            manager = SensitiveDataManager()
            manager.session_store = mock_session_store  # Manually inject the mock
            return manager

    def test_store_success(self, manager, mock_session_store):
        """Test storing a SensitiveData object successfully."""
        session_id = "session-123"
        sensitive_data = SensitiveData(original="secret_value", service="AWS", type="API_KEY")

        # Mock session store behavior
        mock_session_store.add_mapping.return_value = "uuid-123"

        result = manager.store(session_id, sensitive_data)

        # Verify correct function calls
        mock_session_store.add_mapping.assert_called_once_with(
            session_id, sensitive_data.model_dump_json()
        )
        assert result == "uuid-123"

    def test_store_invalid_session_id(self, manager):
        """Test storing data with an invalid session ID (should return None)."""
        sensitive_data = SensitiveData(original="secret_value", service="AWS", type="API_KEY")
        result = manager.store("", sensitive_data)  # Empty session ID
        assert result is None

    def test_store_missing_original_value(self, manager):
        """Test storing data without an original value (should return None)."""
        sensitive_data = SensitiveData(original="", service="AWS", type="API_KEY")  # Empty original
        result = manager.store("session-123", sensitive_data)
        assert result is None
