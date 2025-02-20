import pytest

from codegate.pipeline.secrets.manager import SecretsManager


class TestSecretsManager:
    def setup_method(self):
        """Setup a fresh SecretsManager for each test"""
        self.manager = SecretsManager()
        self.test_value = "super_secret_value"
        self.test_service = "test_service"
        self.test_type = "api_key"

    def test_store_secret(self):
        """Test basic secret storage and retrieval"""
        # Store a secret
        encrypted = self.manager.store_secret(self.test_value, self.test_service, self.test_type)

        # Verify the secret was stored
        stored = self.manager.get_by_session_id(self.test_session)
        assert isinstance(stored, dict)
        assert stored[encrypted].original == self.test_value

        # Verify encrypted value can be retrieved
        retrieved = self.manager.get_original_value(encrypted, self.test_session)
        assert retrieved == self.test_value

    def test_get_original_value_wrong_session(self):
        """Test that secrets can't be accessed with wrong session ID"""
        encrypted = self.manager.store_secret(self.test_value, self.test_service, self.test_type)

        # Try to retrieve with wrong session ID
        wrong_session = "wrong_session_id"
        retrieved = self.manager.get_original_value(encrypted, wrong_session)
        assert retrieved is None

    def test_get_original_value_nonexistent(self):
        """Test handling of non-existent encrypted values"""
        retrieved = self.manager.get_original_value("nonexistent", self.test_session)
        assert retrieved is None

    def test_cleanup_session(self):
        """Test that session cleanup properly removes secrets"""
        # Store multiple secrets in different sessions
        session1 = "session1"
        session2 = "session2"

        encrypted1 = self.manager.store_secret("secret1", "service1", "type1", session1)
        encrypted2 = self.manager.store_secret("secret2", "service2", "type2", session2)

        # Clean up session1
        self.manager.cleanup_session(session1)

        # Verify session1 secrets are gone
        assert self.manager.get_by_session_id(session1) is None
        assert self.manager.get_original_value(encrypted1, session1) is None

        # Verify session2 secrets remain
        assert self.manager.get_by_session_id(session2) is not None
        assert self.manager.get_original_value(encrypted2, session2) == "secret2"

    def test_cleanup(self):
        """Test that cleanup properly wipes all data"""
        # Store multiple secrets
        self.manager.store_secret("secret1", "service1", "type1", "session1")
        self.manager.store_secret("secret2", "service2", "type2", "session2")

        # Perform cleanup
        self.manager.cleanup()

        # Verify all data is wiped
        assert len(self.manager._session_store) == 0
        assert len(self.manager._encrypted_to_session) == 0

    def test_multiple_secrets_same_session(self):
        """Test storing multiple secrets in the same session"""
        # Store multiple secrets in same session
        encrypted1 = self.manager.store_secret("secret1", "service1", "type1", self.test_session)
        encrypted2 = self.manager.store_secret("secret2", "service2", "type2", self.test_session)

        # Latest secret should be retrievable in the session
        stored = self.manager.get_by_session_id(self.test_session)
        assert isinstance(stored, dict)
        assert stored[encrypted1].original == "secret1"
        assert stored[encrypted2].original == "secret2"

        # Both secrets should be retrievable directly
        assert self.manager.get_original_value(encrypted1, self.test_session) == "secret1"
        assert self.manager.get_original_value(encrypted2, self.test_session) == "secret2"

        # Both encrypted values should map to the session
        assert self.manager._encrypted_to_session[encrypted1] == self.test_session
        assert self.manager._encrypted_to_session[encrypted2] == self.test_session

    def test_error_handling(self):
        """Test error handling in secret operations"""
        # Test with None values
        with pytest.raises(ValueError):
            self.manager.store_secret(None, self.test_service, self.test_type, self.test_session)

        with pytest.raises(ValueError):
            self.manager.store_secret(self.test_value, None, self.test_type, self.test_session)

        with pytest.raises(ValueError):
            self.manager.store_secret(self.test_value, self.test_service, None, self.test_session)

        with pytest.raises(ValueError):
            self.manager.store_secret(self.test_value, self.test_service, self.test_type, None)

    def test_secure_cleanup(self):
        """Test that cleanup securely wipes sensitive data"""
        # Store a secret
        self.manager.store_secret(
            self.test_value, self.test_service, self.test_type, self.test_session
        )

        # Get reference to stored data before cleanup
        stored = self.manager.get_by_session_id(self.test_session)
        assert len(stored) == 1

        # Perform cleanup
        self.manager.cleanup()

        # Verify the original string was overwritten, not just removed
        # This test is a bit tricky since Python strings are immutable,
        # but we can at least verify the data is no longer accessible
        assert self.test_value not in str(self.manager._session_store)

    def test_session_isolation(self):
        """Test that sessions are properly isolated"""
        session1 = "session1"
        session2 = "session2"

        # Store secrets in different sessions
        encrypted1 = self.manager.store_secret("secret1", "service1", "type1", session1)
        encrypted2 = self.manager.store_secret("secret2", "service2", "type2", session2)

        # Verify cross-session access is not possible
        assert self.manager.get_original_value(encrypted1, session2) is None
        assert self.manager.get_original_value(encrypted2, session1) is None

        # Verify correct session access works
        assert self.manager.get_original_value(encrypted1, session1) == "secret1"
        assert self.manager.get_original_value(encrypted2, session2) == "secret2"
