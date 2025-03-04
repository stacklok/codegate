import pytest

from codegate.pipeline.sensitive_data.session_store import SessionStore


class TestSessionStore:
    @pytest.fixture
    def session_store(self):
        """Fixture to create a fresh SessionStore instance before each test."""
        return SessionStore()

    def test_add_mapping_creates_uuid(self, session_store):
        """Test that add_mapping correctly stores data and returns a UUID."""
        session_id = "session-123"
        data = "test-data"

        uuid_placeholder = session_store.add_mapping(session_id, data)

        # Ensure the returned placeholder follows the expected format
        assert uuid_placeholder.startswith("#") and uuid_placeholder.endswith("#")
        assert len(uuid_placeholder) > 2  # Should have a UUID inside

        # Verify data is correctly stored
        stored_data = session_store.get_mapping(session_id, uuid_placeholder)
        assert stored_data == data

    def test_add_mapping_creates_unique_uuids(self, session_store):
        """Ensure multiple calls to add_mapping generate unique UUIDs."""
        session_id = "session-123"
        data1 = "data1"
        data2 = "data2"

        uuid_placeholder1 = session_store.add_mapping(session_id, data1)
        uuid_placeholder2 = session_store.add_mapping(session_id, data2)

        assert uuid_placeholder1 != uuid_placeholder2  # UUIDs must be unique

        # Ensure data is correctly stored
        assert session_store.get_mapping(session_id, uuid_placeholder1) == data1
        assert session_store.get_mapping(session_id, uuid_placeholder2) == data2

    def test_get_by_session_id(self, session_store):
        """Test retrieving all stored mappings for a session ID."""
        session_id = "session-123"
        data1 = "data1"
        data2 = "data2"

        uuid1 = session_store.add_mapping(session_id, data1)
        uuid2 = session_store.add_mapping(session_id, data2)

        stored_session_data = session_store.get_by_session_id(session_id)

        assert uuid1 in stored_session_data
        assert uuid2 in stored_session_data
        assert stored_session_data[uuid1] == data1
        assert stored_session_data[uuid2] == data2

    def test_get_by_session_id_not_found(self, session_store):
        """Test get_by_session_id when session does not exist (should return None)."""
        session_id = "non-existent-session"
        assert session_store.get_by_session_id(session_id) is None

    def test_get_mapping_success(self, session_store):
        """Test retrieving a specific mapping."""
        session_id = "session-123"
        data = "test-data"

        uuid_placeholder = session_store.add_mapping(session_id, data)

        assert session_store.get_mapping(session_id, uuid_placeholder) == data

    def test_get_mapping_not_found(self, session_store):
        """Test retrieving a mapping that does not exist (should return None)."""
        session_id = "session-123"
        uuid_placeholder = "#non-existent-uuid#"

        assert session_store.get_mapping(session_id, uuid_placeholder) is None

    def test_cleanup_session(self, session_store):
        """Test that cleanup_session removes all data for a session ID."""
        session_id = "session-123"
        session_store.add_mapping(session_id, "test-data")

        # Ensure session exists before cleanup
        assert session_store.get_by_session_id(session_id) is not None

        session_store.cleanup_session(session_id)

        # Ensure session is removed after cleanup
        assert session_store.get_by_session_id(session_id) is None

    def test_cleanup_session_non_existent(self, session_store):
        """Test cleanup_session on a non-existent session (should not raise errors)."""
        session_id = "non-existent-session"
        session_store.cleanup_session(session_id)  # Should not fail
        assert session_store.get_by_session_id(session_id) is None

    def test_cleanup(self, session_store):
        """Test global cleanup removes all stored sessions."""
        session_id1 = "session-1"
        session_id2 = "session-2"

        session_store.add_mapping(session_id1, "data1")
        session_store.add_mapping(session_id2, "data2")

        # Ensure sessions exist before cleanup
        assert session_store.get_by_session_id(session_id1) is not None
        assert session_store.get_by_session_id(session_id2) is not None

        session_store.cleanup()

        # Ensure all sessions are removed after cleanup
        assert session_store.get_by_session_id(session_id1) is None
        assert session_store.get_by_session_id(session_id2) is None
