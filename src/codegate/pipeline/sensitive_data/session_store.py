from typing import Dict, Optional
import uuid


class SessionStore:
    """
    A generic session store for managing data protection.
    """

    def __init__(self):
        self.sessions: Dict[str, Dict[str, str]] = {}

    def add_mapping(self, session_id: str, data: str) -> str:
        uuid_placeholder = f"#{str(uuid.uuid4())}#"
        if session_id not in self.sessions:
            self.sessions[session_id] = {}
        self.sessions[session_id][uuid_placeholder] = data
        return uuid_placeholder

    def get_by_session_id(self, session_id: str) -> Optional[Dict]:
        return self.sessions.get(session_id, None)

    def get_mapping(self, session_id: str, uuid_placeholder: str) -> Optional[str]:
        return self.sessions.get(session_id, {}).get(uuid_placeholder)

    def cleanup_session(self, session_id: str):
        """Clears all stored mappings for a specific session."""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def cleanup(self):
        """Clears all stored mappings for all sessions."""
        self.sessions.clear()
