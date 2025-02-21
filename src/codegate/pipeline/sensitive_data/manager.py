import json
from typing import Dict, Optional
import structlog
from codegate.pipeline.sensitive_data.session_store import SessionStore

logger = structlog.get_logger("codegate")


class SensitiveData:
    """Represents sensitive data with additional metadata."""

    def __init__(self, original: str, service: Optional[str], type: Optional[str]):
        self.original = original
        self.service = service
        self.type = type

    def to_json(self) -> str:
        """Serializes the object to a JSON string."""
        return json.dumps({key: value for key, value in vars(self).items() if value is not None})

    @staticmethod
    def from_json(data: str) -> "SensitiveData":
        """Deserializes from a JSON string to a SensitiveData object."""
        obj = json.loads(data)
        return SensitiveData(obj["original"], obj.get("service"), obj.get("type"))


class SensitiveDataManager:
    """Manages encryption, storage, and retrieval of secrets"""

    def __init__(self):
        self.session_store = SessionStore()

    def store(self, session_id: str, value: SensitiveData) -> Optional[str]:
        print("in store")
        if not session_id or not value.original:
            return None
        print("i call add mapping")
        print(self.session_store)
        return self.session_store.add_mapping(session_id, value.to_json())

    def get_by_session_id(self, session_id: str) -> Optional[Dict]:
        if not session_id:
            return None
        data = self.session_store.get_by_session_id(session_id)
        return SensitiveData.from_json(data) if data else None

    def get_original_value(self, session_id: str, uuid_placeholder: str) -> Optional[str]:
        if not session_id:
            return None
        secret_entry_json = self.session_store.get_mapping(session_id, uuid_placeholder)
        return SensitiveData.from_json(secret_entry_json).original if secret_entry_json else None

    def cleanup_session(self, session_id: str):
        if session_id:
            self.session_store.cleanup_session(session_id)

    def cleanup(self):
        self.session_store.cleanup()
