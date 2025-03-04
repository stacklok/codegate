import json
from typing import Dict, Optional
import pydantic
import structlog
from codegate.pipeline.sensitive_data.session_store import SessionStore

logger = structlog.get_logger("codegate")


class SensitiveData(pydantic.BaseModel):
    """Represents sensitive data with additional metadata."""

    original: str
    service: Optional[str] = None
    type: Optional[str] = None


class SensitiveDataManager:
    """Manages encryption, storage, and retrieval of secrets"""

    def __init__(self):
        self.session_store = SessionStore()

    def store(self, session_id: str, value: SensitiveData) -> Optional[str]:
        if not session_id or not value.original:
            return None
        return self.session_store.add_mapping(session_id, value.model_dump_json())

    def get_by_session_id(self, session_id: str) -> Optional[Dict]:
        if not session_id:
            return None
        data = self.session_store.get_by_session_id(session_id)
        return SensitiveData.model_validate_json(data) if data else None

    def get_original_value(self, session_id: str, uuid_placeholder: str) -> Optional[str]:
        if not session_id:
            return None
        secret_entry_json = self.session_store.get_mapping(session_id, uuid_placeholder)
        return (
            SensitiveData.model_validate_json(secret_entry_json).original
            if secret_entry_json
            else None
        )

    def cleanup_session(self, session_id: str):
        if session_id:
            self.session_store.cleanup_session(session_id)

    def cleanup(self):
        self.session_store.cleanup()
