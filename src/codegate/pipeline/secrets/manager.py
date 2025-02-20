import json
from typing import Optional

import structlog

from codegate.session.session_store import SessionStore

logger = structlog.get_logger("codegate")


class SecretsManager:
    """Manages encryption, storage and retrieval of secrets"""

    def __init__(self):
        self.session_store = SessionStore()

    def store_secret(self, session_id: str, value: str, service: str, secret_type: str) -> str:
        """
        Encrypts and stores a secret value.
        Returns the encrypted value.
        """
        if not value:
            raise ValueError("Value must be provided")
        if not service:
            raise ValueError("Service must be provided")
        if not secret_type:
            raise ValueError("Secret type must be provided")

        uuid_placeholder = self.session_store.add_mapping(
            session_id,
            json.dumps({"original": value, "service": service, "secret_type": secret_type}),
        )
        logger.debug(
            "Stored secret", service=service, type=secret_type, placeholder=uuid_placeholder
        )
        return uuid_placeholder

    def get_original_value(self, session_id: str, uuid_placeholder: str) -> Optional[str]:
        """Retrieve original value for an encrypted value"""
        secret_entry_json = self.session_store.get_mapping(session_id, uuid_placeholder)
        if secret_entry_json:
            secret_entry = json.loads(secret_entry_json)
            return secret_entry.get("original")
        return None

    def cleanup_session(self, session_id):
        self.session_store.cleanup_session(session_id)
