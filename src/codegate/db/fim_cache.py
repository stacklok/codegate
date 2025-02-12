import datetime
import hashlib
import json
import re
from typing import Dict, List, Optional

import structlog
from pydantic import BaseModel

from codegate.config import Config
from codegate.db.models import Alert, AlertSeverity
from codegate.pipeline.base import PipelineContext

logger = structlog.get_logger("codegate")


class CachedFim(BaseModel):

    timestamp: datetime.datetime
    critical_alerts: List[Alert]
    initial_id: str


class FimCache:

    def __init__(self):
        self.cache: Dict[str, CachedFim] = {}

    def _extract_message_from_fim_request(self, request: str) -> Optional[str]:
        """Extract the user message from the FIM request"""
        ### NEW CODE PATH ###
        if not isinstance(request, str):
            content_message = None
            for message in request.get_messages():
                for content in message.get_content():
                    if content_message is None:
                        content_message = content.get_text()
                    else:
                        logger.warning("Expected one user message, found multiple.")
                        return None
            return content_message

        try:
            parsed_request = json.loads(request)
        except Exception as e:
            logger.error(f"Failed to extract request message: {request}", error=str(e))
            return None

        if not isinstance(parsed_request, dict):
            logger.warning(f"Expected a dictionary, got {type(parsed_request)}.")
            return None

        messages = [
            message
            for message in parsed_request.get("messages", [])
            if isinstance(message, dict) and message.get("role", "") == "user"
        ]
        if len(messages) != 1:
            logger.warning(f"Expected one user message, found {len(messages)}.")
            return None

        content_message = messages[0].get("content")
        return content_message

    def _match_filepath(self, message: str, provider: str) -> Optional[str]:
        # Try to extract the path from the FIM message. The path is in FIM request as a comment:
        # folder/testing_file.py
        # Path: file3.py
        # // Path: file3.js <-- Javascript
        pattern = r"^(#|//|<!--|--|%|;).*?\b([a-zA-Z0-9_\-\/]+\.\w+)\b"
        matches = re.findall(pattern, message, re.MULTILINE)
        # If no path is found, hash the entire prompt message.
        if not matches:
            return None

        # Extract only the paths (2nd group from the match)
        paths = [match[1] for match in matches]

        # Copilot puts the path at the top of the file. Continue providers contain
        # several paths, the one in which the fim is triggered is the last one.
        if provider == "copilot":
            return paths[0]
        else:
            return paths[-1]

    def _calculate_hash_key(self, message: str, provider: str) -> str:
        """Creates a hash key from the message and includes the provider"""
        filepath = self._match_filepath(message, provider)
        if filepath is None:
            logger.warning("No path found in messages. Creating hash key from message.")
            message_to_hash = f"{message}-{provider}"
        else:
            message_to_hash = f"{filepath}-{provider}"

        logger.debug(f"Message to hash: {message_to_hash}")
        hashed_content = hashlib.sha256(message_to_hash.encode("utf-8")).hexdigest()
        logger.debug(f"Hashed content: {hashed_content}")
        return hashed_content

    def _add_cache_entry(self, hash_key: str, context: PipelineContext):
        """Add a new cache entry"""
        if not context.input_request:
            logger.warning("No input request found. Skipping creating a mapping entry")
            return
        critical_alerts = [
            alert
            for alert in context.alerts_raised
            if alert.trigger_category == AlertSeverity.CRITICAL.value
        ]
        new_cache = CachedFim(
            timestamp=context.input_request.timestamp,
            critical_alerts=critical_alerts,
            initial_id=context.input_request.id,
        )
        self.cache[hash_key] = new_cache
        logger.info(f"Added cache entry for hash key: {hash_key}")
        return self.cache[hash_key]

    def _update_cache_entry(self, hash_key: str, context: PipelineContext):
        """Update an existing cache entry without changing the timestamp."""
        existing_entry = self.cache.get(hash_key)
        if existing_entry is not None:
            # Update critical alerts while retaining the original timestamp.
            critical_alerts = [
                alert
                for alert in context.alerts_raised
                if alert.trigger_category == AlertSeverity.CRITICAL.value
            ]
            # Update the entry in the cache with new critical alerts but keep the old timestamp.
            updated_cache = CachedFim(
                timestamp=existing_entry.timestamp,
                critical_alerts=critical_alerts,
                initial_id=existing_entry.initial_id,
            )
            self.cache[hash_key] = updated_cache
            logger.info(f"Updated cache entry for hash key: {hash_key}")
        else:
            # Log a warning if trying to update a non-existent entry - ideally should not happen.
            logger.warning(f"Attempted to update non-existent cache entry for hash key: {hash_key}")

    def _are_new_alerts_present(self, context: PipelineContext, cached_entry: CachedFim) -> bool:
        """Check if there are new alerts present"""
        new_critical_alerts = [
            alert
            for alert in context.alerts_raised
            if alert.trigger_category == AlertSeverity.CRITICAL.value
        ]
        return len(new_critical_alerts) > len(cached_entry.critical_alerts)

    def _is_cached_entry_old(self, context: PipelineContext, cached_entry: CachedFim) -> bool:
        """Check if the cached entry is old"""
        if not context.input_request:
            logger.warning("No input request found. Skipping checking if the cache entry is old")
            return False
        elapsed_seconds = (context.input_request.timestamp - cached_entry.timestamp).total_seconds()
        config = Config.get_config()
        if config is None:
            logger.warning("No configuration found. Skipping checking if the cache entry is old")
            return True
        return elapsed_seconds > Config.get_config().max_fim_hash_lifetime  # type: ignore

    def could_store_fim_request(self, context: PipelineContext):
        if not context.input_request:
            logger.warning("No input request found. Skipping creating a mapping entry")
            return False, "", ""
        # Couldn't process the user message. Skip creating a mapping entry.
        message = self._extract_message_from_fim_request(context.input_request.request)
        if message is None:
            logger.warning(f"Couldn't read FIM message: {message}. Will not record to DB.")
            return False, "", ""

        hash_key = self._calculate_hash_key(message, context.input_request.provider)  # type: ignore
        cached_entry = self.cache.get(hash_key, None)
        if (
            cached_entry is None
            or self._is_cached_entry_old(context, cached_entry)
            or self._are_new_alerts_present(context, cached_entry)
        ):
            cached_entry = self._add_cache_entry(hash_key, context)
            if cached_entry is None:
                logger.warning("Failed to add cache entry")
                return False, "", ""
            return True, "add", cached_entry.initial_id

        self._update_cache_entry(hash_key, context)
        return True, "update", cached_entry.initial_id
