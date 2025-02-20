from typing import Any, Dict, List, Optional, Tuple

import structlog

from codegate.pipeline.base import PipelineContext
from codegate.pipeline.pii.analyzer import PiiAnalyzer
from codegate.session.session_store import SessionStore

logger = structlog.get_logger("codegate")


class PiiManager:
    """
    Manages the analysis and restoration of Personally Identifiable Information
    (PII) in text.

    Attributes:
        analyzer (PiiAnalyzer): The singleton instance of PiiAnalyzer used for
        PII detection and restoration.
        session_store (SessionStore): The session store for the current PII session.

    Methods:
        __init__():
            Initializes the PiiManager with the singleton PiiAnalyzer instance and sets the
            session store.

        analyze(session_id: str, text: str) -> Tuple[str, List[Dict[str, Any]]]:
            Analyzes the given text for PII, anonymizes it, and logs the detected PII details.
            Args:
                session_id (str): The session id to store the PII.
                text (str): The text to be analyzed for PII.
            Returns:
                Tuple[str, List[Dict[str, Any]]]: A tuple containing the anonymized text and
                a list of found PII details.

        restore_pii(session_id: str, anonymized_text: st ) -> str:
            Restores the PII in the given anonymized text using the current session.
            Args:
                session_id (str): The session id for the PII to be restored.
                anonymized_text (str): The text with anonymized PII to be restored.
            Returns:
                str: The text with restored PII.
    """

    def __init__(self):
        """
        Initialize the PiiManager with the singleton PiiAnalyzer instance.
        """
        self.analyzer = PiiAnalyzer.get_instance()
        # Always use the analyzer's session store
        self._session_store = self.analyzer.session_store

    @property
    def session_store(self) -> SessionStore:
        """Get the current session store."""
        # Always return the analyzer's current session store
        return self.analyzer.session_store

    def analyze(
        self, session_id: str, text: str, context: Optional[PipelineContext] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        # Call analyzer and get results
        anonymized_text, found_pii = self.analyzer.analyze(session_id, text, context=context)

        # Log found PII details (without modifying the found_pii list)
        if found_pii:
            for pii in found_pii:
                logger.info(
                    "PII detected",
                    pii_type=pii["type"],
                    value="*" * len(pii["value"]),  # Don't log actual value
                    score=f"{pii['score']:.2f}",
                )

        # Return the exact same objects we got from the analyzer
        return anonymized_text, found_pii

    def restore_pii(self, session_id: str, anonymized_text: str) -> str:
        """
        Restore PII in the given anonymized text using the current session.
        """
        if not session_id:
            return anonymized_text
        # Use the analyzer's restore_pii method with the current session store
        return self.analyzer.restore_pii(session_id, anonymized_text)
