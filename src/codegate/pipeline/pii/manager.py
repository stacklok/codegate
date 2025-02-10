from typing import Any, Dict, List, Tuple

import structlog

from codegate.pipeline.pii.analyzer import PiiAnalyzer, PiiSessionStore

logger = structlog.get_logger("codegate")


class PiiManager:
    """
    Manages the analysis and restoration of Personally Identifiable Information (PII) in text.

    Attributes:
        analyzer (PiiAnalyzer): The singleton instance of PiiAnalyzer used for PII detection and restoration.
        session_store (PiiSessionStore): The session store for the current PII session.

    Methods:
        __init__():
            Initializes the PiiManager with the singleton PiiAnalyzer instance and sets the
            session store.

        analyze(text: str) -> Tuple[str, List[Dict[str, Any]]]:
            Analyzes the given text for PII, anonymizes it, and logs the detected PII details.
            Args:
                text (str): The text to be analyzed for PII.
            Returns:
                Tuple[str, List[Dict[str, Any]]]: A tuple containing the anonymized text and
                a list of found PII details.

        restore_pii(anonymized_text: str) -> str:
            Restores the PII in the given anonymized text using the current session.
            Args:
                anonymized_text (str): The text with anonymized PII to be restored.
            Returns:
                str: The text with restored PII.
    """

    def __init__(self):
        """
        Initialize the PiiManager with the singleton PiiAnalyzer instance.
        """
        self.analyzer = PiiAnalyzer.get_instance()
        self.session_store = self.analyzer.session_store

    def analyze(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        anonymized_text, found_pii, self.session_store = self.analyzer.analyze(text)

        # Log found PII details
        if found_pii:
            for pii in found_pii:
                logger.info(
                    "PII detected",
                    pii_type=pii["type"],
                    value="*" * len(pii["value"]),  # Don't log actual value
                    score=f"{pii['score']:.2f}",
                )

        return anonymized_text, found_pii

    def restore_pii(self, anonymized_text: str) -> str:
        if self.session_store is None:
            logger.warning("No active PII session found. Unable to restore PII.")
            return anonymized_text
        logger.debug("Restoring PII from session.")
        logger.debug(f"Current session: {self.session_store}")
        logger.debug(f"Anonymized text: {anonymized_text}")
        restored_text = self.analyzer.restore_pii(anonymized_text, self.session_store)
        logger.debug(f"Restored text: {restored_text}")
        return restored_text
