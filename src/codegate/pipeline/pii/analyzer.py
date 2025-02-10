import uuid
from typing import Any, Dict, List, Optional, Tuple

import structlog
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from codegate.db.models import AlertSeverity
from codegate.pipeline.base import PipelineContext

logger = structlog.get_logger("codegate.pii.analyzer")


class PiiSessionStore:
    """
    A class to manage PII (Personally Identifiable Information) session storage.

    Attributes:
        session_id (str): The unique identifier for the session. If not provided, a new UUID
        is generated. mappings (Dict[str, str]): A dictionary to store mappings between UUID
        placeholders and PII.

    Methods:
        add_mapping(pii: str) -> str:
            Adds a PII string to the session store and returns a UUID placeholder for it.

        get_pii(uuid_placeholder: str) -> str:
            Retrieves the PII string associated with the given UUID placeholder. If the placeholder
            is not found, returns the placeholder itself.
    """

    def __init__(self, session_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.mappings: Dict[str, str] = {}

    def add_mapping(self, pii: str) -> str:
        uuid_placeholder = f"<{str(uuid.uuid4())}>"
        self.mappings[uuid_placeholder] = pii
        return uuid_placeholder

    def get_pii(self, uuid_placeholder: str) -> str:
        return self.mappings.get(uuid_placeholder, uuid_placeholder)


class PiiAnalyzer:
    """
    PiiAnalyzer class for analyzing and anonymizing text containing PII.
    This is a singleton class - use PiiAnalyzer.get_instance() to get the instance.

    Methods:
        get_instance():
            Get or create the singleton instance of PiiAnalyzer.
        analyze:
            text (str): The text to analyze for PII.
            Tuple[str, List[Dict[str, Any]], PiiSessionStore]: The anonymized text, a list of
            found PII details, and the session store.
            entities (List[str]): The PII entities to analyze for.
        restore_pii:
            anonymized_text (str): The text with anonymized PII.
            session_store (PiiSessionStore): The PiiSessionStore used for anonymization.
            str: The text with original PII restored.
    """

    _instance: Optional["PiiAnalyzer"] = None
    _name = "codegate-pii"

    @classmethod
    def get_instance(cls) -> "PiiAnalyzer":
        """Get or create the singleton instance of PiiAnalyzer"""
        if cls._instance is None:
            logger.debug("Creating new PiiAnalyzer instance")
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """
        Initialize the PiiAnalyzer.
        Note: Use get_instance() instead of creating a new instance directly.
        """
        if PiiAnalyzer._instance is not None:
            raise RuntimeError("Use PiiAnalyzer.get_instance() instead")

        import os

        from presidio_analyzer.nlp_engine import NlpEngineProvider

        # Get the path to our custom spacy config
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "spacy_config.yaml")

        # Initialize the NLP engine with our custom configuration
        provider = NlpEngineProvider(conf_file=config_path)
        nlp_engine = provider.create_engine()

        # Create analyzer with custom NLP engine
        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
        self.anonymizer = AnonymizerEngine()
        self.session_store = PiiSessionStore()

        PiiAnalyzer._instance = self

    def analyze(
        self, text: str, context: Optional["PipelineContext"] = None
    ) -> Tuple[str, List[Dict[str, Any]], PiiSessionStore]:
        # Prioritize credit card detection first
        entities = [
            "PHONE_NUMBER",
            "EMAIL_ADDRESS",
            "CRYPTO",
            "CREDIT_CARD",
            "IBAN_CODE",
            "MEDICAL_LICENSE",
            "US_BANK_NUMBER",
            "US_ITIN",
            "US_PASSPORT",
            "US_SSN",
            "UK_NHS",
            "UK_NINO",
        ]

        # Analyze the text for PII with adjusted threshold for credit cards
        analyzer_results = self.analyzer.analyze(
            text=text,
            entities=entities,
            language="en",
            score_threshold=0.3,  # Lower threshold to catch more potential matches
        )

        # Track found PII
        found_pii = []

        # Only anonymize if PII was found
        if analyzer_results:
            # Log each found PII instance and anonymize
            anonymized_text = text
            for result in analyzer_results:
                pii_value = text[result.start : result.end]
                uuid_placeholder = self.session_store.add_mapping(pii_value)
                pii_info = {
                    "type": result.entity_type,
                    "value": pii_value,
                    "score": result.score,
                    "start": result.start,
                    "end": result.end,
                    "uuid_placeholder": uuid_placeholder,
                }
                found_pii.append(pii_info)
                anonymized_text = anonymized_text.replace(pii_value, uuid_placeholder)

                # Log each PII detection with its UUID mapping
                logger.info(
                    "PII detected and mapped",
                    pii_type=result.entity_type,
                    score=f"{result.score:.2f}",
                    uuid=uuid_placeholder,
                    # Don't log the actual PII value for security
                    value_length=len(pii_value),
                    session_id=self.session_store.session_id,
                )

            # Log summary of all PII found in this analysis
            if found_pii and context:
                # Create notification string for alert
                notify_string = (
                    f"**PII Detected** 🔒\n"
                    f"- Total PII Found: {len(found_pii)}\n"
                    f"- Types Found: {', '.join(set(p['type'] for p in found_pii))}\n"
                )
                context.add_alert(
                    self._name,
                    trigger_string=notify_string,
                    severity_category=AlertSeverity.CRITICAL,
                )

                logger.info(
                    "PII analysis complete",
                    total_pii_found=len(found_pii),
                    pii_types=[p["type"] for p in found_pii],
                    session_id=self.session_store.session_id,
                )

            # Return the anonymized text, PII details, and session store
            return anonymized_text, found_pii, self.session_store

        # If no PII found, return original text, empty list, and session store
        return text, [], self.session_store

    def restore_pii(self, anonymized_text: str, session_store: PiiSessionStore) -> str:
        """
        Restore the original PII (Personally Identifiable Information) in the given anonymized text.

        This method replaces placeholders in the anonymized text with their corresponding original
        PII values using the mappings stored in the provided PiiSessionStore.

        Args:
            anonymized_text (str): The text containing placeholders for PII.
            session_store (PiiSessionStore): The session store containing mappings of placeholders
            to original PII.

        Returns:
            str: The text with the original PII restored.
        """
        for uuid_placeholder, original_pii in session_store.mappings.items():
            anonymized_text = anonymized_text.replace(uuid_placeholder, original_pii)
        return anonymized_text
