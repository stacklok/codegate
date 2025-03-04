from typing import List, Optional

import structlog
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from codegate.db.models import AlertTriggerType
from codegate.pipeline.base import PipelineContext
from codegate.pipeline.sensitive_data.session_store import SessionStore

logger = structlog.get_logger("codegate.pii.analyzer")


class PiiAnalyzer:
    """
    PiiAnalyzer class for analyzing and anonymizing text containing PII.
    This is a singleton class - use PiiAnalyzer.get_instance() to get the instance.

    Methods:
        get_instance():
            Get or create the singleton instance of PiiAnalyzer.
        analyze:
            text (str): The text to analyze for PII.
            Tuple[str, List[Dict[str, Any]], SessionStore]: The anonymized text, a list of
            found PII details, and the session store.
            entities (List[str]): The PII entities to analyze for.
        restore_pii:
            anonymized_text (str): The text with anonymized PII.
            session_store (SessionStore): The SessionStore used for anonymization.
            str: The text with original PII restored.
    """

    _instance: Optional["PiiAnalyzer"] = None
    _name = AlertTriggerType.CODEGATE_PII.value

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
        self.session_store = SessionStore()

        PiiAnalyzer._instance = self

    def analyze(self, text: str, context: Optional[PipelineContext] = None) -> List:
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
        return analyzer_results

    def restore_pii(self, session_id: str, anonymized_text: str) -> str:
        """
        Restore the original PII (Personally Identifiable Information) in the given anonymized text.

        This method replaces placeholders in the anonymized text with their corresponding original
        PII values using the mappings stored in the provided SessionStore.

        Args:
            anonymized_text (str): The text containing placeholders for PII.
            session_id (str): The session id containing mappings of placeholders
            to original PII.

        Returns:
            str: The text with the original PII restored.
        """
        session_data = self.session_store.get_by_session_id(session_id)
        if not session_data:
            logger.warning(
                "No active PII session found for given session ID. Unable to restore PII."
            )
            return anonymized_text

        for uuid_placeholder, original_pii in session_data.items():
            anonymized_text = anonymized_text.replace(uuid_placeholder, original_pii)
        return anonymized_text
