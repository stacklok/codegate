import uuid
from typing import Any, Dict, List, Tuple

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine


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
    Methods:
        __init__:
            Initializes the PiiAnalyzer with a custom NLP engine configuration.
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

    def __init__(self):
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

    def analyze(self, text: str) -> Tuple[str, List[Dict[str, Any]], PiiSessionStore]:
        entities = [
            "PHONE_NUMBER",
            "EMAIL_ADDRESS",
            "CREDIT_CARD",
            "CRYPTO",
            "IBAN_CODE",
            "IP_ADDRESS",
            "NRP",
            "MEDICAL_LICENSE",
            "US_BANK_NUMBER",
            "US_DRIVER_LICENSE",
            "US_ITIN",
            "US_PASSPORT",
            "US_SSN",
            "UK_NHS",
            "UK_NINO",
        ]

        # Analyze the text for PII
        analyzer_results = self.analyzer.analyze(text=text, entities=entities, language="en")

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
