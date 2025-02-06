# signatures.py
import math
import re
from pathlib import Path
from threading import Lock
from typing import ClassVar, Dict, List, NamedTuple, Optional, Union

import structlog
import yaml

logger = structlog.get_logger("codegate")


class Match(NamedTuple):
    """Represents a detected secret within text."""

    service: str
    type: str
    secret_key: str
    value: str
    line_number: int
    start_index: int
    end_index: int


class SignatureGroup:
    """Represents a group of signature patterns for a specific service."""

    def __init__(self, name: str, patterns: Dict[str, str]):
        self.name = name
        self.patterns = patterns


class YAMLParseError(Exception):
    """Custom exception for YAML parsing errors with helpful messages."""

    pass


class CodegateSignatures:
    """Main class for detecting secrets in text using regex patterns."""

    _instance_lock: ClassVar[Lock] = Lock()
    _signature_groups: ClassVar[List[SignatureGroup]] = []
    _compiled_regexes: ClassVar[Dict[str, re.Pattern]] = {}
    _yaml_path: ClassVar[Optional[str]] = None
    HIGH_ENTROPY_THRESHOLD: ClassVar[float] = 4.0

    @classmethod
    def _calculate_entropy(cls, text: str) -> float:
        """Calculate Shannon entropy for a given string."""
        if not text:
            return 0.0

        prob = {char: text.count(char) / len(text) for char in set(text)}
        return -sum(p * math.log2(p) for p in prob.values())

    @classmethod
    def reset(cls) -> None:
        """Reset the cached patterns."""
        with cls._instance_lock:
            cls._signature_groups = []
            cls._compiled_regexes = {}
            cls._yaml_path = None
            logger.debug("SecretFinder cache reset")

    @classmethod
    def initialize(cls, yaml_path: str) -> None:
        """Initialize the SecretFinder with a YAML file path and load signatures."""
        if not Path(yaml_path).exists():
            raise FileNotFoundError(f"Signatures file not found: {yaml_path}")

        with cls._instance_lock:
            # Only initialize if not already initialized with this path
            if cls._yaml_path != yaml_path:
                cls._yaml_path = yaml_path
                cls._load_signatures()
                logger.debug(f"SecretFinder initialized with {yaml_path}")

    @classmethod
    def _preprocess_yaml(cls, content: str) -> str:
        """Preprocess YAML content to handle common issues."""
        content = content.replace("\t", "    ")
        if content.startswith("\ufeff"):
            content = content[1:]
        return content.replace("\r\n", "\n").replace("\r", "\n")

    @classmethod
    def _load_yaml(cls, yaml_path: str) -> List[dict]:
        """Load and parse the YAML file containing signature patterns."""
        try:
            with open(yaml_path, "r", encoding="utf-8") as file:
                content = file.read()

            processed_content = cls._preprocess_yaml(content)

            try:
                data = yaml.safe_load(processed_content)
                if not isinstance(data, list):
                    raise YAMLParseError("YAML root must be a list")
                return data

            except yaml.YAMLError as e:
                error_msg = str(e)
                if "found character" in error_msg and "that cannot start any token" in error_msg:
                    raise YAMLParseError(
                        "Invalid character found in YAML file. Ensure no tabs are used "
                        "(use spaces for indentation) and no special characters are present."
                    )
                raise YAMLParseError(f"YAML parsing error: {error_msg}")

        except (OSError, IOError) as e:
            logger.error(f"Failed to read signatures file: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading YAML: {e}")
            raise

    @classmethod
    def _compile_regex(cls, pattern: str) -> Optional[re.Pattern]:
        """
        Compile a regex pattern with proper handling of flags.

        Args:
            pattern: The regex pattern string

        Returns:
            Compiled regex pattern or None if compilation fails
        """
        try:
            # Handle case-insensitive flag
            if pattern.startswith("(?i)"):
                return re.compile(pattern, re.IGNORECASE)

            # Find all flag indicators
            if "(?i)" in pattern:
                # Remove the flag from middle of pattern and compile with IGNORECASE
                pattern = pattern.replace("(?i)", "")
                return re.compile(pattern, re.IGNORECASE)

            return re.compile(pattern)

        except re.error as e:
            logger.warning(f"Invalid regex pattern '{pattern}': {e}")
            return None

    @classmethod
    def _sanitize_pattern(cls, pattern: str) -> str:
        """
        Sanitize regex pattern by handling common issues.

        Args:
            pattern: The original regex pattern

        Returns:
            Sanitized pattern string
        """
        if not pattern:
            return pattern

        # Convert \b word boundary to more specific pattern if at start/end
        pattern = pattern.replace(r"\b(?i)", "(?i)\\b")

        # Handle other common pattern issues here if needed
        return pattern

    @classmethod
    def _add_signature_group(cls, name: str, patterns: Dict[str, str]) -> None:
        """Add a new signature group and compile its regex patterns."""
        # Check if this group already exists
        if any(group.name == name for group in cls._signature_groups):
            logger.debug(f"Signature group {name} already exists, skipping")
            return

        signature_group = SignatureGroup(name, patterns)

        for pattern_name, pattern in patterns.items():
            regex_key = f"{name}:{pattern_name}"
            if compiled_pattern := cls._compile_regex(pattern):
                cls._compiled_regexes[regex_key] = compiled_pattern
                logger.debug(f"Successfully compiled regex for {regex_key}")
            else:
                logger.warning(f"Skipping invalid regex for {regex_key}")
                continue

        cls._signature_groups.append(signature_group)

    @classmethod
    def _load_signatures(cls) -> None:
        """Load signature patterns from the YAML file."""
        try:
            print("i load signatures")
            print(cls._yaml_path)
            # Clear existing signatures before loading new ones
            cls._signature_groups = []
            cls._compiled_regexes = {}
            yaml_data = cls._load_yaml(cls._yaml_path)
            print(yaml_data)

            # Process patterns from YAML
            for item in yaml_data:
                for service_name, patterns in item.items():
                    service_patterns = {}
                    for pattern_dict in patterns:
                        for pattern_name, pattern in pattern_dict.items():
                            if not pattern or pattern.startswith("#"):
                                continue
                            # Sanitize pattern before adding
                            service_patterns[pattern_name] = cls._sanitize_pattern(pattern)

                    if service_patterns:
                        cls._add_signature_group(service_name, service_patterns)

            logger.info(f"Loaded {len(cls._signature_groups)} signature groups")
            logger.info(f"Compiled {len(cls._compiled_regexes)} regex patterns")

        except Exception as e:
            logger.error(f"Error loading signatures: {e}")
            raise

    @classmethod
    def find_in_string(cls, text: Union[str, List[str]]) -> List[Match]:
        """Search for secrets in the provided string or list of strings."""
        if not text:
            return []

        if not cls._yaml_path:
            raise RuntimeError("SecretFinder not initialized.")

        matches = []
        found_values = set()

        # Split text into lines for processing
        try:
            lines = text.splitlines()
        except Exception as e:
            logger.warning(f"Error splitting text into lines: {e}")
            return []

        for line_num, line in enumerate(lines, start=1):
            matches.extend(cls._find_regex_matches(line, line_num, found_values))
            matches.extend(cls._find_high_entropy_matches(line, line_num, found_values))
        return matches

    @classmethod
    def _find_regex_matches(cls, line: str, line_num: int, found_values: set) -> List[Match]:
        """Find matches using regex patterns."""
        matches = []
        for group in cls._signature_groups:
            for pattern_name, regex in group.patterns.items():
                regex_key = f"{group.name}:{pattern_name}"
                regex = cls._compiled_regexes.get(regex_key)
                if not regex:
                    continue
                for match in regex.finditer(line):
                    value = match.group()
                    key = cls._extract_key_from_line(line, value)
                    pattern = f"{key}:{value}"
                    if value.lower() == "token" or pattern in found_values:
                        continue
                    found_values.add(pattern)
                    matches.append(
                        Match(
                            group.name,
                            pattern_name,
                            key,
                            value,
                            line_num,
                            match.start(),
                            match.end(),
                        )
                    )
        return matches

    @staticmethod
    def _extract_key_from_line(line: str, secret_value: str) -> Optional[str]:
        """
        Extract the key associated with a secret value if it follows a key=value pattern.
        """
        match = re.search(
            r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*["\']?' + re.escape(secret_value) + r'["\']?', line
        )
        return match.group(1) if match else None

    @classmethod
    def _find_high_entropy_matches(cls, line: str, line_num: int, found_values: set) -> List[Match]:
        """Find matches based on high entropy values."""
        matches = []
        assignment_pattern = re.findall(
            r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([\"']?([A-Za-z0-9_\-\.+/=]{8,})[\"']?)", line
        )

        for key, _, word in assignment_pattern:
            pattern = f"{key}:{word}"
            if pattern in found_values or word.startswith("REDACTED"):
                continue
            if cls._calculate_entropy(word) >= cls.HIGH_ENTROPY_THRESHOLD:
                found_values.add(pattern)
                matches.append(
                    Match(
                        "High Entropy",
                        "Potential Secret",
                        key,
                        word,
                        line_num,
                        line.find(word),
                        line.find(word) + len(word),
                    )
                )

        return matches
