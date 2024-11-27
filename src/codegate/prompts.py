"""Prompt management for codegate."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

import yaml

from .exceptions import ConfigurationError


@dataclass
class PromptConfig:
    """Configuration for system prompts."""

    prompts: Dict[str, str] = field(default_factory=dict)

    def __getattr__(self, name: str) -> str:
        """Allow attribute-style access to prompts."""
        if name in self.prompts:
            return self.prompts[name]
        raise AttributeError(f"No prompt named '{name}' found")

    @classmethod
    def from_file(cls, prompt_path: Union[str, Path]) -> "PromptConfig":
        """Load prompts from a YAML file.

        Args:
            prompt_path: Path to the YAML prompts file

        Returns:
            PromptConfig: Prompts configuration instance

        Raises:
            ConfigurationError: If the file cannot be read or parsed
        """
        try:
            with open(prompt_path, "r") as f:
                prompt_data = yaml.safe_load(f)

            if not isinstance(prompt_data, dict):
                raise ConfigurationError("Prompts file must contain a YAML dictionary")

            # Validate all values are strings
            for key, value in prompt_data.items():
                if not isinstance(value, str):
                    raise ConfigurationError(f"Prompt '{key}' must be a string, got {type(value)}")

            return cls(prompts=prompt_data)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse prompts file: {e}")
        except OSError as e:
            raise ConfigurationError(f"Failed to read prompts file: {e}")

    @classmethod
    def load(cls, prompt_path: Optional[Union[str, Path]] = None) -> "PromptConfig":
        """Load prompts with optional file override.

        Args:
            prompt_path: Optional path to prompts file

        Returns:
            PromptConfig: Resolved prompts configuration

        Raises:
            ConfigurationError: If configuration is invalid
        """
        if prompt_path:
            return cls.from_file(prompt_path)
        return cls()
