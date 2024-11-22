"""Codegate - A Generative AI security gateway."""

from importlib import metadata

try:
    __version__ = metadata.version("codegate")
    __description__ = metadata.metadata("codegate")["Summary"]
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
    __description__ = "codegate"

from .config import Config, ConfigurationError
from .logging import setup_logging

__all__ = ["Config", "ConfigurationError", "setup_logging"]
