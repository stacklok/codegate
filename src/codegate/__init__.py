"""Codegate - A Generative AI security gateway."""

import logging as python_logging
from importlib import metadata

from codegate.codegate_logging import LogFormat, LogLevel, setup_logging
from codegate.config import Config
from codegate.exceptions import ConfigurationError

try:
    __version__ = metadata.version("codegate")
    __description__ = metadata.metadata("codegate")["Summary"]
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
    __description__ = "codegate"

__version__ = "0.1.0"
__description__ = "A configurable service gateway"

__all__ = ["Config", "ConfigurationError", "LogFormat", "LogLevel", "setup_logging"]

# Set up null handler to avoid "No handler found" warnings.
# See https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library
python_logging.getLogger(__name__).addHandler(python_logging.NullHandler())
