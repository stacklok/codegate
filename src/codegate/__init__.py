"""CodeGate - A Generative AI security gateway."""

import logging as python_logging
from importlib import metadata

from codegate.codegate_logging import LogFormat, LogLevel, setup_logging
from codegate.config import Config
from codegate.exceptions import ConfigurationError

_VERSION = "dev"
_DESC = "CodeGate - A Generative AI security gateway."

def __get_version_and_description() -> tuple[str, str]:
    try:
        version = metadata.version("codegate")
        description = metadata.metadata("codegate")["Summary"]
    except metadata.PackageNotFoundError:
        version = _VERSION
        description = _DESC
    return version, description

__version__, __description__ = __get_version_and_description()

__all__ = ["Config", "ConfigurationError", "LogFormat", "LogLevel", "setup_logging"]

# Set up null handler to avoid "No handler found" warnings.
# See https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library
python_logging.getLogger(__name__).addHandler(python_logging.NullHandler())
