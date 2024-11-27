import datetime
import json
import logging
import sys
from enum import Enum
from typing import Any, Optional


class LogLevel(str, Enum):
    """Valid log levels."""

    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"

    @classmethod
    def _missing_(cls, value: str) -> Optional["LogLevel"]:
        """Handle case-insensitive lookup of enum values."""
        try:
            # Convert to uppercase and look up directly
            return cls[value.upper()]
        except (KeyError, AttributeError):
            raise ValueError(
                f"'{value}' is not a valid LogLevel. "
                f"Valid levels are: {', '.join(level.value for level in cls)}"
            )


class LogFormat(str, Enum):
    """Valid log formats."""

    JSON = "JSON"
    TEXT = "TEXT"

    @classmethod
    def _missing_(cls, value: str) -> Optional["LogFormat"]:
        """Handle case-insensitive lookup of enum values."""
        try:
            # Convert to uppercase and look up directly
            return cls[value.upper()]
        except (KeyError, AttributeError):
            raise ValueError(
                f"'{value}' is not a valid LogFormat. "
                f"Valid formats are: {', '.join(format.value for format in cls)}"
            )


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs log records as JSON."""

    def __init__(self) -> None:
        """Initialize the JSON formatter."""
        super().__init__()
        self.default_time_format = "%Y-%m-%dT%H:%M:%S"
        self.default_msec_format = "%s.%03dZ"

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string.

        Args:
            record: The log record to format

        Returns:
            str: JSON formatted log entry
        """
        # Create the base log entry
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.default_time_format),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
            "extra": {},
        }

        # Add extra fields from the record
        extra_attrs = {}
        for key, value in record.__dict__.items():
            if key not in {
                "args",
                "asctime",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "thread",
                "threadName",
                "extra",
            }:
                extra_attrs[key] = value

        # Handle the explicit extra parameter if present
        if hasattr(record, "extra"):
            try:
                if isinstance(record.extra, dict):
                    extra_attrs.update(record.extra)
            except Exception:
                extra_attrs["unserializable_extra"] = str(record.extra)

        # Add all extra attributes to the log entry
        if extra_attrs:
            try:
                json.dumps(extra_attrs)  # Test if serializable
                log_entry["extra"] = extra_attrs
            except (TypeError, ValueError):
                # If serialization fails, convert values to strings
                serializable_extra = {}
                for key, value in extra_attrs.items():
                    try:
                        json.dumps({key: value})  # Test individual value
                        serializable_extra[key] = value
                    except (TypeError, ValueError):
                        serializable_extra[key] = str(value)
                log_entry["extra"] = serializable_extra

        # Handle exception info if present
        if record.exc_info:
            log_entry["extra"]["exception"] = self.formatException(record.exc_info)

        # Handle stack info if present
        if record.stack_info:
            log_entry["extra"]["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(log_entry)


class TextFormatter(logging.Formatter):
    """Standard text formatter with consistent timestamp format."""

    def __init__(self) -> None:
        """Initialize the text formatter."""
        super().__init__(
            fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S.%03dZ",
        )

    def formatTime(  # noqa: N802
        self, record: logging.LogRecord, datefmt: Optional[str] = None
    ) -> str:
        """Format the time with millisecond precision.

        Args:
            record: The log record
            datefmt: The date format string (ignored as we use a fixed format)

        Returns:
            str: Formatted timestamp
        """
        ct = datetime.datetime.fromtimestamp(record.created, datetime.UTC)
        return ct.strftime(self.datefmt)


def setup_logging(
    log_level: Optional[LogLevel] = None, log_format: Optional[LogFormat] = None
) -> logging.Logger:
    """Configure the logging system.

    Args:
        log_level: The logging level to use. Defaults to INFO if not specified.
        log_format: The log format to use. Defaults to JSON if not specified.

    This configures two handlers:
    - stderr_handler: For ERROR, CRITICAL, and WARNING messages
    - stdout_handler: For INFO and DEBUG messages
    """
    if log_level is None:
        log_level = LogLevel.INFO
    if log_format is None:
        log_format = LogFormat.JSON

    # Create formatters
    json_formatter = JSONFormatter()
    text_formatter = TextFormatter()
    formatter = json_formatter if log_format == LogFormat.JSON else text_formatter

    # Create handlers for stdout and stderr
    stdout_handler = logging.StreamHandler(sys.stdout)
    stderr_handler = logging.StreamHandler(sys.stderr)

    # Set formatters
    stdout_handler.setFormatter(formatter)
    stderr_handler.setFormatter(formatter)

    # Configure log routing
    stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)
    stderr_handler.addFilter(lambda record: record.levelno > logging.INFO)

    # Get root logger and configure it
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level.value)

    # Remove any existing handlers and add our new ones
    root_logger.handlers.clear()
    root_logger.addHandler(stdout_handler)
    root_logger.addHandler(stderr_handler)

    # Create a logger for our package
    logger = logging.getLogger("codegate")
    logger.debug(
        "Logging initialized",
        extra={
            "log_level": log_level.value,
            "log_format": log_format.value,
            "handlers": ["stdout", "stderr"],
        },
    )

    return logger
