import json
import logging
import sys
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog


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


# Define all LiteLLM logger names
LITELLM_LOGGERS = [
    "LiteLLM Proxy",
    "LiteLLM Router",
    "LiteLLM"
]


def configure_litellm_logging(enabled: bool = False, level: LogLevel = LogLevel.INFO) -> None:
    """Configure LiteLLM logging.
    
    Args:
        enabled: Whether to enable LiteLLM logging
        level: Log level to use if enabled
    """
    # Configure the main litellm logger
    logger = logging.getLogger("litellm")
    logger.disabled = not enabled
    if not enabled:
        logger.setLevel(logging.CRITICAL + 1)  # Effectively disables all logging
    else:
        logger.setLevel(getattr(logging, level.value))
        logger.propagate = False
        # Clear any existing handlers
        logger.handlers.clear()
        # Add a handler to ensure logs are properly routed
        handler = logging.StreamHandler()
        handler.setLevel(getattr(logging, level.value))
        logger.addHandler(handler)

    # Also configure the specific LiteLLM loggers
    for logger_name in LITELLM_LOGGERS:
        logger = logging.getLogger(logger_name)
        logger.disabled = not enabled
        if not enabled:
            logger.setLevel(logging.CRITICAL + 1)
        else:
            logger.setLevel(getattr(logging, level.value))
            logger.propagate = False
            logger.handlers.clear()
            handler = logging.StreamHandler()
            handler.setLevel(getattr(logging, level.value))
            logger.addHandler(handler)


def add_origin(logger, log_method, event_dict):
    # Add 'origin' if it's bound to the logger but not explicitly in the event dict
    if "origin" not in event_dict and hasattr(logger, "_context"):
        origin = logger._context.get("origin")
        if origin:
            event_dict["origin"] = origin
    return event_dict


def setup_logging(
    log_level: Optional[LogLevel] = None, 
    log_format: Optional[LogFormat] = None,
    external_loggers: Optional[Dict[str, bool]] = None
) -> logging.Logger:
    """Configure the logging system.

    Args:
        log_level: The logging level to use. Defaults to INFO if not specified.
        log_format: The log format to use. Defaults to JSON if not specified.
        external_loggers: Dictionary of external logger names and whether they should be enabled.
                        e.g. {"litellm": False, "sqlalchemy": False, "uvicorn.error": False}

    This configures two handlers:
    - stderr_handler: For ERROR, CRITICAL, and WARNING messages
    - stdout_handler: For INFO and DEBUG messages
    """
    if log_level is None:
        log_level = LogLevel.INFO
    if log_format is None:
        log_format = LogFormat.JSON
    if external_loggers is None:
        external_loggers = {
            "litellm": False,
            "sqlalchemy": False, 
            "uvicorn.error": False,
            "aiosqlite": False
        }

    # Configure LiteLLM logging based on external_loggers setting
    configure_litellm_logging(
        enabled=external_loggers.get("litellm", False),
        level=log_level
    )

    # The configuration was taken from structlog documentation
    # https://www.structlog.org/en/stable/standard-library.html
    # Specifically the section "Rendering Using structlog-based Formatters Within logging"

    # Adds log level and timestamp to log entries
    shared_processors = [
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="%Y-%m-%dT%H:%M:%S.%fZ", utc=True),
        add_origin,
        structlog.processors.CallsiteParameterAdder(
            [
                structlog.processors.CallsiteParameter.MODULE,
                structlog.processors.CallsiteParameter.PATHNAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        ),
    ]
    # Not sure why this is needed. I think it is a wrapper for the standard logging module.
    # Should allow to log both with structlog and the standard logging module:
    # import logging
    # import structlog
    # logging.getLogger("stdlog").info("woo")
    # structlog.get_logger("structlog").info("amazing", events="oh yes")
    structlog.configure(
        processors=shared_processors
        + [
            # Prepare event dict for `ProcessorFormatter`.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # The config aboves adds the following keys to all log entries: _record & _from_structlog.
    # remove_processors_meta removes them.
    processors = shared_processors + [structlog.stdlib.ProcessorFormatter.remove_processors_meta]
    # Choose the processors based on the log format
    if log_format == LogFormat.JSON:
        processors = processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = processors + [structlog.dev.ConsoleRenderer()]
    formatter = structlog.stdlib.ProcessorFormatter(
        # foreign_pre_chain run ONLY on `logging` entries that do NOT originate within structlog.
        foreign_pre_chain=shared_processors,
        processors=processors,
    )

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

    # Set explicitly the log level for other modules
    logging.getLogger("sqlalchemy").disabled = True
    logging.getLogger("uvicorn.error").disabled = True
    logging.getLogger("aiosqlite").disabled = True

    # Create a logger for our package
    logger = structlog.get_logger("codegate")
    logger.debug(
        "Logging initialized",
        extra={
            "log_level": log_level.value,
            "log_format": log_format.value,
            "handlers": ["stdout", "stderr"],
        },
    )


def serialize_for_logging(obj: Any) -> Any:
    """Serialize objects for logging, handling non-JSON serializable types"""
    if hasattr(obj, "__dict__"):
        return str(obj)
    elif isinstance(obj, (datetime, bytes)):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: serialize_for_logging(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_logging(item) for item in obj]
    return obj


def log_request(method: str, path: str, status_code: int, client: Any) -> None:
    """Log HTTP request details"""
    logger = logging.getLogger("proxy_pilot")
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "type": "request",
        "method": method,
        "path": path,
        "status_code": status_code,
        "client": serialize_for_logging(client),
    }
    logger.info(f"Request: {json.dumps(log_data, indent=2)}")


def log_proxy_forward(target_url: str, method: str, status_code: int) -> None:
    """Log proxy forwarding details"""
    logger = logging.getLogger("proxy_pilot")
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "type": "proxy_forward",
        "target_url": target_url,
        "method": method,
        "status_code": status_code,
    }
    logger.info(f"Proxy Forward: {json.dumps(log_data, indent=2)}")


def log_error(error_type: str, message: str, details: Dict = None) -> None:
    """Log error details"""
    logger = logging.getLogger("proxy_pilot")
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "type": "error",
        "error_type": error_type,
        "message": message,
        "details": serialize_for_logging(details or {}),
    }
    logger.error(f"Error: {json.dumps(log_data, indent=2)}")
