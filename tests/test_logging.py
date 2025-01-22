import logging
import sys
from io import StringIO

import structlog
from structlog.testing import capture_logs

from codegate.codegate_logging import (
    LogFormat,
    LogLevel,
    setup_logging,
)


def test_setup_logging():
    setup_logging(log_level=LogLevel.DEBUG, log_format=LogFormat.JSON)

    with capture_logs() as cap_logs:
        logger = structlog.get_logger("codegate")
        logger.debug("Debug message")

    # cap_logs is a dictionary with the list of log entries
    log_entry = cap_logs[0]

    assert log_entry["log_level"] == "debug"
    assert log_entry["event"] == "Debug message"


def test_logging_stream_output():
    setup_logging(log_level=LogLevel.DEBUG, log_format=LogFormat.TEXT)
    logger = logging.getLogger("codegate")
    log_output = StringIO()
    handler = logging.StreamHandler(log_output)
    logger.addHandler(handler)

    logger.debug("Debug message")
    log_output.seek(0)
    formatted_log = log_output.getvalue().strip()
    assert "Debug message" in formatted_log


def test_external_logger_configuration():
    # Test enabling litellm logging
    setup_logging(
        log_level=LogLevel.DEBUG,
        log_format=LogFormat.TEXT,
        external_loggers={"litellm": True}
    )
    litellm_logger = logging.getLogger("litellm")
    assert not litellm_logger.disabled
    assert litellm_logger.level == logging.DEBUG

    # Test disabling litellm logging
    setup_logging(
        log_level=LogLevel.DEBUG,
        log_format=LogFormat.TEXT,
        external_loggers={"litellm": False}
    )
    litellm_logger = logging.getLogger("litellm")
    assert litellm_logger.disabled
    assert litellm_logger.level > logging.CRITICAL


def test_external_logger_defaults():
    # Test default behavior (all external loggers disabled)
    setup_logging(log_level=LogLevel.DEBUG, log_format=LogFormat.TEXT)
    
    # Check all external loggers are disabled by default
    litellm_logger = logging.getLogger("litellm")
    sqlalchemy_logger = logging.getLogger("sqlalchemy")
    uvicorn_logger = logging.getLogger("uvicorn.error")
    aiosqlite_logger = logging.getLogger("aiosqlite")

    assert litellm_logger.disabled
    assert sqlalchemy_logger.disabled
    assert uvicorn_logger.disabled
    assert aiosqlite_logger.disabled
