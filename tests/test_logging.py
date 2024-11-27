import json
import logging
from io import StringIO

from codegate.codegate_logging import (
    JSONFormatter,
    LogFormat,
    LogLevel,
    TextFormatter,
    setup_logging,
)


def test_json_formatter():
    log_record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    formatter = JSONFormatter()
    formatted_log = formatter.format(log_record)
    log_entry = json.loads(formatted_log)

    assert log_entry["level"] == "INFO"
    assert log_entry["module"] == "test_logging"
    assert log_entry["message"] == "Test message"
    assert "timestamp" in log_entry
    assert "extra" in log_entry


def test_text_formatter():
    log_record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    formatter = TextFormatter()
    formatted_log = formatter.format(log_record)

    assert "INFO" in formatted_log
    assert "test" in formatted_log
    assert "Test message" in formatted_log


def test_setup_logging_json_format():
    setup_logging(log_level=LogLevel.DEBUG, log_format=LogFormat.JSON)
    logger = logging.getLogger("codegate")
    log_output = StringIO()
    handler = logging.StreamHandler(log_output)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)

    logger.debug("Debug message")
    log_output.seek(0)
    log_entry = json.loads(log_output.getvalue().strip())

    assert log_entry["level"] == "DEBUG"
    assert log_entry["message"] == "Debug message"


def test_setup_logging_text_format():
    setup_logging(log_level=LogLevel.DEBUG, log_format=LogFormat.TEXT)
    logger = logging.getLogger("codegate")
    log_output = StringIO()
    handler = logging.StreamHandler(log_output)
    handler.setFormatter(TextFormatter())
    logger.addHandler(handler)

    logger.debug("Debug message")
    log_output.seek(0)
    formatted_log = log_output.getvalue().strip()

    assert "DEBUG" in formatted_log
    assert "Debug message" in formatted_log
