# Logging system

The logging system in CodeGate (`codegate_logging.py`) provides a flexible and
structured logging solution with support for both JSON and text output.

## Log routing

Logs are automatically routed based on their level:

- **stdout**: INFO and DEBUG messages
- **stderr**: ERROR, CRITICAL, and WARNING messages

## External Logger Configuration

CodeGate provides control over external loggers through configuration:

### LiteLLM Logging

LiteLLM logging can be controlled through:

1. CLI flag:
   ```bash
   codegate serve --enable-litellm
   ```

2. Environment variable:
   ```bash
   export CODEGATE_ENABLE_LITELLM=true
   ```

3. Configuration file:
   ```yaml
   external_loggers:
     litellm: true      # Enable/disable LiteLLM logging (includes LiteLLM Proxy, Router, and core)
     sqlalchemy: false  # Enable/disable SQLAlchemy logging
     uvicorn.error: false # Enable/disable Uvicorn error logging
     aiosqlite: false   # Enable/disable aiosqlite logging
   ```

The configuration follows the priority order:
1. CLI arguments (highest priority)
2. Environment variables
3. Config file
4. Default values (lowest priority)

## Log formats

### JSON format

When using JSON format (default), log entries include:

```json
{
  "timestamp": "YYYY-MM-DDThh:mm:ss.mmmZ",
  "log_level": "LOG_LEVEL",
  "module": "MODULE_NAME",
  "event": "Log message",
  "extra": {
    // Additional fields as you desire
  }
}
```

### Text format

When using text format, log entries follow this pattern:

```plain
YYYY-MM-DDThh:mm:ss.mmmZ - LEVEL - NAME - MESSAGE
```

## Features

- **Consistent timestamps**: ISO-8601 format with millisecond precision in UTC
- **Automatic JSON serialization**: extra fields are automatically serialized to
  JSON
- **Non-serializable handling**: graceful handling of non-serializable values
- **Exception support**: full exception and stack trace integration
- **Dual output**: separate handlers for error and non-error logs
- **Configurable levels**: support for ERROR, WARNING, INFO, and DEBUG levels
- **External logger control**: fine-grained control over third-party logging

## Usage examples

### Basic logging

```python
import structlog

logger = structlog.get_logger(__name__)

# Different log levels
logger.info("This is an info message")
logger.debug("This is a debug message")
logger.error("This is an error message")
logger.warning("This is a warning message")
```

### Logging with extra fields

```python
logger.info("Server starting", extra={
    "host": "0",
    "port": 8989,
    "environment": "production"
})
```

### Exception logging

```python
try:
    # Some code that might raise an exception
    raise ValueError("Something went wrong")
except Exception as e:
    logger.error("Error occurred", exc_info=True)
```

## Configuration

The logging system can be configured through:

1. CLI arguments:

   ```bash
   codegate serve --log-level DEBUG --log-format TEXT --enable-litellm # to enable LiteLLM debug
   ```

2. Environment variables:

   ```bash
   export CODEGATE_APP_LOG_LEVEL=DEBUG
   export CODEGATE_LOG_FORMAT=TEXT
   export CODEGATE_ENABLE_LITELLM=true
   ```

3. Configuration file:

   ```yaml
   log_level: DEBUG
   log_format: TEXT
   external_loggers:
     litellm: true
   ```

## Best practices

1. Always use the appropriate log level:

   - `ERROR` for errors that need immediate attention
   - `WARNING` for potentially harmful situations
   - `INFO` for general operational information
   - `DEBUG` for detailed information useful during development

2. Include relevant context in extra fields:

   ```python
   logger.info("User action", extra={
       "user_id": "123",
       "action": "login",
       "ip_address": "192.168.1.1"
   })
   ```

3. Use structured logging with [JSON format](#json-format) in production for
   better log aggregation and analysis.

4. Enable `DEBUG` level logging during development for maximum visibility.

5. Configure external loggers based on your needs:
   - Enable LiteLLM logging when debugging LLM-related issues
   - Keep external loggers disabled in production unless needed for troubleshooting
