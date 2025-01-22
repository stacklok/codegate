# CLI commands and flags

CodeGate provides a command-line interface through `cli.py` with the following
structure:

## Main command

```bash
codegate [OPTIONS] COMMAND [ARGS]...
```

## Available commands

### `serve`

Start the CodeGate server:

```bash
codegate serve [OPTIONS]
```

#### Options

- `--port INTEGER`: Port to listen on (default: `8989`)
  - Must be between 1 and 65535
  - Overrides configuration file and environment variables

- `--host TEXT`: Host to bind to (default: `localhost`)
  - Overrides configuration file and environment variables

- `--log-level [ERROR|WARNING|INFO|DEBUG]`: Set the log level (default: `INFO`)
  - Optional
  - Case-insensitive
  - Overrides configuration file and environment variables

- `--log-format [JSON|TEXT]`: Set the log format (default: `JSON`)
  - Optional
  - Case-insensitive
  - Overrides configuration file and environment variables

- `--config FILE`: Path to YAML config file
  - Optional
  - Must be a valid YAML file
  - Configuration values can be overridden by environment variables and CLI
    options

- `--prompts FILE`: Path to YAML prompts file
  - Optional
  - Must be a valid YAML file

- `--vllm-url TEXT`: vLLM provider URL
  - Optional
  - Default: http://localhost:8000/v1

- `--openai-url TEXT`: OpenAI provider URL
  - Optional
  - Default: https://api.openai.com/v1

- `--anthropic-url TEXT`: Anthropic provider URL
  - Optional
  - Default: https://api.anthropic.com/v1

- `--ollama-url TEXT`: Ollama provider URL
  - Optional
  - Default: http://localhost:11434/api

- `--model-base-path TEXT`: Path to model base directory
  - Optional
  - Default: ./codegate_volume/models

- `--embedding-model TEXT`: Name of embedding model
  - Optional
  - Default: all-minilm-L6-v2-q5_k_m.gguf

- `--certs-dir TEXT`: Directory for certificate files
  - Optional
  - Default: ./certs

- `--ca-cert TEXT`: CA certificate file name
  - Optional
  - Default: ca.crt

- `--ca-key TEXT`: CA key file name
  - Optional
  - Default: ca.key

- `--server-cert TEXT`: Server certificate file name
  - Optional
  - Default: server.crt

- `--server-key TEXT`: Server key file name
  - Optional
  - Default: server.key

- `--db-path TEXT`: Path to main SQLite database file
  - Optional
  - Default: ./codegate_volume/db/codegate.db

- `--vec-db-path TEXT`: Path to vector SQLite database file
  - Optional
  - Default: ./sqlite_data/vectordb.db

- `--enable-litellm`: Enable LiteLLM logging
  - Optional flag
  - Default: false
  - Enables logging for LiteLLM Proxy, Router, and core components
  - Overrides configuration file and environment variables

## Error handling

The CLI provides user-friendly error messages for:

- Invalid port numbers
- Invalid log levels
- Invalid log formats
- Configuration file errors
- Prompts file errors
- Server startup failures

All errors are output to stderr with appropriate exit codes.

## Examples

Start server with default settings:

```bash
codegate serve
```

Start server on specific port and host:

```bash
codegate serve --port 8989 --host localhost
```

Start server with custom logging:

```bash
codegate serve --log-level DEBUG --log-format TEXT
```

Start server with LiteLLM logging enabled:

```bash
codegate serve --enable-litellm --log-level DEBUG
```

Start server with configuration file:

```bash
codegate serve --config my-config.yaml
```

Start server with custom prompts:

```bash
codegate serve --prompts my-prompts.yaml
```

Start server with custom vLLM endpoint:

```bash
codegate serve --vllm-url https://vllm.example.com
```

Start server with custom Ollama endpoint:

```bash
codegate serve --ollama-url http://localhost:11434
```

Show default system prompts:

```bash
codegate show-prompts
```

Show prompts from a custom file:

```bash
codegate show-prompts --prompts my-prompts.yaml
```

Generate certificates with default settings:

```bash
codegate generate-certs
```

<!-- markdownlint-configure-file { "no-duplicate-heading": { "siblings_only": true } } -->
