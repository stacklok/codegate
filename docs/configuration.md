# Configuration system

The configuration system in CodeGate is managed through the `Config` class in
`config.py`. It supports multiple configuration sources with a clear priority
order.

## Configuration priority

Configuration sources are evaluated in the following order, from highest to
lowest priority:

1. CLI arguments
2. Environment variables
3. Configuration file (YAML)
4. Default values (including default prompts from `prompts/default.yaml`)

Values from higher-priority sources take precedence over lower-priority values.

## Default configuration values

- Port: `8989`
- Proxy port: `8990`
- Host: `"localhost"`
- Log level: `"INFO"`
- Log format: `"JSON"`
- Prompts: default prompts from `prompts/default.yaml`
- Provider URLs:
  - vLLM: `"http://localhost:8000"`
  - OpenAI: `"https://api.openai.com/v1"`
  - Anthropic: `"https://api.anthropic.com/v1"`
  - Ollama: `"http://localhost:11434"`
- Certificate configuration:
  - Certs directory: `"./certs"`
  - CA certificate: `"ca.crt"`
  - CA private key: `"ca.key"`
  - Server certificate: `"server.crt"`
  - Server private key: `"server.key"`

- External logger configuration
  - external_loggers:
   - litellm: `false`
   - sqlalchemy: `false`
   - uvicorn.error: `false`
   - aiosqlite: `false`

## Configuration methods

### Configuration file

Load configuration from a YAML file:

```python
config = Config.from_file("config.yaml")
```

Example config.yaml:

```yaml
port: 8989
proxy_port: 8990
host: localhost
log_level: INFO
log_format: JSON
provider_urls:
  vllm: "https://vllm.example.com"
  openai: "https://api.openai.com/v1"
  anthropic: "https://api.anthropic.com/v1"
  ollama: "http://localhost:11434"
certs_dir: "./certs"
ca_cert: "ca.crt"
ca_key: "ca.key"
server_cert: "server.crt"
server_key: "server.key"
```

### Environment variables

Environment variables are automatically loaded with these mappings:

- `CODEGATE_APP_PORT`: server port
- `CODEGATE_APP_PROXY_PORT`: server proxy port
- `CODEGATE_APP_HOST`: server host
- `CODEGATE_APP_LOG_LEVEL`: logging level
- `CODEGATE_LOG_FORMAT`: log format
- `CODEGATE_PROMPTS_FILE`: path to prompts YAML file
- `CODEGATE_PROVIDER_VLLM_URL`: vLLM provider URL
- `CODEGATE_PROVIDER_OPENAI_URL`: OpenAI provider URL
- `CODEGATE_PROVIDER_ANTHROPIC_URL`: Anthropic provider URL
- `CODEGATE_PROVIDER_OLLAMA_URL`: Ollama provider URL
- `CODEGATE_CERTS_DIR`: directory for certificate files
- `CODEGATE_CA_CERT`: CA certificate file name
- `CODEGATE_CA_KEY`: CA key file name
- `CODEGATE_SERVER_CERT`: server certificate file name
- `CODEGATE_SERVER_KEY`: server key file name
- `CODEGATE_ENABLE_LITELLM`: enable LiteLLM logging
- `CODEGATE_ENABLE_SQLALCHEMY`: enable SQLAlchemy logging
- `CODEGATE_ENABLE_UVICORN_ERROR`: enable Uvicorn error logging
- `CODEGATE_ENABLE_AIOSQLITE`: enable aiosqlite logging

```python
config = Config.from_env()
```

## Configuration options

### Network settings

Network settings can be configured in several ways:

1. Configuration file:

   ```yaml
   port: 8989 # Port to listen on (1-65535)
   proxy_port: 8990 # Proxy port to listen on (1-65535)
   host: "localhost" # Host to bind to
   ```

2. Environment variables:

   ```bash
   export CODEGATE_APP_PORT=8989
   export CODEGATE_APP_PROXY_PORT=8990
   export CODEGATE_APP_HOST=localhost
   ```

3. CLI flags:

   ```bash
   codegate serve --port 8989 --proxy-port 8990 --host localhost
   ```

### Provider URLs

Provider URLs can be configured in several ways:

1. Configuration file:

   ```yaml
   provider_urls:
     vllm: "https://vllm.example.com" # /v1 path is added automatically
     openai: "https://api.openai.com/v1"
     anthropic: "https://api.anthropic.com/v1"
     ollama: "http://localhost:11434" # /api path is added automatically
   ```

2. Environment variables:

   ```bash
   export CODEGATE_PROVIDER_VLLM_URL=https://vllm.example.com
   export CODEGATE_PROVIDER_OPENAI_URL=https://api.openai.com/v1
   export CODEGATE_PROVIDER_ANTHROPIC_URL=https://api.anthropic.com/v1
   export CODEGATE_PROVIDER_OLLAMA_URL=http://localhost:11434
   ```

3. CLI flags:

   ```bash
   codegate serve --vllm-url https://vllm.example.com --ollama-url http://localhost:11434
   ```

Note:

- For the vLLM provider, the `/v1` path is automatically appended to the base
  URL if not present.
- For the Ollama provider, the `/api` path is automatically appended to the base
  URL if not present.

### Certificate configuration

Certificate files can be configured in several ways:

1. Configuration file:

   ```yaml
   certs_dir: "./certs"
   ca_cert: "ca.crt"
   ca_key: "ca.key"
   server_cert: "server.crt"
   server_key: "server.key"
   ```

2. Environment variables:

   ```bash
   export CODEGATE_CERTS_DIR=./certs
   export CODEGATE_CA_CERT=ca.crt
   export CODEGATE_CA_KEY=ca.key
   export CODEGATE_SERVER_CERT=server.crt
   export CODEGATE_SERVER_KEY=server.key
   ```

3. CLI flags:

   ```bash
   codegate serve --certs-dir ./certs --ca-cert ca.crt --ca-key ca.key --server-cert server.crt --server-key server.key
   ```

### Log levels

Available log levels (case-insensitive):

- `ERROR`
- `WARNING`
- `INFO`
- `DEBUG`

### Log formats

Available log formats (case-insensitive):

- `JSON`
- `TEXT`

### External logger configuration

External loggers can be configured in several ways:

1. Configuration file:

   ```yaml
   external_loggers:
     litellm: false
     sqlalchemy: false
     uvicorn.error: false
     aiosqlite: false
   ```

### Prompts configuration

Prompts can be configured in several ways:

1. Default prompts:

   - Located in `prompts/default.yaml`
   - Loaded automatically if no other prompts are specified

2. Configuration file:

   ```yaml
   # Option 1: Direct prompts definition
   prompts:
     my_prompt: "Custom prompt text"
     another_prompt: "Another prompt text"

   # Option 2: Reference to prompts file
   prompts: "path/to/prompts.yaml"
   ```

3. Environment variable:

   ```bash
   export CODEGATE_PROMPTS_FILE=path/to/prompts.yaml
   ```

4. CLI flag:

   ```bash
   codegate serve --prompts path/to/prompts.yaml
   ```

### Prompts file format

Prompts files are defined in YAML format with string values:

```yaml
prompt_name: "Prompt text content"

another_prompt: "More prompt text"

multiline_prompt: |
  This is a multi-line prompt.
  It can span multiple lines.
```

Access prompts in code:

```python
config = Config.load()
prompt = config.prompts.prompt_name
```

## Error handling

The configuration system uses a custom `ConfigurationError` exception for
handling configuration-related errors, such as:

- Invalid port numbers (must be between 1 and 65535)
- Invalid proxy port numbers (must be between 1 and 65535)
- Invalid [log levels](#log-levels)
- Invalid [log formats](#log-formats)
- YAML parsing errors
- File reading errors
- Invalid prompt values (must be strings)
- Missing or invalid [prompts files](#prompts-file-format)
