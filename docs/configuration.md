# Configuration

CodeGate's configuration system provides flexible configuration through multiple
methods with clear priority resolution.

## Configuration methods

Configuration can be set through:

1. CLI arguments (highest priority)
2. Environment variables
3. Configuration file
4. Default values (lowest priority)

## Configuration file

The configuration file uses YAML format:

```yaml
# Network settings
port: 8989
proxy_port: 8990
host: "localhost"

# Logging configuration
log_level: INFO  # ERROR, WARNING, INFO, or DEBUG
log_format: JSON # JSON or TEXT

# External logger configuration
external_loggers:
  litellm: false      # Enable/disable LiteLLM logging (includes LiteLLM Proxy, Router, and core)
  sqlalchemy: false   # Enable/disable SQLAlchemy logging
  uvicorn.error: false # Enable/disable Uvicorn error logging
  aiosqlite: false    # Enable/disable aiosqlite logging

# Model configuration
model_base_path: "./codegate_volume/models"
chat_model_n_ctx: 32768
chat_model_n_gpu_layers: -1
embedding_model: "all-minilm-L6-v2-q5_k_m.gguf"

# Database configuration
db_path: "./codegate_volume/db/codegate.db"
vec_db_path: "./sqlite_data/vectordb.db"

# Certificate configuration
certs_dir: "./codegate_volume/certs"
ca_cert: "ca.crt"
ca_key: "ca.key"
server_cert: "server.crt"
server_key: "server.key"
force_certs: false

# Provider URLs
provider_urls:
  vllm: "http://localhost:8000"
  openai: "https://api.openai.com/v1"
  anthropic: "https://api.anthropic.com/v1"
  ollama: "http://localhost:11434"
```

## Environment variables

Environment variables follow the pattern `CODEGATE_*`:

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

### Logging configuration

Logging can be configured through:

1. Configuration file:

   ```yaml
   log_level: DEBUG
   log_format: TEXT
   external_loggers:
     litellm: true
     sqlalchemy: false
     uvicorn.error: false
     aiosqlite: false
   ```

2. Environment variables:

   ```bash
   export CODEGATE_APP_LOG_LEVEL=DEBUG
   export CODEGATE_LOG_FORMAT=TEXT
   export CODEGATE_ENABLE_LITELLM=true
   ```

3. CLI flags:

   ```bash
   codegate serve --log-level DEBUG --log-format TEXT --enable-litellm
   ```

### Provider URLs

Provider URLs can be configured through:

1. Configuration file:

   ```yaml
   provider_urls:
     vllm: "http://localhost:8000"
     openai: "https://api.openai.com/v1"
     anthropic: "https://api.anthropic.com/v1"
     ollama: "http://localhost:11434"
   ```

2. Environment variables:

   ```bash
   export CODEGATE_PROVIDER_VLLM_URL=http://localhost:8000
   export CODEGATE_PROVIDER_OPENAI_URL=https://api.openai.com/v1
   export CODEGATE_PROVIDER_ANTHROPIC_URL=https://api.anthropic.com/v1
   export CODEGATE_PROVIDER_OLLAMA_URL=http://localhost:11434
   ```

3. CLI flags:

   ```bash
   codegate serve --vllm-url http://localhost:8000 --openai-url https://api.openai.com/v1
   ```

### Certificate configuration

Certificate settings can be configured through:

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

### External loggers

External logger configuration controls logging for third-party components:

1. Configuration file:
   ```yaml
   external_loggers:
     litellm: false      # LiteLLM logging (Proxy, Router, core)
     sqlalchemy: false   # SQLAlchemy logging
     uvicorn.error: false # Uvicorn error logging
     aiosqlite: false    # aiosqlite logging
   ```

2. Environment variables:
   ```bash
   export CODEGATE_ENABLE_LITELLM=true
   export CODEGATE_ENABLE_SQLALCHEMY=true
   export CODEGATE_ENABLE_UVICORN_ERROR=true
   export CODEGATE_ENABLE_AIOSQLITE=true
   ```

3. CLI flags:
   ```bash
   codegate serve --enable-litellm
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
