# Development guide

This guide provides comprehensive information for developers working on the
CodeGate project.

## Project overview

CodeGate is a configurable generative AI gateway designed to protect developers
from potential AI-related security risks. Key features include:

- Secrets exfiltration prevention
- Secure coding recommendations
- Prevention of AI recommending deprecated/malicious libraries
- Modular system prompts configuration
- Multiple AI provider support with configurable endpoints

## Development setup

### Prerequisites

- Python 3.12 or higher
- [Poetry](https://python-poetry.org/docs/#installation) for dependency
  management
- [Docker](https://docs.docker.com/get-docker/) or
  [Podman](https://podman.io/getting-started/installation) (for containerized
  deployment)
- [Visual Studio Code](https://code.visualstudio.com/download) (recommended IDE)

### Initial setup

1. Clone the repository:

   ```bash
   git clone https://github.com/stacklok/codegate.git
   cd codegate
   ```

2. Install Poetry following the
   [official installation guide](https://python-poetry.org/docs/#installation)

3. Set up virtual environment

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install project dependencies:

   ```bash
   poetry install --with dev
   ```

## Dashboard CodeGate UI 
### Setting up local development environment

Clone the repository
   ```bash
      git clone https://github.com/stacklok/codegate-ui
      cd codegate-ui
   ```

To install all dependencies for your local development environment, run

```bash
npm install
```

### Running the development server

Run the development server using:

```bash
npm run dev
```

### Build production

Run the build command:

```bash
npm run build
```

### Running production build

Run the production build command:

```bash
npm run preview
```

## Project structure

```plain
codegate/
├── pyproject.toml    # Project configuration and dependencies
├── poetry.lock      # Lock file (committed to version control)
├── prompts/         # System prompts configuration
│   └── default.yaml # Default system prompts
├── src/
│   └── codegate/    # Source code
│       ├── __init__.py
│       ├── cli.py           # Command-line interface
│       ├── config.py        # Configuration management
│       ├── exceptions.py    # Shared exceptions
│       ├── codegate_logging.py       # Logging setup
│       ├── prompts.py       # Prompts management
│       ├── server.py        # Main server implementation
│       └── providers/       # External service providers
│           ├── anthropic/   # Anthropic provider implementation
│           ├── openai/      # OpenAI provider implementation
│           ├── vllm/        # vLLM provider implementation
│           └── base.py      # Base provider interface
├── tests/           # Test files
└── docs/            # Documentation
```

## Development workflow

### 1. Environment management

Poetry commands for managing your development environment:

- `poetry install`: Install project dependencies
- `poetry add package-name`: Add a new package dependency
- `poetry add --group dev package-name`: Add a development dependency
- `poetry remove package-name`: Remove a package
- `poetry update`: Update dependencies to their latest versions
- `poetry show`: List all installed packages
- `poetry env info`: Show information about the virtual environment

### 2. Code style and quality

The project uses several tools to maintain code quality:

- [**Black**](https://black.readthedocs.io/en/stable/) for code formatting:

  ```bash
  poetry run black .
  ```

- [**Ruff**](https://docs.astral.sh/ruff/) for linting:

  ```bash
  poetry run ruff check .
  ```

- [**Bandit**](https://bandit.readthedocs.io/) for security checks:

  ```bash
  poetry run bandit -r src/
  ```

### 3. Testing

#### Unit Tests
To run the unit test suite with coverage:

```bash
poetry run pytest
```

Tests are located in the `tests/` directory and follow the same structure as the
source code.

#### Integration Tests
To run the integration tests, create a `.env` file in the repo root directory and add the
following properties to it:
```
ENV_OPENAI_KEY=<YOUR_KEY>
ENV_VLLM_KEY=<YOUR_KEY>
ENV_ANTHROPIC_KEY=<YOUR_KEY>
```

Then the integration tests can be executed by running:
```bash
python tests/integration/integration_tests.py
```

You can include additional properties to specify test scope and other information. For instance, to execute the tests for Copilot providers, for instance, run:
```bash
CODEGATE_PROVIDERS=copilot CA_CERT_FILE=./codegate_volume/certs/ca.crt python tests/integration/integration_tests.py
```

### 4. Make commands

The project includes a Makefile for common development tasks:

- `make install`: install all dependencies
- `make format`: format code using black and ruff
- `make lint`: run linting checks
- `make test`: run tests with coverage
- `make security`: run security checks
- `make build`: build distribution packages
- `make all`: run all checks and build (recommended before committing)

## 🐳 Docker deployment

### Build the image

```bash
make image-build
```

### Run the container

```bash
# Basic usage with local image
docker run -p 8989:8989 -p 9090:9090 codegate:latest

# With pre-built pulled image
docker pull ghcr.io/stacklok/codegate:latest
docker run --name codegate -d -p 8989:8989 -p 9090:9090 ghcr.io/stacklok/codegate:latest

# It will mount a volume to /app/codegate_volume
# The directory supports storing Llama CPP models under subdirectory /models
# A sqlite DB with the messages and alerts is stored under the subdirectory /db
docker run --name codegate -d -v /path/to/volume:/app/codegate_volume -p 8989:8989 -p 9090:9090 ghcr.io/stacklok/codegate:latest
```

### Exposed parameters

- CODEGATE_VLLM_URL: URL for the inference engine (defaults to
  [https://inference.codegate.ai](https://inference.codegate.ai))
- CODEGATE_OPENAI_URL: URL for OpenAI inference engine (defaults to
  [https://api.openai.com/v1](https://api.openai.com/v1))
- CODEGATE_ANTHROPIC_URL: URL for Anthropic inference engine (defaults to
  [https://api.anthropic.com/v1](https://api.anthropic.com/v1))
- CODEGATE_OLLAMA_URL: URL for OLlama inference engine (defaults to
  [http://localhost:11434/api](http://localhost:11434/api))
- CODEGATE_APP_LOG_LEVEL: Level of debug desired when running the codegate
  server (defaults to WARNING, can be ERROR/WARNING/INFO/DEBUG)
- CODEGATE_LOG_FORMAT: Type of log formatting desired when running the codegate
  server (default to TEXT, can be JSON/TEXT)

```bash
docker run -p 8989:8989 -p 9090:9090 -e CODEGATE_OLLAMA_URL=http://1.2.3.4:11434/api ghcr.io/stacklok/codegate:latest
```

## Configuration system

CodeGate uses a hierarchical configuration system with the following priority
(highest to lowest):

1. CLI arguments
2. Environment variables
3. Config file (YAML)
4. Default values (including default prompts)

### Configuration options

- Port: server port (default: `8989`)
- Host: server host (default: `"localhost"`)
- Log level: logging verbosity level (`ERROR`|`WARNING`|`INFO`|`DEBUG`)
- Log format: log format (`JSON`|`TEXT`)
- Prompts: system prompts configuration
- Provider URLs: AI provider endpoint configuration

See [Configuration system](configuration.md) for detailed information.

## Working with providers

CodeGate supports multiple AI providers through a modular provider system.

### Available providers

1. **vLLM provider**

   - Default URL: `http://localhost:8000`
   - Supports OpenAI-compatible APIs
   - Automatically adds `/v1` path to base URL
   - Model names are prefixed with `hosted_vllm/`

2. **OpenAI provider**

   - Default URL: `https://api.openai.com/v1`
   - Standard OpenAI API implementation

3. **Anthropic provider**

   - Default URL: `https://api.anthropic.com/v1`
   - Anthropic Claude API implementation

4. **Ollama provider**
   - Default URL: `http://localhost:11434`
   - Endpoints:
     - Native Ollama API: `/ollama/api/chat`
     - OpenAI-compatible: `/ollama/chat/completions`

### Configuring providers

Provider URLs can be configured through:

1. Config file (config.yaml):

   ```yaml
   provider_urls:
     vllm: "https://vllm.example.com"
     openai: "https://api.openai.com/v1"
     anthropic: "https://api.anthropic.com/v1"
     ollama: "http://localhost:11434" # /api path added automatically
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

### Implementing new providers

To add a new provider:

1. Create a new directory in `src/codegate/providers/`
2. Implement required components:
   - `provider.py`: Main provider class extending BaseProvider
   - `adapter.py`: Input/output normalizers
   - `__init__.py`: Export provider class

Example structure:

```python
from codegate.providers.base import BaseProvider

class NewProvider(BaseProvider):
    def __init__(self, ...):
        super().__init__(
            InputNormalizer(),
            OutputNormalizer(),
            completion_handler,
            pipeline_processor,
            fim_pipeline_processor
        )

    @property
    def provider_route_name(self) -> str:
        return "provider_name"

    def _setup_routes(self):
        # Implement route setup
        pass
```

## Working with prompts

### Default prompts

Default prompts are stored in `prompts/default.yaml`. These prompts are loaded
automatically when no other prompts are specified.

### Creating custom prompts

1. Create a new YAML file following the format:

   ```yaml
   prompt_name: "Prompt text content"
   another_prompt: "More prompt text"
   ```

2. Use the prompts file:

   ```bash
   # Via CLI
   codegate serve --prompts my-prompts.yaml

   # Via config.yaml
   prompts: "path/to/prompts.yaml"

   # Via environment
   export CODEGATE_PROMPTS_FILE=path/to/prompts.yaml
   ```

### Testing prompts

1. View loaded prompts:

   ```bash
   # Show default prompts
   codegate show-prompts

   # Show custom prompts
   codegate show-prompts --prompts my-prompts.yaml
   ```

2. Write tests for prompt functionality:

   ```python
   def test_custom_prompts():
       config = Config.load(prompts_path="path/to/test/prompts.yaml")
       assert config.prompts.my_prompt == "Expected prompt text"
   ```

## CLI interface

The main command-line interface is implemented in `cli.py`. Basic usage:

```bash
# Start server with default settings
codegate serve

# Start with custom configuration
codegate serve --port 8989 --host localhost --log-level DEBUG

# Start with custom prompts
codegate serve --prompts my-prompts.yaml

# Start with custom provider URL
codegate serve --vllm-url https://vllm.example.com
```

See [CLI commands and flags](cli.md) for detailed command information.
