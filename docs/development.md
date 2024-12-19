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

- Python 3.11 or higher
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

3. Install project dependencies:

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

Run the test suite with coverage:

```bash
poetry run pytest
```

Tests are located in the `tests/` directory and follow the same structure as the
source code.

### 4. Make commands

The project includes a Makefile for common development tasks:

- `make install`: install all dependencies
- `make format`: format code using black and ruff
- `make lint`: run linting checks
- `make test`: run tests with coverage
- `make security`: run security checks
- `make build`: build distribution packages
- `make all`: run all checks and build (recommended before committing)

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
