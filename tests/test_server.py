"""Tests for the server module."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from click.testing import CliRunner
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from httpx import AsyncClient
from uvicorn.config import Config as UvicornConfig

from codegate import __version__
from codegate.pipeline.factory import PipelineFactory
from codegate.pipeline.sensitive_data.manager import SensitiveDataManager
from codegate.providers.registry import ProviderRegistry
from codegate.server import init_app
from src.codegate.cli import UvicornServer, cli
from src.codegate.codegate_logging import LogFormat, LogLevel


@pytest.fixture
def mock_provider_registry():
    """Create a mock provider registry."""
    return MagicMock(spec=ProviderRegistry)


@pytest.fixture
def mock_pipeline_factory():
    """Create a mock pipeline factory."""
    mock_factory = MagicMock(spec=PipelineFactory)
    # Mock the methods that are called on the pipeline factory
    mock_factory.create_input_pipeline.return_value = MagicMock()
    mock_factory.create_fim_pipeline.return_value = MagicMock()
    mock_factory.create_output_pipeline.return_value = MagicMock()
    mock_factory.create_fim_output_pipeline.return_value = MagicMock()
    return mock_factory


@pytest.fixture
def test_client(mock_pipeline_factory) -> TestClient:
    """Create a test client for the FastAPI application."""
    app = init_app(mock_pipeline_factory)
    return TestClient(app)


def test_app_initialization(mock_pipeline_factory) -> None:
    """Test that the FastAPI application initializes correctly."""
    app = init_app(mock_pipeline_factory)
    assert app is not None
    assert app.title == "CodeGate"
    assert app.version == __version__


def test_cors_middleware(mock_pipeline_factory) -> None:
    """Test that CORS middleware is properly configured."""
    app = init_app(mock_pipeline_factory)
    cors_middleware = None
    for middleware in app.user_middleware:
        if isinstance(middleware.cls, type) and issubclass(middleware.cls, CORSMiddleware):
            cors_middleware = middleware
            break
    assert cors_middleware is not None
    assert cors_middleware.kwargs.get("allow_origins") == ["*"]
    assert cors_middleware.kwargs.get("allow_credentials") is True
    assert cors_middleware.kwargs.get("allow_methods") == ["*"]
    assert cors_middleware.kwargs.get("allow_headers") == ["*"]


def test_health_check(test_client: TestClient) -> None:
    """Test the health check endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@patch("codegate.api.v1_processing.fetch_latest_version", return_value="foo")
def test_version_endpoint(mock_fetch_latest_version, test_client: TestClient) -> None:
    """Test the version endpoint."""
    response = test_client.get("/api/v1/version")
    assert response.status_code == 200

    response_data = response.json()

    assert response_data["current_version"] == __version__.lstrip("v")
    assert response_data["latest_version"] == "foo"
    assert isinstance(response_data["is_latest"], bool)
    assert response_data["is_latest"] is False


@patch("codegate.pipeline.sensitive_data.manager.SensitiveDataManager")
@patch("codegate.server.get_provider_registry")
def test_provider_registration(mock_registry, mock_pipeline_factory) -> None:
    """Test that all providers are registered correctly."""
    init_app(mock_pipeline_factory)

    # Verify ProviderRegistry was initialized with the app
    mock_registry.assert_called_once()

    # Verify all providers were registered
    registry_instance = mock_registry.return_value
    assert (
        registry_instance.add_provider.call_count == 7
    )  # openai, anthropic, llamacpp, vllm, ollama, lm_studio, openrouter

    # Verify specific providers were registered
    provider_names = [call.args[0] for call in registry_instance.add_provider.call_args_list]
    assert "openai" in provider_names
    assert "anthropic" in provider_names
    assert "llamacpp" in provider_names
    assert "vllm" in provider_names
    assert "ollama" in provider_names


def test_pipeline_initialization(mock_pipeline_factory) -> None:
    """Test that pipelines are initialized correctly."""
    app = init_app(mock_pipeline_factory)

    # Access the provider registry to check pipeline configuration
    registry = next((route for route in app.routes if hasattr(route, "registry")), None)

    if registry:
        for provider in registry.registry.values():
            # Verify each provider has the required pipelines
            assert hasattr(provider, "pipeline_processor")
            assert hasattr(provider, "fim_pipeline_processor")
            assert hasattr(provider, "output_pipeline_processor")


def test_workspaces_routes(mock_pipeline_factory) -> None:
    """Test that dashboard routes are included."""
    app = init_app(mock_pipeline_factory)
    routes = [route.path for route in app.routes]

    # Verify dashboard endpoints are included
    dashboard_routes = [route for route in routes if route.startswith("/api/v1/workspaces")]
    assert len(dashboard_routes) > 0


def test_system_routes(mock_pipeline_factory) -> None:
    """Test that system routes are included."""
    app = init_app(mock_pipeline_factory)
    routes = [route.path for route in app.routes]

    # Verify system endpoints are included
    assert "/health" in routes


@pytest.mark.asyncio
async def test_async_health_check(mock_pipeline_factory) -> None:
    """Test the health check endpoint with async client."""
    app = init_app(mock_pipeline_factory)
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


def test_error_handling(test_client: TestClient) -> None:
    """Test error handling for non-existent endpoints."""
    response = test_client.get("/nonexistent")
    assert response.status_code == 404

    # Test method not allowed
    response = test_client.post("/health")  # Health endpoint only allows GET
    assert response.status_code == 405


@pytest.fixture
def mock_app():
    # Create a simple mock for the ASGI application
    return MagicMock()


@pytest.fixture
def uvicorn_config(mock_app):
    # Assuming mock_app is defined to simulate ASGI application
    return UvicornConfig(app=mock_app, host="localhost", port=8000, log_level="info")


@pytest.fixture
def server_instance(uvicorn_config):
    with patch("src.codegate.cli.Server", autospec=True) as mock_server_class:
        mock_server_instance = mock_server_class.return_value
        mock_server_instance.serve = AsyncMock()
        yield UvicornServer(uvicorn_config, mock_server_instance)


@pytest.mark.asyncio
async def test_server_starts_and_runs(server_instance):
    await server_instance.serve()
    server_instance.server.serve.assert_awaited_once()


@pytest.fixture
def cli_runner():
    return CliRunner()


def test_serve_default_options(cli_runner):
    """Test serve command with default options."""
    # Use patches for run_servers and logging setup
    with (
        patch("src.codegate.cli.run_servers") as mock_run,
        patch("src.codegate.cli.setup_logging") as mock_setup_logging,
    ):
        # Invoke the CLI command
        result = cli_runner.invoke(cli, ["serve"])

        # Basic checks to ensure the command executed successfully
        assert result.exit_code == 0

        # Check if the logging setup was called with expected defaults
        mock_setup_logging.assert_called_once_with(LogLevel.INFO, LogFormat.JSON)

        # Validate run_servers was called once
        mock_run.assert_called_once()


def test_serve_custom_options(cli_runner):
    """Test serve command with custom options."""
    with (
        patch("src.codegate.cli.run_servers") as mock_run,
        patch("src.codegate.cli.setup_logging") as mock_setup_logging,
    ):
        # Invoke the CLI command with custom options
        result = cli_runner.invoke(
            cli,
            [
                "serve",
                "--port",
                "8989",
                "--host",
                "localhost",
                "--log-level",
                "DEBUG",
                "--log-format",
                "TEXT",
                "--certs-dir",
                "./custom-certs",
                "--ca-cert",
                "custom-ca.crt",
                "--ca-key",
                "custom-ca.key",
                "--server-cert",
                "custom-server.crt",
                "--server-key",
                "custom-server.key",
            ],
        )

        # Check the command executed successfully
        assert result.exit_code == 0

        # Assert logging setup was called with the provided log level and format
        mock_setup_logging.assert_called_once_with(LogLevel.DEBUG, LogFormat.TEXT)

        # Validate run_servers was called once
        mock_run.assert_called_once()
        # Retrieve the actual Config object passed to run_servers
        config_arg = mock_run.call_args[0][0]  # Assuming Config is the first positional arg

        # Define expected values that should be present in the Config object
        expected_values = {
            "port": 8989,
            "host": "localhost",
            "log_level": LogLevel.DEBUG,
            "log_format": LogFormat.TEXT,
            "certs_dir": "./custom-certs",
            "ca_cert": "custom-ca.crt",
            "ca_key": "custom-ca.key",
            "server_cert": "custom-server.crt",
            "server_key": "custom-server.key",
        }

        # Check if Config object attributes match the expected values
        for key, expected_value in expected_values.items():
            assert (
                getattr(config_arg, key) == expected_value
            ), f"{key} does not match expected value"


def test_serve_invalid_port(cli_runner):
    """Test serve command with invalid port."""
    result = cli_runner.invoke(cli, ["serve", "--port", "999999"])
    assert result.exit_code == 2  # Typically 2 is used for CLI errors in Click
    assert "Port must be between 1 and 65535" in result.output


def test_serve_invalid_log_level(cli_runner):
    """Test serve command with invalid log level."""
    result = cli_runner.invoke(cli, ["serve", "--log-level", "INVALID"])
    assert result.exit_code == 2
    assert "Invalid value for '--log-level'" in result.output


@pytest.fixture
def temp_config_file(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
    log_level: DEBUG
    log_format: JSON
    port: 8989
    host: localhost
    certs_dir: ./test-certs
    """
    )
    return config_path


def test_serve_with_config_file(cli_runner, temp_config_file):
    """Test serve command with config file."""
    with (
        patch("src.codegate.cli.run_servers") as mock_run,
        patch("src.codegate.cli.setup_logging") as mock_setup_logging,
    ):
        # Invoke the CLI command with the configuration file
        result = cli_runner.invoke(cli, ["serve", "--config", str(temp_config_file)])

        # Assertions to ensure the CLI ran successfully
        assert result.exit_code == 0
        mock_setup_logging.assert_called_once_with(LogLevel.DEBUG, LogFormat.JSON)

        # Validate that run_servers was called with the expected configuration
        mock_run.assert_called_once()
        config_arg = mock_run.call_args[0][0]

        # Define expected values based on the temp_config_file content
        expected_values = {
            "port": 8989,
            "host": "localhost",
            "log_level": LogLevel.DEBUG,
            "log_format": LogFormat.JSON,
            "certs_dir": "./test-certs",
        }

        # Check if passed arguments match the expected values
        for key, expected_value in expected_values.items():
            assert (
                getattr(config_arg, key) == expected_value
            ), f"{key} does not match expected value"


def test_serve_with_nonexistent_config_file(cli_runner: CliRunner) -> None:
    """Test serve command with nonexistent config file."""
    result = cli_runner.invoke(cli, ["serve", "--config", "nonexistent.yaml"])
    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_serve_priority_resolution(cli_runner: CliRunner, temp_config_file: Path) -> None:
    """Test serve command respects configuration priority."""
    # Set up environment variables and ensure they get cleaned up after the test
    with (
        patch.dict(os.environ, {"LOG_LEVEL": "INFO", "PORT": "9999"}, clear=True),
        patch("src.codegate.cli.run_servers") as mock_run,
        patch("src.codegate.cli.setup_logging") as mock_setup_logging,
    ):
        # Execute CLI command with specific options overriding environment and config file settings
        result = cli_runner.invoke(
            cli,
            [
                "serve",
                "--config",
                str(temp_config_file),
                "--port",
                "8080",
                "--host",
                "example.com",
                "--log-level",
                "ERROR",
                "--log-format",
                "TEXT",
                "--certs-dir",
                "./cli-certs",
                "--ca-cert",
                "cli-ca.crt",
                "--ca-key",
                "cli-ca.key",
                "--server-cert",
                "cli-server.crt",
                "--server-key",
                "cli-server.key",
            ],
        )

        # Check the result of the command
        assert result.exit_code == 0

        # Ensure logging setup was called with the highest priority settings (CLI arguments)
        mock_setup_logging.assert_called_once_with("ERROR", "TEXT")

        # Verify that the run_servers was called with the overridden settings
        config_arg = mock_run.call_args[0][0]  # Assuming Config is the first positional arg

        expected_values = {
            "port": 8080,
            "host": "example.com",
            "log_level": "ERROR",
            "log_format": "TEXT",
            "certs_dir": "./cli-certs",
            "ca_cert": "cli-ca.crt",
            "ca_key": "cli-ca.key",
            "server_cert": "cli-server.crt",
            "server_key": "cli-server.key",
        }

        # Verify if Config object attributes match the expected values from CLI arguments
        for key, expected_value in expected_values.items():
            assert (
                getattr(config_arg, key) == expected_value
            ), f"{key} does not match expected value"


def test_serve_certificate_options(cli_runner: CliRunner) -> None:
    """Test serve command with certificate options."""
    with (
        patch("src.codegate.cli.run_servers") as mock_run,
        patch("src.codegate.cli.setup_logging") as mock_setup_logging,
    ):
        # Execute CLI command with certificate options
        result = cli_runner.invoke(
            cli,
            [
                "serve",
                "--certs-dir",
                "./custom-certs",
                "--ca-cert",
                "custom-ca.crt",
                "--ca-key",
                "custom-ca.key",
                "--server-cert",
                "custom-server.crt",
                "--server-key",
                "custom-server.key",
            ],
        )

        # Check the result of the command
        assert result.exit_code == 0

        # Ensure logging setup was called with expected arguments
        mock_setup_logging.assert_called_once_with("INFO", "JSON")

        # Verify that run_servers was called with the provided certificate options
        config_arg = mock_run.call_args[0][0]  # Assuming Config is the first positional arg

        expected_values = {
            "certs_dir": "./custom-certs",
            "ca_cert": "custom-ca.crt",
            "ca_key": "custom-ca.key",
            "server_cert": "custom-server.crt",
            "server_key": "custom-server.key",
        }

        # Check if Config object attributes match the expected values
        for key, expected_value in expected_values.items():
            assert (
                getattr(config_arg, key) == expected_value
            ), f"{key} does not match expected value"


def test_main_function() -> None:
    """Test main function."""
    with patch("sys.argv", ["cli"]), patch("codegate.cli.cli") as mock_cli:
        from codegate.cli import main

        main()
        mock_cli.assert_called_once()


@pytest.fixture
def mock_uvicorn_server():
    mock_config = MagicMock()  # Setup the configuration mock
    mock_server = MagicMock(spec=UvicornServer)
    mock_server.shutdown = AsyncMock()  # Ensure shutdown is an async mock

    uvicorn_server = UvicornServer(config=mock_config, server=mock_server)
    return uvicorn_server


@pytest.mark.asyncio
async def test_uvicorn_server_cleanup(mock_uvicorn_server):
    with (
        patch("asyncio.get_running_loop"),
        patch.object(mock_uvicorn_server.server, "shutdown", AsyncMock()),
    ):
        # Mock the loop or other components as needed

        # Start the server or trigger the condition you want to test
        await mock_uvicorn_server.cleanup()  # This should now complete without error

        # Verify that the shutdown was called
        mock_uvicorn_server.server.shutdown.assert_awaited_once()
