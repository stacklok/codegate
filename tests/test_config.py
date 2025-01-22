"""Tests for configuration management."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from codegate.config import DEFAULT_PROVIDER_URLS, Config, ConfigurationError, LogFormat, LogLevel


def test_default_config(default_config: Config) -> None:
    """Test default configuration values."""
    assert default_config.port == 8989
    assert default_config.host == "localhost"
    assert default_config.log_level == LogLevel.INFO
    assert default_config.log_format == LogFormat.JSON
    assert default_config.provider_urls == DEFAULT_PROVIDER_URLS
    # Test default certificate configuration
    assert default_config.certs_dir == "./codegate_volume/certs"
    assert default_config.ca_cert == "ca.crt"
    assert default_config.ca_key == "ca.key"
    assert default_config.server_cert == "server.crt"
    assert default_config.server_key == "server.key"


def test_config_from_file(temp_config_file: Path) -> None:
    """Test loading configuration from file."""
    config = Config.from_file(temp_config_file)
    assert config.port == 8989
    assert config.host == "localhost"
    assert config.log_level == LogLevel.DEBUG
    assert config.log_format == LogFormat.JSON
    assert config.provider_urls == DEFAULT_PROVIDER_URLS


def test_config_from_invalid_file(tmp_path: Path) -> None:
    """Test loading configuration from invalid file."""
    invalid_file = tmp_path / "invalid.yaml"
    with open(invalid_file, "w") as f:
        f.write("invalid: yaml: content")

    with pytest.raises(ConfigurationError):
        Config.from_file(invalid_file)


def test_config_from_nonexistent_file() -> None:
    """Test loading configuration from nonexistent file."""
    with pytest.raises(ConfigurationError):
        Config.from_file("nonexistent.yaml")


def test_config_from_env(env_vars: None) -> None:
    """Test loading configuration from environment variables."""
    config = Config.from_env()
    assert config.port == 8989
    assert config.host == "localhost"
    assert config.log_level == LogLevel.WARNING
    assert config.log_format == LogFormat.TEXT
    assert config.provider_urls == DEFAULT_PROVIDER_URLS


def test_config_priority_resolution(temp_config_file: Path, env_vars: None) -> None:
    """Test configuration priority resolution."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        custom_certs_dir = os.path.join(tmpdirname, "custom-certs")
        os.makedirs(custom_certs_dir)

        # CLI args should override everything
        config = Config.load(
            config_path=temp_config_file,
            cli_port=8080,
            cli_host="example.com",
            cli_log_level="WARNING",
            cli_log_format="TEXT",
            cli_provider_urls={"vllm": "https://custom.vllm.server"},
            certs_dir=custom_certs_dir,
            ca_cert="custom-ca.crt",
            ca_key="custom-ca.key",
            server_cert="custom-server.crt",
            server_key="custom-server.key",
        )
        assert config.port == 8080
        assert config.host == "example.com"
        assert config.log_level == LogLevel.WARNING
        assert config.log_format == LogFormat.TEXT
        assert config.provider_urls["vllm"] == "https://custom.vllm.server"
        assert config.certs_dir == custom_certs_dir
        assert config.ca_cert == "custom-ca.crt"
        assert config.ca_key == "custom-ca.key"
        assert config.server_cert == "custom-server.crt"
        assert config.server_key == "custom-server.key"

        # Env vars should override config file
        config = Config.load(config_path=temp_config_file)
        assert config.port == 8989  # from env
        assert config.host == "localhost"  # from env
        assert config.log_level == LogLevel.WARNING  # from env
        assert config.log_format == LogFormat.TEXT  # from env
        assert config.provider_urls == DEFAULT_PROVIDER_URLS  # no env override

        # Config file should override defaults
        os.environ.clear()  # Remove env vars
        config = Config.load(config_path=temp_config_file)
        assert config.port == 8989  # from file
        assert config.host == "localhost"  # from file
        assert config.log_level == LogLevel.DEBUG  # from file
        assert config.log_format == LogFormat.JSON  # from file
        assert config.provider_urls == DEFAULT_PROVIDER_URLS  # default values


def test_provider_urls_from_config(tmp_path: Path) -> None:
    """Test loading provider URLs from config file."""
    config_file = tmp_path / "config.yaml"
    custom_urls = {
        "vllm": "https://custom.vllm.server",
        "openai": "https://custom.openai.server",
    }
    with open(config_file, "w") as f:
        yaml.dump({"provider_urls": custom_urls}, f)

    config = Config.from_file(config_file)
    assert config.provider_urls["vllm"] == custom_urls["vllm"]
    assert config.provider_urls["openai"] == custom_urls["openai"]
    assert config.provider_urls["anthropic"] == DEFAULT_PROVIDER_URLS["anthropic"]


def test_provider_urls_from_env() -> None:
    """Test loading provider URLs from environment variables."""
    os.environ["CODEGATE_PROVIDER_VLLM_URL"] = "https://custom.vllm.server"
    try:
        config = Config.from_env()
        assert config.provider_urls["vllm"] == "https://custom.vllm.server"
        assert config.provider_urls["openai"] == DEFAULT_PROVIDER_URLS["openai"]
    finally:
        del os.environ["CODEGATE_PROVIDER_VLLM_URL"]


def test_certificate_config_from_file(tmp_path: Path) -> None:
    """Test loading certificate configuration from file."""
    config_file = tmp_path / "config.yaml"
    cert_config = {
        "certs_dir": "./custom-certs",
        "ca_cert": "custom-ca.crt",
        "ca_key": "custom-ca.key",
        "server_cert": "custom-server.crt",
        "server_key": "custom-server.key",
    }
    with open(config_file, "w") as f:
        yaml.dump(cert_config, f)

    config = Config.from_file(config_file)
    assert config.certs_dir == cert_config["certs_dir"]
    assert config.ca_cert == cert_config["ca_cert"]
    assert config.ca_key == cert_config["ca_key"]
    assert config.server_cert == cert_config["server_cert"]
    assert config.server_key == cert_config["server_key"]


def test_certificate_config_from_env() -> None:
    """Test loading certificate configuration from environment variables."""
    os.environ.update(
        {
            "CODEGATE_CERTS_DIR": "./env-certs",
            "CODEGATE_CA_CERT": "env-ca.crt",
            "CODEGATE_CA_KEY": "env-ca.key",
            "CODEGATE_SERVER_CERT": "env-server.crt",
            "CODEGATE_SERVER_KEY": "env-server.key",
        }
    )
    try:
        config = Config.from_env()
        assert config.certs_dir == "./env-certs"
        assert config.ca_cert == "env-ca.crt"
        assert config.ca_key == "env-ca.key"
        assert config.server_cert == "env-server.crt"
        assert config.server_key == "env-server.key"
    finally:
        for key in [
            "CODEGATE_CERTS_DIR",
            "CODEGATE_CA_CERT",
            "CODEGATE_CA_KEY",
            "CODEGATE_SERVER_CERT",
            "CODEGATE_SERVER_KEY",
        ]:
            os.environ.pop(key, None)


def test_invalid_log_level() -> None:
    """Test handling of invalid log level."""
    with pytest.raises(ConfigurationError):
        Config(log_level="INVALID")


def test_invalid_log_format() -> None:
    """Test handling of invalid log format."""
    with pytest.raises(ConfigurationError):
        Config(log_format="INVALID")


def test_invalid_port() -> None:
    """Test handling of invalid port number."""
    with pytest.raises(ConfigurationError):
        Config(port=0)
    with pytest.raises(ConfigurationError):
        Config(port=65536)


def test_log_format_case_insensitive(tmp_path: Path) -> None:
    """Test log format is case insensitive."""
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump({"log_format": "json"}, f)

    config = Config.from_file(config_file)
    assert config.log_format == LogFormat.JSON

    with open(config_file, "w") as f:
        yaml.dump({"log_format": "TEXT"}, f)

    config = Config.from_file(config_file)
    assert config.log_format == LogFormat.TEXT


@pytest.fixture
def config_file_with_format(tmp_path: Path) -> Path:
    """Create a config file with log format."""
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump({"log_format": "TEXT", "log_level": "DEBUG"}, f)
    return config_file


def test_env_var_priority(config_file_with_format: Path) -> None:
    """Test environment variable priority for log format."""
    os.environ["CODEGATE_LOG_FORMAT"] = "JSON"
    try:
        config = Config.load(config_path=config_file_with_format)
        assert config.log_format == LogFormat.JSON  # env var overrides file
    finally:
        del os.environ["CODEGATE_LOG_FORMAT"]


def test_external_loggers_from_env() -> None:
    """Test loading external logger configuration from environment variables."""
    os.environ.update(
        {
            "CODEGATE_ENABLE_LITELLM": "true",
            "CODEGATE_ENABLE_SQLALCHEMY": "true",
            "CODEGATE_ENABLE_UVICORN_ERROR": "false",
            "CODEGATE_ENABLE_AIOSQLITE": "false",
        }
    )
    try:
        config = Config.from_env()
        assert config.external_loggers["litellm"] is True
        assert config.external_loggers["sqlalchemy"] is True
        assert config.external_loggers["uvicorn.error"] is False
        assert config.external_loggers["aiosqlite"] is False
    finally:
        for key in [
            "CODEGATE_ENABLE_LITELLM",
            "CODEGATE_ENABLE_SQLALCHEMY",
            "CODEGATE_ENABLE_UVICORN_ERROR",
            "CODEGATE_ENABLE_AIOSQLITE",
        ]:
            os.environ.pop(key, None)


def test_external_loggers_from_config_file(tmp_path: Path) -> None:
    """Test loading external logger configuration from config file."""
    config_file = tmp_path / "config.yaml"
    config_data = {
        "external_loggers": {
            "litellm": True,
            "sqlalchemy": False,
            "uvicorn.error": True,
            "aiosqlite": False,
        }
    }
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    config = Config.from_file(config_file)
    assert config.external_loggers["litellm"] is True
    assert config.external_loggers["sqlalchemy"] is False
    assert config.external_loggers["uvicorn.error"] is True
    assert config.external_loggers["aiosqlite"] is False


def test_external_loggers_defaults() -> None:
    """Test default values for external loggers."""
    config = Config()
    assert config.external_loggers["litellm"] is False
    assert config.external_loggers["sqlalchemy"] is False
    assert config.external_loggers["uvicorn.error"] is False
    assert config.external_loggers["aiosqlite"] is False


def test_external_loggers_env_override_config(tmp_path: Path) -> None:
    """Test environment variables override config file for external loggers."""
    config_file = tmp_path / "config.yaml"
    config_data = {
        "external_loggers": {
            "litellm": False,
            "sqlalchemy": False,
            "uvicorn.error": False,
            "aiosqlite": False,
        }
    }
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    os.environ["CODEGATE_ENABLE_LITELLM"] = "true"
    try:
        config = Config.load(config_path=config_file)
        assert config.external_loggers["litellm"] is True  # env var overrides file
        assert config.external_loggers["sqlalchemy"] is False  # from file
        assert config.external_loggers["uvicorn.error"] is False  # from file
        assert config.external_loggers["aiosqlite"] is False  # from file
    finally:
        del os.environ["CODEGATE_ENABLE_LITELLM"]
