"""Tests for the prompts module."""

from pathlib import Path

import pytest
import yaml

from codegate.config import Config
from codegate.exceptions import ConfigurationError
from codegate.prompts import PromptConfig


@pytest.fixture
def temp_prompts_file(tmp_path):
    """Create a temporary prompts file for testing."""
    prompts_data = {
        "test_prompt": "This is a test prompt",
        "another_prompt": "Another test prompt",
    }
    prompts_file = tmp_path / "test_prompts.yaml"
    with open(prompts_file, "w") as f:
        yaml.safe_dump(prompts_data, f)
    return prompts_file


@pytest.fixture
def temp_env_prompts_file(tmp_path):
    """Create a temporary prompts file for environment testing."""
    prompts_data = {
        "env_prompt": "This is an environment prompt",
        "another_env": "Another environment prompt",
    }
    prompts_file = tmp_path / "env_prompts.yaml"
    with open(prompts_file, "w") as f:
        yaml.safe_dump(prompts_data, f)
    return prompts_file


@pytest.fixture
def temp_config_file(tmp_path, temp_prompts_file):
    """Create a temporary config file for testing."""
    config_data = {
        "prompts": {
            "inline_prompt": "This is an inline prompt",
            "another_inline": "Another inline prompt",
        }
    }
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(config_data, f)
    return config_file


@pytest.fixture
def temp_config_with_prompts_file(tmp_path, temp_prompts_file):
    """Create a temporary config file that references a prompts file."""
    config_data = {
        "prompts": str(temp_prompts_file),
    }
    config_file = tmp_path / "test_config_with_prompts.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(config_data, f)
    return config_file


def test_default_prompts():
    """Test loading default prompts."""
    config = Config.load()
    assert len(config.prompts.prompts) > 0
    assert hasattr(config.prompts, "default_chat")
    assert "You are a helpful assistant" in config.prompts.default_chat


def test_cli_prompts_override_default(temp_prompts_file):
    """Test that CLI prompts override default prompts."""
    config = Config.load(prompts_path=temp_prompts_file)
    assert len(config.prompts.prompts) == 2
    assert config.prompts.test_prompt == "This is a test prompt"
    assert not hasattr(config.prompts, "default_chat")


def test_env_prompts_override_default(temp_env_prompts_file, monkeypatch):
    """Test that environment prompts override default prompts."""
    monkeypatch.setenv("CODEGATE_PROMPTS_FILE", str(temp_env_prompts_file))
    config = Config.load()
    assert len(config.prompts.prompts) == 2
    assert config.prompts.env_prompt == "This is an environment prompt"
    assert not hasattr(config.prompts, "default_chat")


def test_config_prompts_override_default(temp_config_file):
    """Test that config prompts override default prompts."""
    config = Config.load(config_path=temp_config_file)
    assert len(config.prompts.prompts) == 2
    assert config.prompts.inline_prompt == "This is an inline prompt"
    assert not hasattr(config.prompts, "default_chat")


def test_load_prompts_from_file(temp_prompts_file):
    """Test loading prompts from a YAML file."""
    config = Config.load(prompts_path=temp_prompts_file)
    assert len(config.prompts.prompts) == 2
    assert config.prompts.test_prompt == "This is a test prompt"
    assert config.prompts.another_prompt == "Another test prompt"


def test_load_prompts_from_config(temp_config_file):
    """Test loading inline prompts from config file."""
    config = Config.load(config_path=temp_config_file)
    assert len(config.prompts.prompts) == 2
    assert config.prompts.inline_prompt == "This is an inline prompt"
    assert config.prompts.another_inline == "Another inline prompt"


def test_load_prompts_from_config_file_reference(temp_config_with_prompts_file):
    """Test loading prompts from a file referenced in config."""
    config = Config.load(config_path=temp_config_with_prompts_file)
    assert len(config.prompts.prompts) == 2
    assert config.prompts.test_prompt == "This is a test prompt"
    assert config.prompts.another_prompt == "Another test prompt"


def test_prompt_attribute_access():
    """Test accessing prompts via attributes."""
    prompts = PromptConfig(prompts={"test": "Test prompt"})
    assert prompts.test == "Test prompt"
    with pytest.raises(AttributeError):
        _ = prompts.nonexistent


def test_prompt_validation():
    """Test prompt validation."""
    # Valid prompts (all strings)
    PromptConfig(prompts={"test": "Test prompt", "another": "Another prompt"})

    # Invalid prompts (non-string value)
    with pytest.raises(ConfigurationError):
        PromptConfig.from_file(Path(__file__).parent / "data" / "invalid_prompts.yaml")


def test_environment_variable_override(temp_env_prompts_file, monkeypatch):
    """Test loading prompts from environment variable."""
    monkeypatch.setenv("CODEGATE_PROMPTS_FILE", str(temp_env_prompts_file))
    config = Config.load()
    assert len(config.prompts.prompts) == 2
    assert config.prompts.env_prompt == "This is an environment prompt"
    assert config.prompts.another_env == "Another environment prompt"


def test_cli_override_takes_precedence(temp_prompts_file, temp_env_prompts_file, monkeypatch):
    """Test that CLI prompts override config and environment."""
    # Set environment variable
    monkeypatch.setenv("CODEGATE_PROMPTS_FILE", str(temp_env_prompts_file))

    # Load with CLI override
    config = Config.load(prompts_path=temp_prompts_file)

    # Should use prompts from CLI-specified file
    assert len(config.prompts.prompts) == 2
    assert config.prompts.test_prompt == "This is a test prompt"
    assert config.prompts.another_prompt == "Another test prompt"


def test_invalid_yaml_file():
    """Test handling of invalid YAML file."""
    with pytest.raises(ConfigurationError):
        PromptConfig.from_file(Path(__file__).parent / "nonexistent.yaml")


def test_empty_prompts_file(tmp_path):
    """Test handling of empty prompts file."""
    empty_file = tmp_path / "empty.yaml"
    empty_file.write_text("")

    with pytest.raises(ConfigurationError):
        PromptConfig.from_file(empty_file)


def test_non_dict_prompts_file(tmp_path):
    """Test handling of non-dictionary prompts file."""
    invalid_file = tmp_path / "invalid.yaml"
    invalid_file.write_text("- not a dictionary")

    with pytest.raises(ConfigurationError):
        PromptConfig.from_file(invalid_file)


def test_missing_default_prompts(monkeypatch):
    """Test graceful handling of missing default prompts file."""

    # Temporarily modify the path to point to a nonexistent location
    def mock_load_default_prompts():
        return PromptConfig()

    monkeypatch.setattr(Config, "_load_default_prompts", mock_load_default_prompts)

    config = Config.load()
    assert isinstance(config.prompts, PromptConfig)
    assert len(config.prompts.prompts) == 0
