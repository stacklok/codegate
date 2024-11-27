"""Tests for the CLI prompts functionality."""

import pytest
from click.testing import CliRunner

from codegate.cli import cli


@pytest.fixture
def temp_prompts_file(tmp_path):
    """Create a temporary prompts file for testing."""
    prompts_content = """
test_prompt: "This is a test prompt"
another_prompt: "Another test prompt"
"""
    prompts_file = tmp_path / "test_prompts.yaml"
    prompts_file.write_text(prompts_content)
    return prompts_file


def test_show_prompts_command(temp_prompts_file):
    """Test the show-prompts command with custom prompts file."""
    runner = CliRunner()
    result = runner.invoke(cli, ["show-prompts", "--prompts", str(temp_prompts_file)])

    assert result.exit_code == 0
    assert "Loaded prompts:" in result.output
    assert "test_prompt:" in result.output
    assert "This is a test prompt" in result.output
    assert "another_prompt:" in result.output
    assert "Another test prompt" in result.output


def test_show_default_prompts():
    """Test the show-prompts command without --prompts flag shows default prompts."""
    runner = CliRunner()
    result = runner.invoke(cli, ["show-prompts"])

    assert result.exit_code == 0
    assert "Loaded prompts:" in result.output
    assert "default_chat:" in result.output
    assert "default_snippet:" in result.output
    assert "codegate_chat:" in result.output
    assert "codegate_snippet:" in result.output
    assert "security_audit:" in result.output
    assert "red_team:" in result.output
    assert "blue_team:" in result.output


def test_show_prompts_nonexistent_file():
    """Test show-prompts with nonexistent file."""
    runner = CliRunner()
    result = runner.invoke(cli, ["show-prompts", "--prompts", "nonexistent.yaml"])

    assert result.exit_code == 2  # Click's error exit code
    assert "does not exist" in result.output


def test_show_prompts_invalid_yaml(tmp_path):
    """Test show-prompts with invalid YAML file."""
    invalid_file = tmp_path / "invalid.yaml"
    invalid_file.write_text("invalid: yaml: content")

    runner = CliRunner()
    result = runner.invoke(cli, ["show-prompts", "--prompts", str(invalid_file)])

    assert result.exit_code == 1
    assert "error" in result.output.lower()


def test_serve_with_prompts(temp_prompts_file):
    """Test the serve command with prompts file."""
    runner = CliRunner()
    # Use --help to avoid actually starting the server
    result = runner.invoke(cli, ["serve", "--prompts", str(temp_prompts_file), "--help"])

    assert result.exit_code == 0
    assert "Path to YAML prompts file" in result.output
