"""Command-line interface for codegate."""

import sys
from pathlib import Path
from typing import Dict, Optional

import click
import structlog

from codegate.codegate_logging import LogFormat, LogLevel, setup_logging
from codegate.config import Config, ConfigurationError
from codegate.db.connection import init_db_sync
from codegate.pipeline.factory import PipelineFactory
from codegate.pipeline.secrets.manager import SecretsManager
from codegate.server import init_app
from codegate.storage.utils import restore_storage_backup


def validate_port(ctx: click.Context, param: click.Parameter, value: int) -> int:
    """Validate the port number is in valid range."""
    if value is not None and not (1 <= value <= 65535):
        raise click.BadParameter("Port must be between 1 and 65535")
    return value


@click.group()
@click.version_option()
def cli() -> None:
    """Codegate - A configurable service gateway."""
    pass


@cli.command()
@click.option(
    "--prompts",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=False,
    help="Path to YAML prompts file (optional, shows default prompts if not provided)",
)
def show_prompts(prompts: Optional[Path]) -> None:
    """Display prompts from the specified file or default if no file specified."""
    try:
        cfg = Config.load(prompts_path=prompts)
        click.echo("Loaded prompts:")
        click.echo("-" * 40)
        for name, content in cfg.prompts.prompts.items():
            click.echo(f"\n{name}:")
            click.echo(f"{content}")
            click.echo("-" * 40)
    except ConfigurationError as e:
        click.echo(f"Configuration error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--port",
    type=int,
    default=None,
    callback=validate_port,
    help="Port to listen on (default: 8989)",
)
@click.option(
    "--host",
    type=str,
    default=None,
    help="Host to bind to (default: localhost)",
)
@click.option(
    "--log-level",
    type=click.Choice([level.value for level in LogLevel]),
    default=None,
    help="Set the log level (default: INFO)",
)
@click.option(
    "--log-format",
    type=click.Choice([fmt.value for fmt in LogFormat], case_sensitive=False),
    default=None,
    help="Set the log format (default: JSON)",
)
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to YAML config file",
)
@click.option(
    "--prompts",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to YAML prompts file",
)
@click.option(
    "--vllm-url",
    type=str,
    default=None,
    help="vLLM provider URL (default: http://localhost:8000/v1)",
)
@click.option(
    "--openai-url",
    type=str,
    default=None,
    help="OpenAI provider URL (default: https://api.openai.com/v1)",
)
@click.option(
    "--anthropic-url",
    type=str,
    default=None,
    help="Anthropic provider URL (default: https://api.anthropic.com/v1)",
)
@click.option(
    "--ollama-url",
    type=str,
    default=None,
    help="Ollama provider URL (default: http://localhost:11434/api)",
)
@click.option(
    "--model-base-path",
    type=str,
    default="./models",
    help="Path to the model base directory",
)
@click.option(
    "--embedding-model",
    type=str,
    default="all-minilm-L6-v2-q5_k_m.gguf",
    help="Name of the model to use for embeddings",
)
def serve(
    port: Optional[int],
    host: Optional[str],
    log_level: Optional[str],
    log_format: Optional[str],
    config: Optional[Path],
    prompts: Optional[Path],
    vllm_url: Optional[str],
    openai_url: Optional[str],
    anthropic_url: Optional[str],
    ollama_url: Optional[str],
    model_base_path: Optional[str],
    embedding_model: Optional[str],
) -> None:
    """Start the codegate server."""
    logger = None
    try:
        # Create provider URLs dict from CLI options
        cli_provider_urls: Dict[str, str] = {}
        if vllm_url:
            cli_provider_urls["vllm"] = vllm_url
        if openai_url:
            cli_provider_urls["openai"] = openai_url
        if anthropic_url:
            cli_provider_urls["anthropic"] = anthropic_url
        if ollama_url:
            cli_provider_urls["ollama"] = ollama_url

        # Load configuration with priority resolution
        cfg = Config.load(
            config_path=config,
            prompts_path=prompts,
            cli_port=port,
            cli_host=host,
            cli_log_level=log_level,
            cli_log_format=log_format,
            cli_provider_urls=cli_provider_urls,
            model_base_path=model_base_path,
            embedding_model=embedding_model,
        )

        setup_logging(cfg.log_level, cfg.log_format)
        logger = structlog.get_logger("codegate")
        logger.info(
            "Starting server",
            extra={
                "host": cfg.host,
                "port": cfg.port,
                "log_level": cfg.log_level.value,
                "log_format": cfg.log_format.value,
                "prompts_loaded": len(cfg.prompts.prompts),
                "provider_urls": cfg.provider_urls,
                "model_base_path": cfg.model_base_path,
                "embedding_model": cfg.embedding_model,
            },
        )

        init_db_sync()
        # Initialize secrets manager and pipeline factory
        secrets_manager = SecretsManager()
        pipeline_factory = PipelineFactory(secrets_manager)

        app = init_app(pipeline_factory)

        import uvicorn

        uvicorn.run(
            app,
            host=cfg.host,
            port=cfg.port,
            log_level=cfg.log_level.value.lower(),
            log_config=None,  # Default logging configuration
        )

    except KeyboardInterrupt:
        if logger:
            logger.info("Shutting down server")
    except ConfigurationError as e:
        click.echo(f"Configuration error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        if logger:
            logger.exception("Unexpected error occurred")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--backup-path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory path where the backup file is located.",
)
@click.option(
    "--backup-name",
    type=str,
    required=True,
    help="Name of the backup file to restore.",
)
def restore_backup(backup_path: Path, backup_name: str) -> None:
    """Restore the database from the specified backup."""
    try:
        restore_storage_backup(backup_path, backup_name)
        click.echo(f"Successfully restored the backup '{backup_name}' from {backup_path}.")
    except Exception as e:
        click.echo(f"Error restoring backup: {e}", err=True)
        sys.exit(1)


def main() -> None:
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
