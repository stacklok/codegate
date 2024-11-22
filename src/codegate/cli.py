"""Command-line interface for codegate."""

import logging
import sys
from pathlib import Path
from typing import Optional

import click

from .config import Config, ConfigurationError, LogFormat, LogLevel
from .logging import setup_logging
from .server import init_app


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
def serve(
    port: Optional[int],
    host: Optional[str],
    log_level: Optional[str],
    log_format: Optional[str],
    config: Optional[Path],
) -> None:
    """Start the codegate server."""
    try:
        # Load configuration with priority resolution
        cfg = Config.load(
            config_path=config,
            cli_port=port,
            cli_host=host,
            cli_log_level=log_level,
            cli_log_format=log_format,
        )

        setup_logging(cfg.log_level, cfg.log_format)
        logger = logging.getLogger(__name__)

        logger.info(
            "Starting server",
            extra={
                "host": cfg.host,
                "port": cfg.port,
                "log_level": cfg.log_level.value,
                "log_format": cfg.log_format.value,
            },
        )

        app = init_app()

        import uvicorn

        uvicorn.run(
            app,
            host=cfg.host,
            port=cfg.port,
            log_level=cfg.log_level.value.lower(),
            log_config=None,  # Default logging configuration
        )

    except KeyboardInterrupt:
        logger.info("Shutting down server")
    except ConfigurationError as e:
        click.echo(f"Configuration error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error occurred")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def main() -> None:
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
