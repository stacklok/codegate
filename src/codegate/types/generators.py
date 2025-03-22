from typing import (
    Callable,
)

import structlog

logger = structlog.get_logger("codegate")


def completion_handler_replacement(
    completion_handler: Callable,
):
    async def _inner(
        request,
        base_url,
        api_key,
        stream=None,
        is_fim_request=None,
    ):
        # Execute e.g. acompletion from Anthropic types
        return completion_handler(
            request,
            api_key,
            base_url,
        )

    return _inner
