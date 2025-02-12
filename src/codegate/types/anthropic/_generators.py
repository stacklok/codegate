import os
from typing import (
    Any,
    AsyncIterator,
)

import httpx
import structlog
from pydantic import BaseModel

from ._request_models import (
    ChatCompletionRequest,
)

from ._response_models import (
    ApiError,
    ContentBlockDelta,
    ContentBlockStart,
    ContentBlockStop,
    MessageDelta,
    MessageError,
    MessagePing,
    MessageStart,
    MessageStop,
)


logger = structlog.get_logger("codegate")


async def stream_generator(stream: AsyncIterator[Any]) -> AsyncIterator[str]:
    """Anthropic-style SSE format"""
    try:
        async for chunk in stream:
            try:
                body = chunk.json(exclude_defaults=True, exclude_unset=True)
                if os.getenv("CODEGATE_DEBUG_ANTHROPIC") is not None:
                    print(body)
                yield f"event: {chunk.type}\ndata: {body}\n\n"
            except Exception as e:
                logger.error("failed serializing payload", exc_info=e)
                yield f"event: {event_type}\ndata: {str(e)}\n\n"
    except Exception as e:
        logger.error("failed generating output payloads", exc_info=e)
        yield f"data: {str(e)}\n\n"


async def acompletion(request, api_key):
    return _inner(request, api_key)


# This function is here only to prevent more changes in the callers,
# but it's totally redundant. We should try to get rid of this wrapper
# eventually.
async def _inner(request, api_key):
    headers = {
        "anthropic-version": "2023-06-01",
        "x-api-key": api_key,
        "accept": "application/json",
        "content-type": "application/json",
    }
    payload = request.json(exclude_defaults=True)

    if os.getenv("CODEGATE_DEBUG_ANTHROPIC") is not None:
        print(payload)

    client = httpx.AsyncClient()
    async with client.stream(
            "POST", "https://api.anthropic.com/v1/messages",
            headers=headers,
            content=payload,
            timeout=30, # TODO this should not be hardcoded
    ) as resp:
        # TODO figure out how to best return failures
        match resp.status_code:
            case 200:
                async for event in anthropic_message_wrapper(resp.aiter_lines()):
                    yield event
            case 400 | 401 | 403 | 404 | 413 | 429:
                yield MessageError.model_validate_json(resp.text)
            case 500 | 529:
                yield MessageError.model_validate_json(resp.text)
            case _:
                logger.error(f"unexpected status code {resp.status_code}", provider="anthropic")
                raise ValueError(f"unexpected status code {resp.status_code}", provider="anthropic")


async def get_data_lines(lines):
    count = 0
    while True:
        # Get the `event: <type>` line.
        event_line = await anext(lines)
        # Get the `data: <json>` line.
        data_line = await anext(lines)
        # Get the empty line.
        _ = await anext(lines)

        count = count + 1

        # Event lines always begin with `event: `, and Data lines
        # always begin with `data: `, so we can skip the first few
        # characters and just return the payload.
        yield event_line[7:], data_line[6:]
    logger.debug(f"Consumed {count} messages", provider="anthropic", count=count)


async def anthropic_message_wrapper(lines):
    events = get_data_lines(lines)
    event_type, payload = await anext(events)

    # We expect the first line to always be `event: message_start`.
    if event_type != "message_start" and event_type != "error":
        raise ValueError(f"anthropic: unexpected event type '{event_type}'")

    match event_type:
        case "error":
            yield MessageError.model_validate_json(payload)
            return
        case "message_start":
            yield MessageStart.model_validate_json(payload)

    async for event_type, payload in events:
        match event_type:
            case "message_delta":
                yield MessageDelta.model_validate_json(payload)
            case "content_block_start":
                yield ContentBlockStart.model_validate_json(payload)
            case "content_block_delta":
                yield ContentBlockDelta.model_validate_json(payload)
            case "content_block_stop":
                yield ContentBlockStop.model_validate_json(payload)
            case "message_stop":
                yield MessageStop.model_validate_json(payload)
                # We break the loop at this poiunt since this is the
                # final payload defined by the protocol.
                break
            case "ping":
                yield MessagePing.model_validate_json(payload)
            case "error":
                yield MessageError.model_validate_json(payload)
                break
            case _:
                # TODO this should be a log entry, as per
                # https://docs.anthropic.com/en/api/messages-streaming#other-events
                raise ValueError(f"anthropic: unexpected event type '{event_type}'")

    # The following should always hold when we get here
    assert event_type == "message_stop" or event_type == "error"
    return
