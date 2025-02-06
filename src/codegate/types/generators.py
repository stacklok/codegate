import json
import os
from typing import (
    Any,
    AsyncIterator,
)

import httpx
from litellm import (
    acompletion as legacy_acompletion,
    atext_completion as legacy_atext_completion,
)
from litellm import (
    ModelResponseStream,
)
from pydantic import BaseModel
import structlog

from codegate.types.anthropic import (
    # request objects
    ChatCompletionRequest,
    # response objects
    ApiError,
    AuthenticationError,
    ContentBlockDelta,
    ContentBlockStart,
    ContentBlockStop,
    InvalidRequestError,
    MessageDelta,
    MessageError,
    MessagePing,
    MessageStart,
    MessageStop,
    NotFoundError,
    OverloadedError,
    PermissionError,
    RateLimitError,
    RequestTooLargeError,
)
from codegate.types.common import (
    CodegateFunction,
    CodegateChatCompletionDeltaToolCall,
    CodegateDelta,
    CodegateStreamingChoices,
    CodegateModelResponseStream,
)


logger = structlog.get_logger("codegate")


# Since different providers typically use one of these formats for streaming
# responses, we have a single stream generator for each format that is then plugged
# into the adapter.


async def sse_stream_generator(stream: AsyncIterator[Any]) -> AsyncIterator[str]:
    """OpenAI-style SSE format"""
    try:
        async for chunk in stream:
            if isinstance(chunk, BaseModel):
                # alternatively we might want to just dump the whole object
                # this might even allow us to tighten the typing of the stream
                chunk = chunk.model_dump_json(exclude_none=True, exclude_unset=True)
            try:
                yield f"data:{chunk}\n\n"
            except Exception as e:
                logger.error("failed generating output payloads", exc_info=e)
                yield f"data:{str(e)}\n\n"
    except Exception as e:
        logger.error("failed generating output payloads", exc_info=e)
        yield f"data: {str(e)}\n\n"
    finally:
        yield "data: [DONE]\n\n"


async def anthropic_stream_generator(stream: AsyncIterator[Any]) -> AsyncIterator[str]:
    """Anthropic-style SSE format"""
    try:
        async for chunk in stream:
            if isinstance(chunk, CodegateModelResponseStream) and chunk.payload:
                if chunk.payload:
                    event_type = chunk.payload["type"]
                    body = json.dumps(chunk.payload)
            else:
                event_type = chunk.get("type", "content_block_delta")
                body = json.dumps(chunk)

            if os.getenv("CODEGATE_DEBUG_ANTHROPIC") is not None:
                print(f"CODEGATE_DEBUG_ANTHROPIC: anthropic_stream_generator: {body}")

            try:
                yield f"event: {event_type}\ndata: {body}\n\n"
            except Exception as e:
                logger.error("failed generating output payloads", exc_info=e)
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

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            content=payload,
        )

        if os.getenv("CODEGATE_DEBUG_ANTHROPIC") is not None:
            print(f"CODEGATE_DEBUG_ANTHROPIC: acompletion: {payload}")
            print(f"CODEGATE_DEBUG_ANTHROPIC: acompletion: {resp.text}")

        # TODO figure out how to best return failures
        match resp.status_code:
            case 400: # invalid_request_error
                yield InvalidRequestError.parse_raw(resp.text)
            case 401: # authentication_error
                yield AuthenticationError.parse_raw(resp.text)
            case 403: # permission_error
                yield PermissionError.parse_raw(resp.text)
            case 404: # not_found_error
                yield NotFoundError.parse_raw(resp.text)
            case 413: # request_too_large
                yield RequestTooLargeError.parse_raw(resp.text)
            case 429: # rate_limit_error
                yield RateLimitError.parse_raw(resp.text)
            case 500: # api_error
                yield ApiError.parse_raw(resp.text)
            case 529: # overloaded_error
                yield OverloadedError.parse_raw(resp.text)
            case 200: # happy path
                async for event in anthropic_message_wrapper(resp.aiter_lines()):
                    yield event
            case _:
                raise ValueError(f"anthropic: unexpected status code {resp.status_code}")


async def get_data_lines(lines):
    while True:
        # Get the `event: <type>` line.
        event_line = await anext(lines)
        # Get the `data: <json>` line.
        data_line = await anext(lines)
        # Get the empty line.
        _ = await anext(lines)

        # Event lines always begin with `event: `, and Data lines
        # always begin with `data: `, so we can skip the first few
        # characters and just return the payload.
        yield event_line[7:], data_line[6:]


async def anthropic_message_wrapper(lines):
    events = get_data_lines(lines)
    event_type, payload = await anext(events)

    # We expect the first line to always be `event: message_start`.
    if event_type != "message_start":
        raise ValueError(f"anthropic: unexpected event type '{event_type}'")

    yield MessageStart.parse_raw(payload)

    async for event_type, payload in events:
        if os.getenv("CODEGATE_DEBUG_ANTHROPIC") is not None:
            print(f"{anthropic_message_wrapper.__name__}: {payload}")

        match event_type:
            case "message_delta":
                yield MessageDelta.parse_raw(payload)
            case "content_block_start":
                yield ContentBlockStart.parse_raw(payload)
            case "content_block_delta":
                yield ContentBlockDelta.parse_raw(payload)
            case "content_block_stop":
                yield ContentBlockStop.parse_raw(payload)
            case "message_stop":
                yield MessageStop.parse_raw(payload)
                # We break the loop at this poiunt since this is the
                # final payload defined by the protocol.
                break
            case "ping":
                yield MessagePing.parse_raw(payload)
            case "error":
                yield MessageError.parse_raw(payload)
                break
            case _:
                # TODO this should be a log entry, as per
                # https://docs.anthropic.com/en/api/messages-streaming#other-events
                raise ValueError(f"anthropic: unexpected event type '{event_type}'")

    # The following should always hold when we get here
    assert event_type == "message_stop" or event_type == "error"
    return
