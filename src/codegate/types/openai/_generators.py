import os
from typing import (
    AsyncIterator,
)

import httpx
import structlog

from ._legacy_models import (
    LegacyCompletionRequest,
)
from ._response_models import (
    ChatCompletion,
    ErrorDetails,
    MessageError,
    StreamingChatCompletion,
    VllmMessageError,
)

logger = structlog.get_logger("codegate")


async def stream_generator(stream: AsyncIterator[StreamingChatCompletion]) -> AsyncIterator[str]:
    """OpenAI-style SSE format"""
    try:
        async for chunk in stream:
            # alternatively we might want to just dump the whole
            # object this might even allow us to tighten the typing of
            # the stream
            chunk = chunk.model_dump_json(exclude_none=True, exclude_unset=True)
            try:
                yield f"data: {chunk}\n\n"
            except Exception as e:
                logger.error("failed generating output payloads", exc_info=e)
                yield f"data: {str(e)}\n\n"
    except Exception as e:
        logger.error("failed generating output payloads", exc_info=e)
        err = MessageError(
            error=ErrorDetails(
                message=str(e),
                code=500,
            ),
        )
        data = err.model_dump_json(exclude_none=True, exclude_unset=True)
        yield f"data: {data}\n\n"
    finally:
        # during SSE processing.
        yield "data: [DONE]\n\n"


async def single_response_generator(
    first: ChatCompletion,
    stream: AsyncIterator[ChatCompletion],
) -> AsyncIterator[ChatCompletion]:
    """Wraps a single response object in an AsyncIterator. This is
    meant to be used for non-streaming responses.

    """
    yield first.model_dump_json(exclude_none=True, exclude_unset=True)

    # Note: this async for loop is necessary to force Python to return
    # an AsyncIterator. This is necessary because of the wiring at the
    # Provider level expecting an AsyncIterator rather than a single
    # response payload.
    #
    # Refactoring this means adding a code path specific for when we
    # expect single response payloads rather than an SSE stream.
    async for item in stream:
        if item:
            logger.error("no further items were expected", item=item)
        yield item.model_dump_json(exclude_none=True, exclude_unset=True)


async def completions_streaming(request, api_key, base_url):
    if base_url is None:
        base_url = "https://api.openai.com"
    # TODO refactor this. This is a ugly hack, we have to fix the way
    # we calculate base urls.
    if "/v1" not in base_url:
        base_url = f"{base_url}/v1"

    # TODO refactor. This is yet another Ugly hack caused by having a
    # single code path for both legacy and current APIs.
    url = f"{base_url}/chat/completions"
    if isinstance(request, LegacyCompletionRequest):
        url = f"{base_url}/completions"

    async for item in streaming(request, api_key, url):
        yield item


async def streaming(request, api_key, url, cls=StreamingChatCompletion):
    headers = {
        "Content-Type": "application/json",
    }

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = request.json(exclude_defaults=True)
    if os.getenv("CODEGATE_DEBUG_OPENAI") is not None:
        print(payload)

    client = httpx.AsyncClient()
    async with client.stream(
        "POST",
        url,
        headers=headers,
        content=payload,
        timeout=30,  # TODO this should not be hardcoded
    ) as resp:
        # TODO figure out how to best return failures
        match resp.status_code:
            case 200:
                if not request.stream:
                    body = await resp.aread()
                    if os.getenv("CODEGATE_DEBUG_OPENAI") is not None:
                        print(body.decode("utf-8"))
                    yield ChatCompletion.model_validate_json(body)
                    return

                async for message in message_wrapper(resp.aiter_lines(), cls):
                    yield message
            case 400 | 401 | 403 | 404 | 413 | 429:
                text = await resp.aread()
                # Ugly hack because VLLM is not 100% compatible with
                # OpenAI message structure.
                try:
                    item = MessageError.model_validate_json(text)
                    yield item
                except Exception:
                    try:
                        item = VllmMessageError.model_validate_json(text)
                        yield item
                    except Exception as e:
                        raise e
            case 500 | 529:
                text = await resp.aread()
                yield MessageError.model_validate_json(text)
            case _:
                logger.error(f"unexpected status code {resp.status_code}", provider="openai")
                raise ValueError(f"unexpected status code {resp.status_code}", provider="openai")


async def get_data_lines(lines):
    count = 0
    while True:
        # Get the `data: <type>` line.
        data_line = await anext(lines)
        # Get the empty line.
        _ = await anext(lines)

        # As per standard, we ignore comment lines
        # https://html.spec.whatwg.org/multipage/server-sent-events.html#event-stream-interpretation
        if data_line.startswith(":"):
            continue

        count = count + 1

        if "[DONE]" in data_line:
            break

        yield data_line[6:]
    logger.debug(f"Consumed {count} messages", provider="openai", count=count)


async def message_wrapper(lines, cls=StreamingChatCompletion):
    messages = get_data_lines(lines)
    async for payload in messages:
        try:
            if os.getenv("CODEGATE_DEBUG_OPENAI") is not None:
                print(payload)
            item = cls.model_validate_json(payload)
            yield item
        except Exception as e:
            logger.warn("HTTP error while consuming SSE stream", payload=payload, exc_info=e)
            err = MessageError(
                error=ErrorDetails(
                    message=str(e),
                    code=500,
                ),
            )
            yield err
