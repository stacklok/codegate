import os
from typing import (
    AsyncIterator,
)

import httpx
import structlog

from ._response_models import (
    StreamingChatCompletion,
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
                if os.getenv("CODEGATE_DEBUG_OPENAI") is not None:
                    print(chunk)
                yield f"data: {chunk}\n\n"
            except Exception as e:
                logger.error("failed generating output payloads", exc_info=e)
                yield f"data: {str(e)}\n\n"
    except Exception as e:
        logger.error("failed generating output payloads", exc_info=e)
        yield f"data: {str(e)}\n\n"
    finally:
        yield "data: [DONE]\n\n"


async def completions_streaming(request, api_key, base_url):
    if base_url is None:
        base_url = "https://api.openai.com"
    return streaming(request, api_key, f"{base_url}/v1/chat/completions")


async def streaming(request, api_key, url):
    payload = request.json(exclude_defaults=True)
    if os.getenv("CODEGATE_DEBUG_OPENAI") is not None:
        print(payload)

    client = httpx.AsyncClient()
    async with client.stream(
            "POST", url,
            content=payload,
            timeout=30, # TODO this should not be hardcoded
    ) as resp:
        # TODO figure out how to best return failures
        match resp.status_code:
            case 200:
                async for message in parser(resp.aiter_lines()):
                    yield message
            case 400 | 401 | 403 | 404 | 413 | 429:
                logger.error(f"unexpected status code {resp.status_code}: {resp.text}", provider="openai")
                yield MessageError.model_validate_json(resp.text)
            # case 500 | 529:
            #     yield MessageError.model_validate_json(resp.text)
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

        count = count + 1

        if "[DONE]" in data_line:
            break

        yield data_line[6:]
    logger.debug(f"Consumed {count} messages", provider="anthropic", count=count)


async def parser(lines):
    messages = get_data_lines(lines)
    async for payload in messages:
        item = StreamingChatCompletion.model_validate_json(payload)
        yield item
