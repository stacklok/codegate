import os

import httpx
import structlog

from ._response_models import (
    MessageError,
    StreamingChatCompletion,
    StreamingGenerateCompletion,
)


logger = structlog.get_logger("codegate")


async def chat_streaming(request, api_key, base_url):
    if base_url is None:
        base_url = "http://localhost:11434"
    return streaming(request, api_key, f"{base_url}/api/chat", StreamingChatCompletion)


async def generate_streaming(request, api_key, base_url):
    if base_url is None:
        base_url = "http://localhost:11434"
    return streaming(request, api_key, f"{base_url}/api/generate", StreamingGenerateCompletion)


async def streaming(request, api_key, url, cls):
    payload = request.json(exclude_defaults=True)
    if os.getenv("CODEGATE_DEBUG_OLLAMA") is not None:
        print(payload)

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            url,
            content=payload,
            timeout=300, # TODO this should not be hardcoded
        )

        # TODO figure out how to best return failures
        match resp.status_code:
            case 200:
                # TODO this loop causes the connection to be kept open
                # causing read timeouts and blocks (breaks the
                # typewriter effect), fix this
                async for message in parser(cls, resp.aiter_lines()):
                    yield message
            case 400 | 401 | 403 | 404 | 413 | 429:
                yield MessageError.model_validate_json(resp.text)
            # case 500 | 529:
            #     yield MessageError.model_validate_json(resp.text)
            case _:
                logger.error(f"unexpected status code {resp.status_code}", provider="ollama")
                raise ValueError(f"unexpected status code {resp.status_code}", provider="ollama")


async def get_data_lines(lines):
    count = 0
    while True:
        # Every line has a single JSON payload
        line = await anext(lines)
        count = count + 1
        yield line
    logger.debug(f"Consumed {count} messages", provider="anthropic", count=count)


async def parser(cls, lines):
    messages = get_data_lines(lines)
    async for payload in messages:
        item = cls.model_validate_json(payload)
        yield item
        if item.done:
            break
