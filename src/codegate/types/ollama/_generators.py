import json
import os
from typing import (
    AsyncIterator,
)

import httpx
import structlog

from ._response_models import (
    MessageError,
    StreamingChatCompletion,
    StreamingGenerateCompletion,
)


logger = structlog.get_logger("codegate")


async def stream_generator(
    stream: AsyncIterator[StreamingChatCompletion | StreamingGenerateCompletion],
) -> AsyncIterator[str]:
    """Ollama-style SSE format"""
    try:
        async for chunk in stream:
            try:
                body = chunk.model_dump_json(exclude_none=True, exclude_unset=True)
                data = f"{body}\n"

                if os.getenv("CODEGATE_DEBUG_OLLAMA") is not None:
                    print("---> OLLAMA DEBUG")
                    print(data)

                yield data
            except Exception as e:
                logger.error("failed serializing payload", exc_info=e, provider="ollama")
                yield f"{json.dumps({'error': str(e)})}\n"
    except Exception as e:
        logger.error("failed generating output payloads", exc_info=e, provider="ollama")
        yield f"{json.dumps({'error': str(e)})}\n"


async def chat_streaming(request, api_key, base_url):
    if base_url is None:
        base_url = "http://localhost:11434"
    async for item in streaming(request, api_key, f"{base_url}/api/chat", StreamingChatCompletion):
        yield item


async def generate_streaming(request, api_key, base_url):
    if base_url is None:
        base_url = "http://localhost:11434"
    async for item in streaming(
        request, api_key, f"{base_url}/api/generate", StreamingGenerateCompletion
    ):
        yield item


async def streaming(request, api_key, url, cls):
    headers = dict()

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = request.json(exclude_defaults=True)
    if os.getenv("CODEGATE_DEBUG_OLLAMA") is not None:
        print(payload)

    client = httpx.AsyncClient()
    async with client.stream(
        "POST",
        url,
        headers=headers,
        content=payload,
        timeout=300,  # TODO this should not be hardcoded
    ) as resp:
        # TODO figure out how to best return failures
        match resp.status_code:
            case 200:
                async for message in message_wrapper(cls, resp.aiter_lines()):
                    yield message
            case 400 | 401 | 403 | 404 | 413 | 429:
                body = await resp.aread()
                yield MessageError.model_validate_json(body)
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


# todo: this should have the same signature as message_wrapper in openai
async def message_wrapper(cls, lines):
    messages = get_data_lines(lines)
    async for payload in messages:
        try:
            item = cls.model_validate_json(payload)
            yield item
            if item.done:
                break
        except Exception as e:
            logger.warn("HTTP error while consuming SSE stream", payload=payload, exc_info=e)
            err = MessageError(
                error=ErrorDetails(
                    message=str(e),
                    code=500,
                ),
            )
            item = MessageError.model_validate_json(payload)
            yield item
