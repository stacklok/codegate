import os
from typing import (
    Any,
    AsyncIterator,
)

import pydantic
import structlog

logger = structlog.get_logger("codegate")


# Since different providers typically use one of these formats for streaming
# responses, we have a single stream generator for each format that is then plugged
# into the adapter.


async def sse_stream_generator(stream: AsyncIterator[Any]) -> AsyncIterator[str]:
    """OpenAI-style SSE format"""
    try:
        async for chunk in stream:
            if isinstance(chunk, pydantic.BaseModel):
                # alternatively we might want to just dump the whole object
                # this might even allow us to tighten the typing of the stream
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
