import json
from typing import Any, AsyncIterator

from pydantic import BaseModel

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
                yield f"data:{str(e)}\n\n"
    except Exception as e:
        yield f"data: {str(e)}\n\n"
    finally:
        yield "data: [DONE]\n\n"


async def anthropic_stream_generator(stream: AsyncIterator[Any]) -> AsyncIterator[str]:
    """Anthropic-style SSE format"""
    try:
        async for chunk in stream:
            event_type = chunk.get("type")
            try:
                yield f"event: {event_type}\ndata:{json.dumps(chunk)}\n\n"
            except Exception as e:
                yield f"event: {event_type}\ndata:{str(e)}\n\n"
    except Exception as e:
        yield f"data: {str(e)}\n\n"
