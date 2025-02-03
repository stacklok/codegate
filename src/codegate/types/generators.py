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
                yield f"data: {chunk}\n\n"
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
            if isinstance(chunk, CodegateModelResponseStream):
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


# async def spacer(stream):
#     async for item in stream:
#         print("\n\n\n\n")
#         yield item


# async def printer(stream):
#     async for item in stream:
#         print(f"{item}")
#         yield item


async def acompletion(
        **request,
): # -> Union[ModelResponse, CustomStreamWrapper]:
    headers = {
        "anthropic-version": "2023-06-01",
        "x-api-key": request.get("api_key"),
        "accept": "application/json",
        "content-type": "application/json",
    }

    copy = dict(request)
    del copy["api_key"]
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=copy,
        )

        if os.getenv("CODEGATE_DEBUG_ANTHROPIC") is not None:
            print(f"CODEGATE_DEBUG_ANTHROPIC: acompletion: {json.dumps(copy)}")
            print(f"CODEGATE_DEBUG_ANTHROPIC: acompletion: {resp.text}")

        # return spacer(printer(anthropic_chunk_wrapper(printer(resp.aiter_lines()))))
        return anthropic_chunk_wrapper(resp.aiter_lines())


async def get_data_lines(lines):
    while True:
        # Get the `event: <type>` line.
        _ = await anext(lines)
        # Get the `data: <json>` line.
        data_line = await anext(lines)
        # Get the empty line.
        _ = await anext(lines)

        # Data lines always begin with `data: `, so we can skip the first
        # few characters and just deserialized the payload.
        yield json.loads(data_line[5:])


async def anthropic_chunk_wrapper(lines):
    payloads = get_data_lines(lines)
    payload = await anext(payloads)

    # We expect the first line to always be `event: message_start`.
    if payload["type"] != "message_start":
        raise ValueError(f"anthropic: unexpected event type '{payload['type']}'")

    model = payload["message"]["model"]
    yield make_empty_response(payload, model)

    async for payload in payloads:
        if os.getenv("CODEGATE_DEBUG_ANTHROPIC") is not None:
            print(f"CODEGATE_DEBUG_ANTHROPIC: anthropic_chunk_wrapper: {payload}")

        if payload["type"] == "ping":
            pass # ignored
        elif payload["type"] == "content_block_start":
            content_block_type = payload["content_block"]["type"]

            if content_block_type == "text":
                item = make_empty_response(payload, model)
                yield item
                # Procesing `text` block type is done by purely
                # yielding block deltas as they are received.
                async for item in anthropic_content_chunk_wrapper(payloads, model):
                    yield item

            elif content_block_type == "tool_use":
                content_block = payload["content_block"]
                assert content_block["type"] == "tool_use"

                # Processing `tool_use` block type is done by emitting
                # a single message for the start and one message for
                # each additional block delta entry.
                function = CodegateFunction(
                    name=content_block["name"],
                    arguments=None,
                )
                tool_call = CodegateChatCompletionDeltaToolCall(
                    id=content_block["id"],
                    function=function,
                    type="function",
                    index=payload["index"],
                )
                delta = CodegateDelta(
                    role="assistant",
                    content=None,
                    tool_calls=[tool_call],
                )
                choice = CodegateStreamingChoices(
                    index=payload["index"],
                    delta=delta,
                )

                item = make_response(payload, model, choice)
                yield item

                async for item in anthropic_tool_chunk_wrapper(payloads, model):
                    yield item
            else:
                raise ValueError(
                    f"anthropic: unexpected block type '{payload['content_block']['type']}'",
                )
        elif payload["type"] == "message_delta":
            pdelta = payload["delta"]
            if pdelta["stop_reason"] == "tool_use":
                choice = CodegateStreamingChoices(
                    delta=CodegateDelta(
                        role="assistant",
                    ),
                    # Confusingly enough, Anthropic's finish
                    # reason `tool_use` maps to OpenAI's
                    # `tool_calls`.
                    finish_reason="tool_calls",
                )

                item = make_response(payload, model, choice)
                yield item
        elif payload["type"] == "message_stop":
            item = make_empty_response(payload, model)
            yield item
            break
        elif payload["type"] == "error":
            yield make_response(payload, model)
        else:
            # TODO this should be a log entry, as per
            # https://docs.anthropic.com/en/api/messages-streaming#other-events
            raise ValueError(f"anthropic: unexpected event type '{payload['type']}'")

    # The following should always hold when we get here
    assert payload["type"] == "message_stop"
    return


async def anthropic_content_chunk_wrapper(payloads, model):
    async for payload in payloads:
        if os.getenv("CODEGATE_DEBUG_ANTHROPIC") is not None:
            print(f"CODEGATE_DEBUG_ANTHROPIC: anthropic_chunk_wrapper: {payload}")

        if payload["type"] == "content_block_delta":
            pdelta = payload["delta"]
            # We assume that if we have to process text completion if
            # we got into this function.
            assert pdelta["type"] == "text_delta"

            delta = CodegateDelta(
                role="assistant",
                content=pdelta["text"],
                tool_calls=None,
            )
            choice = CodegateStreamingChoices(
                index=payload["index"],
                delta=delta,
            )

            item = make_response(payload, model, choice)
            yield item
        elif payload["type"] == "content_block_stop":
            item = make_empty_response(payload, model)
            yield item
            # We break the loop at this point since we have to go back
            # processing the rest of the state machine.
            return
        elif payload["type"] == "error":
            yield make_response(payload, model)


async def anthropic_tool_chunk_wrapper(payloads, model):
    async for payload in payloads:
        if os.getenv("CODEGATE_DEBUG_ANTHROPIC") is not None:
            print(f"CODEGATE_DEBUG_ANTHROPIC: anthropic_chunk_wrapper: {payload}")

        if payload["type"] == "content_block_delta":
            pdelta = payload["delta"]
            # We assume that if we have to process json completion if
            # we got into this function.
            assert pdelta["type"] == "input_json_delta"

            function = CodegateFunction(
                # Function name is emitted as part of the
                # `content_block_start`.
                name=None,
                arguments=pdelta["partial_json"],
            )
            tool_call = CodegateChatCompletionDeltaToolCall(
                # id=delta["id"],
                function=function,
                type="function",
                index=0,
            )
            delta = CodegateDelta(
                role="assistant",
                content=None,
                tool_calls=[
                    tool_call,
                ],
            )
            choice = CodegateStreamingChoices(
                index=payload["index"],
                delta=delta,
            )

            item = make_response(payload, model, choice)
            yield item
        elif payload["type"] == "content_block_stop":
            item = make_empty_response(payload, model)
            yield item
            # We break the loop at this point since we have to go back
            # processing the rest of the state machine.
            return
        elif payload["type"] == "error":
            yield make_response(payload, model)


def make_empty_response(payload: str, model: str):
    delta = CodegateDelta(
        role="assistant",
        content=None,
        tool_calls=None,
    )
    choice = CodegateStreamingChoices(
        index=payload.get("index", 0),
        delta=delta,
    )
    return make_response(payload, model, choice)


def make_response(payload: str, model: str, *choices):
    return CodegateModelResponseStream(
        # id=None,
        created=None,
        model=model,
        object='chat.completion.chunk',
        choices=list(choices),
        payload=payload,
    )
