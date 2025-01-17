import asyncio
from typing import AsyncGenerator, List, Optional

import requests
import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute

from codegate import __version__
from codegate.api.dashboard.post_processing import (
    parse_get_alert_conversation,
    parse_messages_in_conversations,
)
from codegate.api.dashboard.request_models import AlertConversation, Conversation
from codegate.db.connection import DbReader, alert_queue

logger = structlog.get_logger("codegate")

dashboard_router = APIRouter()
db_reader = None


def uniq_name(route: APIRoute):
    return f"v1_{route.name}"


def get_db_reader():
    global db_reader
    if db_reader is None:
        db_reader = DbReader()
    return db_reader


def fetch_latest_version() -> str:
    url = "https://api.github.com/repos/stacklok/codegate/releases/latest"
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    response = requests.get(url, headers=headers, timeout=5)
    response.raise_for_status()
    data = response.json()
    return data.get("tag_name", "unknown")


@dashboard_router.get(
    "/dashboard/messages", tags=["Dashboard"], generate_unique_id_function=uniq_name
)
def get_messages(db_reader: DbReader = Depends(get_db_reader)) -> List[Conversation]:
    """
    Get all the messages from the database and return them as a list of conversations.
    """
    prompts_outputs = asyncio.run(db_reader.get_prompts_with_output())

    return asyncio.run(parse_messages_in_conversations(prompts_outputs))


@dashboard_router.get(
    "/dashboard/alerts", tags=["Dashboard"], generate_unique_id_function=uniq_name
)
def get_alerts(db_reader: DbReader = Depends(get_db_reader)) -> List[Optional[AlertConversation]]:
    """
    Get all the messages from the database and return them as a list of conversations.
    """
    alerts_prompt_output = asyncio.run(db_reader.get_alerts_with_prompt_and_output())
    return asyncio.run(parse_get_alert_conversation(alerts_prompt_output))


async def generate_sse_events() -> AsyncGenerator[str, None]:
    """
    SSE generator from queue
    """
    while True:
        message = await alert_queue.get()
        yield f"data: {message}\n\n"


@dashboard_router.get(
    "/dashboard/alerts_notification", tags=["Dashboard"], generate_unique_id_function=uniq_name
)
async def stream_sse():
    """
    Send alerts event
    """
    return StreamingResponse(generate_sse_events(), media_type="text/event-stream")


@dashboard_router.get(
    "/dashboard/version", tags=["Dashboard"], generate_unique_id_function=uniq_name
)
def version_check():
    try:
        latest_version = fetch_latest_version()

        # normalize the versions as github will return them with a 'v' prefix
        current_version = __version__.lstrip("v")
        latest_version_stripped = latest_version.lstrip("v")

        is_latest: bool = latest_version_stripped == current_version

        return {
            "current_version": current_version,
            "latest_version": latest_version_stripped,
            "is_latest": is_latest,
            "error": None,
        }
    except requests.RequestException as e:
        logger.error(f"RequestException: {str(e)}")
        return {
            "current_version": __version__,
            "latest_version": "unknown",
            "is_latest": None,
            "error": "An error occurred while fetching the latest version",
        }
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {
            "current_version": __version__,
            "latest_version": "unknown",
            "is_latest": None,
            "error": "An unexpected error occurred",
        }
