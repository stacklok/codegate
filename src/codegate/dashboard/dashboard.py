import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from typing import AsyncGenerator, List, Optional

from httpx import AsyncClient, HTTPStatusError
import structlog
from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from codegate import __version__

from codegate.dashboard.post_processing import (
    parse_get_alert_conversation,
    parse_messages_in_conversations,
)
from codegate.dashboard.request_models import AlertConversation, Conversation
from codegate.db.connection import DbReader, alert_queue

logger = structlog.get_logger("codegate")

dashboard_router = APIRouter(tags=["Dashboard"])
db_reader = None

def get_db_reader():
    global db_reader
    if db_reader is None:
        db_reader = DbReader()
    return db_reader

def get_http_client() -> AsyncClient:
    return AsyncClient()

async def fetch_latest_version(client: AsyncClient) -> str:
    url = "https://api.github.com/repos/stacklok/codegate/releases/latest"
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    response = await client.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    return data.get("tag_name", "unknown")

def fetch_latest_version_sync(client: AsyncClient) -> str:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(fetch_latest_version(client))

@dashboard_router.get("/dashboard/messages")
def get_messages(db_reader: DbReader = Depends(get_db_reader)) -> List[Conversation]:
    """
    Get all the messages from the database and return them as a list of conversations.
    """
    prompts_outputs = asyncio.run(db_reader.get_prompts_with_output())

    return asyncio.run(parse_messages_in_conversations(prompts_outputs))


@dashboard_router.get("/dashboard/alerts")
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


@dashboard_router.get("/dashboard/alerts_notification")
async def stream_sse():
    """
    Send alerts event
    """
    return StreamingResponse(generate_sse_events(), media_type="text/event-stream")

@dashboard_router.get("/dashboard/version")
def version_check(client: AsyncClient = Depends(get_http_client)):
    try:
        with ThreadPoolExecutor() as executor:
            latest_version = executor.submit(fetch_latest_version_sync, client).result()

        # normalize the versions as github will return them with a 'v' prefix
        current_version = __version__.lstrip('v')
        latest_version_stripped = latest_version.lstrip('v')

        is_latest: bool = latest_version_stripped == current_version
        
        return {
            "current_version": current_version,
            "latest_version": latest_version_stripped,
            "is_latest": is_latest,
            "error": None,
        }
    except HTTPException as e:
        return {
            "current_version": __version__,
            "latest_version": "unknown",
            "is_latest": None,
            "error": e.detail
        }


def generate_openapi():
    # Create a temporary FastAPI app instance
    app = FastAPI()

    # Include your defined router
    app.include_router(dashboard_router)

    # Generate OpenAPI JSON
    openapi_schema = app.openapi()

    # Convert the schema to JSON string for easier handling or storage
    openapi_json = json.dumps(openapi_schema, indent=2)
    print(openapi_json)
