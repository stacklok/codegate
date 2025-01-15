import asyncio
import json
from pathlib import Path
from typing import List, Optional, Type

import structlog
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from pydantic import BaseModel
from sqlalchemy import TextClause, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import create_async_engine

from codegate.db.fim_cache import FimCache
from codegate.db.models import (
    Alert,
    GetAlertsWithPromptAndOutputRow,
    GetPromptWithOutputsRow,
    Output,
    Prompt,
    Workspace,
)
from codegate.pipeline.base import PipelineContext

logger = structlog.get_logger("codegate")
alert_queue = asyncio.Queue()
fim_cache = FimCache()


class DbCodeGate:

    def __init__(self, sqlite_path: Optional[str] = None):
        # Initialize SQLite database engine with proper async URL
        if not sqlite_path:
            current_dir = Path(__file__).parent
            sqlite_path = (
                current_dir.parent.parent.parent / "codegate_volume" / "db" / "codegate.db"
            )  # type: ignore
        self._db_path = Path(sqlite_path).absolute()  # type: ignore
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Connecting to DB from path: {self._db_path}")
        engine_dict = {
            "url": f"sqlite+aiosqlite:///{self._db_path}",
            "echo": False,  # Set to False in production
            "isolation_level": "AUTOCOMMIT",  # Required for SQLite
        }
        self._async_db_engine = create_async_engine(**engine_dict)

    def does_db_exist(self):
        return self._db_path.is_file()


class DbRecorder(DbCodeGate):

    def __init__(self, sqlite_path: Optional[str] = None):
        super().__init__(sqlite_path)

    async def _execute_update_pydantic_model(
        self, model: BaseModel, sql_command: TextClause
    ) -> Optional[BaseModel]:
        """Execute an update or insert command for a Pydantic model."""
        async with self._async_db_engine.begin() as conn:
            try:
                result = await conn.execute(sql_command, model.model_dump())
                row = result.first()
                if row is None:
                    return None

                # Get the class of the Pydantic object to create a new object
                model_class = model.__class__
                return model_class(**row._asdict())
            except Exception as e:
                logger.error(f"Failed to update model: {model}.", error=str(e))
                return None

    async def record_request(self, prompt_params: Optional[Prompt] = None) -> Optional[Prompt]:
        if prompt_params is None:
            return None
        sql = text(
            """
                INSERT INTO prompts (id, timestamp, provider, request, type)
                VALUES (:id, :timestamp, :provider, :request, :type)
                RETURNING *
                """
        )
        recorded_request = await self._execute_update_pydantic_model(prompt_params, sql)
        # Uncomment to debug the recorded request
        # logger.debug(f"Recorded request: {recorded_request}")
        return recorded_request  # type: ignore

    async def update_request(
        self, initial_id: str, prompt_params: Optional[Prompt] = None
    ) -> Optional[Prompt]:
        if prompt_params is None:
            return None
        prompt_params.id = initial_id  # overwrite the initial id of the request
        sql = text(
            """
                UPDATE prompts
                SET timestamp = :timestamp, provider = :provider, request = :request, type = :type
                WHERE id = :id
                RETURNING *
                """
        )
        updated_request = await self._execute_update_pydantic_model(prompt_params, sql)
        # Uncomment to debug the recorded request
        # logger.debug(f"Recorded request: {recorded_request}")
        return updated_request  # type: ignore

    async def record_outputs(
        self, outputs: List[Output], initial_id: Optional[str]
    ) -> Optional[Output]:
        if not outputs:
            return

        first_output = outputs[0]
        # Create a single entry on DB but encode all of the chunks in the stream as a list
        # of JSON objects in the field `output`
        if initial_id:
            first_output.prompt_id = initial_id
        output_db = Output(
            id=first_output.id,
            prompt_id=first_output.prompt_id,
            timestamp=first_output.timestamp,
            output=first_output.output,
        )
        full_outputs = []
        # Just store the model respnses in the list of JSON objects.
        for output in outputs:
            full_outputs.append(output.output)
        output_db.output = json.dumps(full_outputs)

        sql = text(
            """
                INSERT INTO outputs (id, prompt_id, timestamp, output)
                VALUES (:id, :prompt_id, :timestamp, :output)
                RETURNING *
                """
        )
        recorded_output = await self._execute_update_pydantic_model(output_db, sql)
        # Uncomment to debug
        # logger.debug(f"Recorded output: {recorded_output}")
        return recorded_output  # type: ignore

    async def record_alerts(self, alerts: List[Alert], initial_id: Optional[str]) -> List[Alert]:
        if not alerts:
            return []
        sql = text(
            """
                INSERT INTO alerts (
                id, prompt_id, code_snippet, trigger_string, trigger_type, trigger_category,
                timestamp
                )
                VALUES (:id, :prompt_id, :code_snippet, :trigger_string, :trigger_type,
                :trigger_category, :timestamp)
                RETURNING *
                """
        )
        # We can insert each alert independently in parallel.
        alerts_tasks = []
        async with asyncio.TaskGroup() as tg:
            for alert in alerts:
                try:
                    if initial_id:
                        alert.prompt_id = initial_id
                    result = tg.create_task(self._execute_update_pydantic_model(alert, sql))
                    alerts_tasks.append(result)
                except Exception as e:
                    logger.error(f"Failed to record alert: {alert}.", error=str(e))

        recorded_alerts = []
        for alert_coro in alerts_tasks:
            alert_result = alert_coro.result()
            recorded_alerts.append(alert_result)
            if alert_result and alert_result.trigger_category == "critical":
                await alert_queue.put(f"New alert detected: {alert.timestamp}")
        # Uncomment to debug the recorded alerts
        # logger.debug(f"Recorded alerts: {recorded_alerts}")
        return recorded_alerts

    def _should_record_context(self, context: Optional[PipelineContext]) -> tuple:
        """Check if the context should be recorded in DB and determine the action."""
        if context is None or context.metadata.get("stored_in_db", False):
            return False, None, None

        if not context.input_request:
            logger.warning("No input request found. Skipping recording context.")
            return False, None, None

        # If it's not a FIM prompt, we don't need to check anything else.
        if context.input_request.type != "fim":
            return True, "add", ""  # Default to add if not FIM, since no cache check is required

        return fim_cache.could_store_fim_request(context)  # type: ignore

    async def record_context(self, context: Optional[PipelineContext]) -> None:
        try:
            if not context:
                logger.info("No context provided, skipping")
                return
            should_record, action, initial_id = self._should_record_context(context)
            if not should_record:
                logger.info("Skipping record of context, not needed")
                return
            if action == "add":
                await self.record_request(context.input_request)
                await self.record_outputs(context.output_responses, None)
                await self.record_alerts(context.alerts_raised, None)
                context.metadata["stored_in_db"] = True
                logger.info(
                    f"Recorded context in DB. Output chunks: {len(context.output_responses)}. "
                    f"Alerts: {len(context.alerts_raised)}."
                )
            else:
                # update them
                await self.update_request(initial_id, context.input_request)
                await self.record_outputs(context.output_responses, initial_id)
                await self.record_alerts(context.alerts_raised, initial_id)
                context.metadata["stored_in_db"] = True
                logger.info(
                    f"Recorded context in DB. Output chunks: {len(context.output_responses)}. "
                    f"Alerts: {len(context.alerts_raised)}."
                )
        except Exception as e:
            logger.error(f"Failed to record context: {context}.", error=str(e))


class DbReader(DbCodeGate):

    def __init__(self, sqlite_path: Optional[str] = None):
        super().__init__(sqlite_path)

    async def _execute_select_pydantic_model(
        self, model_type: Type[BaseModel], sql_command: TextClause
    ) -> Optional[BaseModel]:
        async with self._async_db_engine.begin() as conn:
            try:
                result = await conn.execute(sql_command)
                if not result:
                    return None
                rows = [model_type(**row._asdict()) for row in result.fetchall() if row]
                return rows
            except Exception as e:
                logger.error(f"Failed to select model: {model_type}.", error=str(e))
                return None

    async def get_prompts_with_output(self) -> List[GetPromptWithOutputsRow]:
        sql = text(
            """
            SELECT
                p.id, p.timestamp, p.provider, p.request, p.type,
                o.id as output_id,
                o.output,
                o.timestamp as output_timestamp
            FROM prompts p
            LEFT JOIN outputs o ON p.id = o.prompt_id
            ORDER BY o.timestamp DESC
            """
        )
        prompts = await self._execute_select_pydantic_model(GetPromptWithOutputsRow, sql)
        return prompts

    async def get_alerts_with_prompt_and_output(self) -> List[GetAlertsWithPromptAndOutputRow]:
        sql = text(
            """
            SELECT
                a.id,
                a.prompt_id,
                a.code_snippet,
                a.trigger_string,
                a.trigger_type,
                a.trigger_category,
                a.timestamp,
                p.timestamp as prompt_timestamp,
                p.provider,
                p.request,
                p.type,
                o.id as output_id,
                o.output,
                o.timestamp as output_timestamp
            FROM alerts a
            LEFT JOIN prompts p ON p.id = a.prompt_id
            LEFT JOIN outputs o ON p.id = o.prompt_id
            ORDER BY a.timestamp DESC
            """
        )
        prompts = await self._execute_select_pydantic_model(GetAlertsWithPromptAndOutputRow, sql)
        return prompts

    async def get_workspaces(self) -> List[Workspace]:
        sql = text(
            """
            SELECT
                id, name, is_active
            FROM workspaces
            ORDER BY is_active DESC
            """
        )
        workspaces = await self._execute_select_pydantic_model(Workspace, sql)
        return workspaces


def init_db_sync(db_path: Optional[str] = None):
    """DB will be initialized in the constructor in case it doesn't exist."""
    current_dir = Path(__file__).parent
    alembic_ini_path = current_dir.parent.parent.parent / "alembic.ini"
    alembic_cfg = AlembicConfig(alembic_ini_path)
    # Only set the db path if it's provided. Otherwise use the one in alembic.ini
    if db_path:
        alembic_cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")

    try:
        alembic_command.upgrade(alembic_cfg, "head")
    except OperationalError:
        # An OperationalError is expected if the DB already exists, i.e. it was created before
        # migrations were introduced. In this case, we need to stamp the DB with the initial
        # revision and then upgrade it to the latest revision.
        alembic_command.stamp(alembic_cfg, "30d0144e1a50")
        alembic_command.upgrade(alembic_cfg, "head")
    logger.info("DB initialized successfully.")


if __name__ == "__main__":
    init_db_sync()
