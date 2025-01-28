import asyncio
import json
import uuid
from pathlib import Path
from typing import List, Optional, Type

import structlog
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from pydantic import BaseModel
from sqlalchemy import CursorResult, TextClause, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import create_async_engine

from codegate.db.fim_cache import FimCache
from codegate.db.models import (
    ActiveWorkspace,
    Alert,
    GetPromptWithOutputsRow,
    GetWorkspaceByNameConditions,
    Output,
    Prompt,
    ProviderAuthMaterial,
    ProviderEndpoint,
    ProviderModel,
    Session,
    WorkspaceRow,
    WorkspaceWithSessionInfo,
)
from codegate.db.token_usage import TokenUsageParser
from codegate.pipeline.base import PipelineContext

logger = structlog.get_logger("codegate")
alert_queue = asyncio.Queue()
fim_cache = FimCache()


class AlreadyExistsError(Exception):
    pass


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """
    Ensures that foreign keys are enabled for the SQLite database at every connection.
    SQLite does not enforce foreign keys by default, so we need to enable them manually.
    [SQLAlchemy docs](https://docs.sqlalchemy.org/en/20/dialects/sqlite.html#foreign-key-support)
    [SQLite docs](https://www.sqlite.org/foreignkeys.html)
    [SO](https://stackoverflow.com/questions/2614984/sqlite-sqlalchemy-how-to-enforce-foreign-keys)
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


class DbCodeGate:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, sqlite_path: Optional[str] = None):
        if not hasattr(self, "_initialized"):
            # Ensure __init__ is only executed once
            self._initialized = True

            # Initialize SQLite database engine with proper async URL
            if not sqlite_path:
                current_dir = Path(__file__).parent
                sqlite_path = (
                    current_dir.parent.parent.parent / "codegate_volume" / "db" / "codegate.db"
                )
            self._db_path = Path(sqlite_path).absolute()
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            # logger.debug(f"Connecting to DB from path: {self._db_path}")
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
        self, model: BaseModel, sql_command: TextClause, should_raise: bool = False
    ) -> Optional[BaseModel]:
        """Execute an update or insert command for a Pydantic model."""
        try:
            async with self._async_db_engine.begin() as conn:
                result = await conn.execute(sql_command, model.model_dump())
                row = result.first()
                if row is None:
                    return None

                # Get the class of the Pydantic object to create a new object
                model_class = model.__class__
                return model_class(**row._asdict())
        except Exception as e:
            logger.error(f"Failed to update model: {model}.", error=str(e))
            if should_raise:
                raise e
            return None

    async def record_request(self, prompt_params: Optional[Prompt] = None) -> Optional[Prompt]:
        if prompt_params is None:
            return None
        # Get the active workspace to store the request
        active_workspace = await DbReader().get_active_workspace()
        workspace_id = active_workspace.id if active_workspace else "1"
        prompt_params.workspace_id = workspace_id
        sql = text(
            """
                INSERT INTO prompts (id, timestamp, provider, request, type, workspace_id)
                VALUES (:id, :timestamp, :provider, :request, :type, :workspace_id)
                ON CONFLICT(id) DO UPDATE SET
                timestamp = excluded.timestamp, provider = excluded.provider,
                request = excluded.request, type = excluded.type,
                workspace_id = excluded.workspace_id
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

        # Parse the token usage from the outputs
        token_parser = TokenUsageParser()
        full_token_usage = await token_parser.parse_outputs(outputs)

        output_db.output = json.dumps(full_outputs)
        output_db.input_tokens = full_token_usage.input_tokens
        output_db.output_tokens = full_token_usage.output_tokens
        output_db.input_cost = full_token_usage.input_cost
        output_db.output_cost = full_token_usage.output_cost

        sql = text(
            """
                INSERT INTO outputs (
                    id, prompt_id, timestamp, output, input_tokens, output_tokens, input_cost,
                    output_cost
                )
                VALUES (
                    :id, :prompt_id, :timestamp, :output, :input_tokens, :output_tokens,
                    :input_cost, :output_cost
                )
                ON CONFLICT (id) DO UPDATE SET
                timestamp = excluded.timestamp,
                output = excluded.output,
                input_tokens = excluded.input_tokens,
                output_tokens = excluded.output_tokens,
                input_cost = excluded.input_cost,
                output_cost = excluded.output_cost
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
                ON CONFLICT (id) DO UPDATE SET
                code_snippet = excluded.code_snippet, trigger_string = excluded.trigger_string,
                trigger_type = excluded.trigger_type, trigger_category = excluded.trigger_category,
                timestamp = excluded.timestamp, prompt_id = excluded.prompt_id
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
                logger.info(
                    f"Recorded context in DB. Output chunks: {len(context.output_responses)}. "
                    f"Alerts: {len(context.alerts_raised)}."
                )
            else:
                # update them
                await self.update_request(initial_id, context.input_request)
                await self.record_outputs(context.output_responses, initial_id)
                await self.record_alerts(context.alerts_raised, initial_id)
                logger.info(
                    f"Recorded context in DB. Output chunks: {len(context.output_responses)}. "
                    f"Alerts: {len(context.alerts_raised)}."
                )
        except Exception as e:
            logger.error(f"Failed to record context: {context}.", error=str(e))

    async def add_workspace(self, workspace_name: str) -> WorkspaceRow:
        """Add a new workspace to the DB.

        This handles validation and insertion of a new workspace.

        It may raise a ValidationError if the workspace name is invalid.
        or a AlreadyExistsError if the workspace already exists.
        """
        workspace = WorkspaceRow(
            id=str(uuid.uuid4()), name=workspace_name, custom_instructions=None
        )
        sql = text(
            """
            INSERT INTO workspaces (id, name)
            VALUES (:id, :name)
            RETURNING *
            """
        )

        try:
            added_workspace = await self._execute_update_pydantic_model(
                workspace, sql, should_raise=True
            )
        except IntegrityError as e:
            logger.debug(f"Exception type: {type(e)}")
            raise AlreadyExistsError(f"Workspace {workspace_name} already exists.")
        return added_workspace

    async def update_workspace(self, workspace: WorkspaceRow) -> WorkspaceRow:
        sql = text(
            """
            UPDATE workspaces SET
            name = :name,
            custom_instructions = :custom_instructions
            WHERE id = :id
            RETURNING *
            """
        )
        updated_workspace = await self._execute_update_pydantic_model(
            workspace, sql, should_raise=True
        )
        return updated_workspace

    async def update_session(self, session: Session) -> Optional[Session]:
        sql = text(
            """
            INSERT INTO sessions (id, active_workspace_id, last_update)
            VALUES (:id, :active_workspace_id, :last_update)
            ON CONFLICT (id) DO UPDATE SET
            active_workspace_id = excluded.active_workspace_id, last_update = excluded.last_update
            WHERE id = excluded.id
            RETURNING *
            """
        )
        # We only pass an object to respect the signature of the function
        active_session = await self._execute_update_pydantic_model(session, sql, should_raise=True)
        return active_session

    async def soft_delete_workspace(self, workspace: WorkspaceRow) -> Optional[WorkspaceRow]:
        sql = text(
            """
            UPDATE workspaces
            SET deleted_at = CURRENT_TIMESTAMP
            WHERE id = :id
            RETURNING *
            """
        )
        deleted_workspace = await self._execute_update_pydantic_model(
            workspace, sql, should_raise=True
        )
        return deleted_workspace

    async def hard_delete_workspace(self, workspace: WorkspaceRow) -> Optional[WorkspaceRow]:
        sql = text(
            """
            DELETE FROM workspaces
            WHERE id = :id
            RETURNING *
            """
        )
        deleted_workspace = await self._execute_update_pydantic_model(
            workspace, sql, should_raise=True
        )
        return deleted_workspace

    async def recover_workspace(self, workspace: WorkspaceRow) -> Optional[WorkspaceRow]:
        sql = text(
            """
            UPDATE workspaces
            SET deleted_at = NULL
            WHERE id = :id
            RETURNING *
            """
        )
        recovered_workspace = await self._execute_update_pydantic_model(
            workspace, sql, should_raise=True
        )
        return recovered_workspace

    async def add_provider_endpoint(self, provider: ProviderEndpoint) -> ProviderEndpoint:
        sql = text(
            """
            INSERT INTO provider_endpoints (
                id, name, description, provider_type, endpoint, auth_type, auth_blob
            )
            VALUES (:id, :name, :description, :provider_type, :endpoint, :auth_type, "")
            RETURNING *
            """
        )
        added_provider = await self._execute_update_pydantic_model(provider, sql, should_raise=True)
        return added_provider

    async def update_provider_endpoint(self, provider: ProviderEndpoint) -> ProviderEndpoint:
        sql = text(
            """
            UPDATE provider_endpoints
            SET name = :name, description = :description, provider_type = :provider_type,
            endpoint = :endpoint, auth_type = :auth_type
            WHERE id = :id
            RETURNING *
            """
        )
        updated_provider = await self._execute_update_pydantic_model(
            provider, sql, should_raise=True
        )
        return updated_provider

    async def delete_provider_endpoint(
        self,
        provider: ProviderEndpoint,
    ) -> Optional[ProviderEndpoint]:
        sql = text(
            """
            DELETE FROM provider_endpoints
            WHERE id = :id
            RETURNING *
            """
        )
        deleted_provider = await self._execute_update_pydantic_model(
            provider, sql, should_raise=True
        )
        return deleted_provider

    async def push_provider_auth_material(self, auth_material: ProviderAuthMaterial):
        sql = text(
            """
            UPDATE provider_endpoints
            SET auth_type = :auth_type, auth_blob = :auth_blob
            WHERE id = :provider_endpoint_id
            """
        )
        _ = await self._execute_update_pydantic_model(auth_material, sql, should_raise=True)
        return

    async def add_provider_model(self, model: ProviderModel) -> ProviderModel:
        sql = text(
            """
            INSERT INTO provider_models (provider_endpoint_id, name)
            VALUES (:provider_endpoint_id, :name)
            RETURNING *
            """
        )
        added_model = await self._execute_update_pydantic_model(model, sql, should_raise=True)
        return added_model


class DbReader(DbCodeGate):

    def __init__(self, sqlite_path: Optional[str] = None):
        super().__init__(sqlite_path)

    async def _dump_result_to_pydantic_model(
        self, model_type: Type[BaseModel], result: CursorResult
    ) -> Optional[List[BaseModel]]:
        try:
            if not result:
                return None
            rows = [model_type(**row._asdict()) for row in result.fetchall() if row]
            return rows
        except Exception as e:
            logger.error(f"Failed to dump to pydantic model: {model_type}.", error=str(e))
            return None

    async def _execute_select_pydantic_model(
        self, model_type: Type[BaseModel], sql_command: TextClause
    ) -> Optional[List[BaseModel]]:
        async with self._async_db_engine.begin() as conn:
            try:
                result = await conn.execute(sql_command)
                return await self._dump_result_to_pydantic_model(model_type, result)
            except Exception as e:
                logger.error(f"Failed to select model: {model_type}.", error=str(e))
                return None

    async def _exec_select_conditions_to_pydantic(
        self,
        model_type: Type[BaseModel],
        sql_command: TextClause,
        conditions: dict,
        should_raise: bool = False,
    ) -> Optional[List[BaseModel]]:
        async with self._async_db_engine.begin() as conn:
            try:
                result = await conn.execute(sql_command, conditions)
                return await self._dump_result_to_pydantic_model(model_type, result)
            except Exception as e:
                logger.error(f"Failed to select model with conditions: {model_type}.", error=str(e))
                # Exposes errors to the caller
                if should_raise:
                    raise e
                return None

    async def get_prompts_with_output(self, workpace_id: str) -> List[GetPromptWithOutputsRow]:
        sql = text(
            """
            SELECT
                p.id, p.timestamp, p.provider, p.request, p.type,
                o.id as output_id,
                o.output,
                o.timestamp as output_timestamp,
                o.input_tokens,
                o.output_tokens,
                o.input_cost,
                o.output_cost
            FROM prompts p
            LEFT JOIN outputs o ON p.id = o.prompt_id
            WHERE p.workspace_id = :workspace_id
            ORDER BY o.timestamp DESC
            """
        )
        conditions = {"workspace_id": workpace_id}
        prompts = await self._exec_select_conditions_to_pydantic(
            GetPromptWithOutputsRow, sql, conditions, should_raise=True
        )
        return prompts

    async def get_alerts_by_workspace(self, workspace_id: str) -> List[Alert]:
        sql = text(
            """
            SELECT
                a.id,
                a.prompt_id,
                a.code_snippet,
                a.trigger_string,
                a.trigger_type,
                a.trigger_category,
                a.timestamp
            FROM alerts a
            INNER JOIN prompts p ON p.id = a.prompt_id
            WHERE p.workspace_id = :workspace_id
            ORDER BY a.timestamp DESC
            """
        )
        conditions = {"workspace_id": workspace_id}
        prompts = await self._exec_select_conditions_to_pydantic(
            Alert, sql, conditions, should_raise=True
        )
        return prompts

    async def get_workspaces(self) -> List[WorkspaceWithSessionInfo]:
        sql = text(
            """
            SELECT
                w.id, w.name, s.id as session_id
            FROM workspaces w
            LEFT JOIN sessions s ON w.id = s.active_workspace_id
            WHERE w.deleted_at IS NULL
            """
        )
        workspaces = await self._execute_select_pydantic_model(WorkspaceWithSessionInfo, sql)
        return workspaces

    async def get_archived_workspaces(self) -> List[WorkspaceRow]:
        sql = text(
            """
            SELECT
                id, name, custom_instructions
            FROM workspaces
            WHERE deleted_at IS NOT NULL
            ORDER BY deleted_at DESC
            """
        )
        workspaces = await self._execute_select_pydantic_model(WorkspaceRow, sql)
        return workspaces

    async def get_workspace_by_name(self, name: str) -> Optional[WorkspaceRow]:
        sql = text(
            """
            SELECT
                id, name, custom_instructions
            FROM workspaces
            WHERE name = :name AND deleted_at IS NULL
            """
        )
        conditions = GetWorkspaceByNameConditions(name=name).get_conditions()
        workspaces = await self._exec_select_conditions_to_pydantic(
            WorkspaceRow, sql, conditions, should_raise=True
        )
        return workspaces[0] if workspaces else None

    async def get_archived_workspace_by_name(self, name: str) -> Optional[WorkspaceRow]:
        sql = text(
            """
            SELECT
                id, name, custom_instructions
            FROM workspaces
            WHERE name = :name AND deleted_at IS NOT NULL
            """
        )
        conditions = GetWorkspaceByNameConditions(name=name).get_conditions()
        workspaces = await self._exec_select_conditions_to_pydantic(
            WorkspaceRow, sql, conditions, should_raise=True
        )
        return workspaces[0] if workspaces else None

    async def get_sessions(self) -> List[Session]:
        sql = text(
            """
            SELECT
                id, active_workspace_id, last_update
            FROM sessions
            """
        )
        sessions = await self._execute_select_pydantic_model(Session, sql)
        return sessions

    async def get_active_workspace(self) -> Optional[ActiveWorkspace]:
        sql = text(
            """
            SELECT
                w.id, w.name, w.custom_instructions, s.id as session_id, s.last_update
            FROM sessions s
            INNER JOIN workspaces w ON w.id = s.active_workspace_id
            """
        )
        active_workspace = await self._execute_select_pydantic_model(ActiveWorkspace, sql)
        return active_workspace[0] if active_workspace else None

    async def get_provider_endpoint_by_name(self, provider_name: str) -> Optional[ProviderEndpoint]:
        sql = text(
            """
            SELECT id, name, description, provider_type, endpoint, auth_type, created_at, updated_at
            FROM provider_endpoints
            WHERE name = :name
            """
        )
        conditions = {"name": provider_name}
        provider = await self._exec_select_conditions_to_pydantic(
            ProviderEndpoint, sql, conditions, should_raise=True
        )
        return provider[0] if provider else None

    async def get_provider_endpoint_by_id(self, provider_id: str) -> Optional[ProviderEndpoint]:
        sql = text(
            """
            SELECT id, name, description, provider_type, endpoint, auth_type, created_at, updated_at
            FROM provider_endpoints
            WHERE id = :id
            """
        )
        conditions = {"id": provider_id}
        provider = await self._exec_select_conditions_to_pydantic(
            ProviderEndpoint, sql, conditions, should_raise=True
        )
        return provider[0] if provider else None

    async def get_provider_endpoints(self) -> List[ProviderEndpoint]:
        sql = text(
            """
            SELECT id, name, description, provider_type, endpoint, auth_type, created_at, updated_at
            FROM provider_endpoints
            """
        )
        providers = await self._execute_select_pydantic_model(ProviderEndpoint, sql)
        return providers

    async def get_provider_models_by_provider_id(self, provider_id: str) -> List[ProviderModel]:
        sql = text(
            """
            SELECT provider_endpoint_id, name
            FROM provider_models
            WHERE provider_endpoint_id = :provider_endpoint_id
            """
        )
        conditions = {"provider_endpoint_id": provider_id}
        models = await self._exec_select_conditions_to_pydantic(
            ProviderModel, sql, conditions, should_raise=True
        )
        return models

    async def get_all_provider_models(self) -> List[ProviderModel]:
        sql = text(
            """
            SELECT pm.provider_endpoint_id, pm.name, pe.name as provider_endpoint_name
            FROM provider_models pm
            INNER JOIN provider_endpoints pe ON pm.provider_endpoint_id = pe.id
            """
        )
        models = await self._execute_select_pydantic_model(ProviderModel, sql)
        return models


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


def init_session_if_not_exists(db_path: Optional[str] = None):
    import datetime

    db_reader = DbReader(db_path)
    sessions = asyncio.run(db_reader.get_sessions())
    # If there are no sessions, create a new one
    # TODO: For the moment there's a single session. If it already exists, we don't create a new one
    if not sessions:
        session = Session(
            id=str(uuid.uuid4()),
            active_workspace_id="1",
            last_update=datetime.datetime.now(datetime.timezone.utc),
        )
        db_recorder = DbRecorder(db_path)
        try:
            asyncio.run(db_recorder.update_session(session))
        except Exception as e:
            logger.error(f"Failed to initialize session in DB: {e}")
            return
        logger.info("Session in DB initialized successfully.")


if __name__ == "__main__":
    init_db_sync()
