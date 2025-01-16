import datetime
from typing import Any, Optional

import pydantic


class Alert(pydantic.BaseModel):
    id: Any
    prompt_id: Any
    code_snippet: Optional[Any]
    trigger_string: Optional[Any]
    trigger_type: Any
    trigger_category: Optional[Any]
    timestamp: Any


class Output(pydantic.BaseModel):
    id: Any
    prompt_id: Any
    timestamp: Any
    output: Any


class Prompt(pydantic.BaseModel):
    id: Any
    timestamp: Any
    provider: Optional[Any]
    request: Any
    type: Any
    workspace_id: Optional[str]


class Setting(pydantic.BaseModel):
    id: Any
    ip: Optional[Any]
    port: Optional[Any]
    llm_model: Optional[Any]
    system_prompt: Optional[Any]
    other_settings: Optional[Any]


class Workspace(pydantic.BaseModel):
    id: str
    name: str


class Session(pydantic.BaseModel):
    id: str
    active_workspace_id: str
    last_update: datetime.datetime


# Models for select queries


class GetAlertsWithPromptAndOutputRow(pydantic.BaseModel):
    id: Any
    prompt_id: Any
    code_snippet: Optional[Any]
    trigger_string: Optional[Any]
    trigger_type: Any
    trigger_category: Optional[Any]
    timestamp: Any
    prompt_timestamp: Optional[Any]
    provider: Optional[Any]
    request: Optional[Any]
    type: Optional[Any]
    output_id: Optional[Any]
    output: Optional[Any]
    output_timestamp: Optional[Any]


class GetPromptWithOutputsRow(pydantic.BaseModel):
    id: Any
    timestamp: Any
    provider: Optional[Any]
    request: Any
    type: Any
    output_id: Optional[Any]
    output: Optional[Any]
    output_timestamp: Optional[Any]


class WorkspaceActive(pydantic.BaseModel):
    id: str
    name: str
    active_workspace_id: Optional[str]


class ActiveWorkspace(pydantic.BaseModel):
    id: str
    name: str
    session_id: str
    last_update: datetime.datetime
