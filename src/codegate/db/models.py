import datetime
import re
from typing import Any, Optional

from pydantic import BaseModel, field_validator


class Alert(BaseModel):
    id: Any
    prompt_id: Any
    code_snippet: Optional[Any]
    trigger_string: Optional[Any]
    trigger_type: Any
    trigger_category: Optional[Any]
    timestamp: Any


class Output(BaseModel):
    id: Any
    prompt_id: Any
    timestamp: Any
    output: Any


class Prompt(BaseModel):
    id: Any
    timestamp: Any
    provider: Optional[Any]
    request: Any
    type: Any
    workspace_id: Optional[str]


class Setting(BaseModel):
    id: Any
    ip: Optional[Any]
    port: Optional[Any]
    llm_model: Optional[Any]
    system_prompt: Optional[Any]
    other_settings: Optional[Any]


class Workspace(BaseModel):
    id: str
    name: str
    system_prompt: Optional[str]

    @field_validator("name", mode="plain")
    @classmethod
    def validate_name(cls, value):
        if not re.match(r"^[a-zA-Z0-9_-]+$", value):
            raise ValueError("name must be alphanumeric and can only contain _ and -")
        return value


class Session(BaseModel):
    id: str
    active_workspace_id: str
    last_update: datetime.datetime


# Models for select queries


class GetAlertsWithPromptAndOutputRow(BaseModel):
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


class GetPromptWithOutputsRow(BaseModel):
    id: Any
    timestamp: Any
    provider: Optional[Any]
    request: Any
    type: Any
    output_id: Optional[Any]
    output: Optional[Any]
    output_timestamp: Optional[Any]


class WorkspaceActive(BaseModel):
    id: str
    name: str
    active_workspace_id: Optional[str]


class ActiveWorkspace(BaseModel):
    id: str
    name: str
    system_prompt: Optional[str]
    session_id: str
    last_update: datetime.datetime
