import datetime
from typing import Annotated, Any, Optional

from pydantic import BaseModel, StringConstraints


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


WorskpaceNameStr = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, to_lower=True, pattern=r"^[a-zA-Z0-9_-]+$", strict=True
    ),
]


class Workspace(BaseModel):
    id: str
    name: WorskpaceNameStr
    custom_instructions: Optional[str]


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
    custom_instructions: Optional[str]
    session_id: str
    last_update: datetime.datetime
