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


WorkspaceNameStr = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, to_lower=True, pattern=r"^[a-zA-Z0-9_-]+$", strict=True
    ),
]


class WorkspaceRow(BaseModel):
    """A workspace row entry.

    Since our model currently includes instructions
    in the same table, this is returned as a single
    object.
    """

    id: str
    name: WorkspaceNameStr
    custom_instructions: Optional[str]


class GetWorkspaceByNameConditions(BaseModel):
    name: WorkspaceNameStr

    def get_conditions(self):
        return {"name": self.name}


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


class WorkspaceWithSessionInfo(BaseModel):
    """Returns a workspace ID with an optional
    session ID. If the session ID is None, then
    the workspace is not active.
    """

    id: str
    name: WorkspaceNameStr
    session_id: Optional[str]


class ActiveWorkspace(BaseModel):
    """Returns a full active workspace object with the
    with the session information.
    """

    id: str
    name: WorkspaceNameStr
    custom_instructions: Optional[str]
    session_id: str
    last_update: datetime.datetime
