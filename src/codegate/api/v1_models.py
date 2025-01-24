import datetime
from enum import Enum
from typing import Any, List, Optional, Union

import pydantic

from codegate.db import models as db_models
from codegate.pipeline.base import CodeSnippet


class Workspace(pydantic.BaseModel):
    name: str
    is_active: bool


class CustomInstructions(pydantic.BaseModel):
    prompt: str


class ActiveWorkspace(Workspace):
    # TODO: use a more specific type for last_updated
    last_updated: Any


class ListWorkspacesResponse(pydantic.BaseModel):
    workspaces: list[Workspace]

    @classmethod
    def from_db_workspaces_with_sessioninfo(
        cls, db_workspaces: List[db_models.WorkspaceWithSessionInfo]
    ) -> "ListWorkspacesResponse":
        return cls(
            workspaces=[
                Workspace(name=ws.name, is_active=ws.session_id is not None) for ws in db_workspaces
            ]
        )

    @classmethod
    def from_db_workspaces(
        cls, db_workspaces: List[db_models.WorkspaceRow]
    ) -> "ListWorkspacesResponse":
        return cls(workspaces=[Workspace(name=ws.name, is_active=False) for ws in db_workspaces])


class ListActiveWorkspacesResponse(pydantic.BaseModel):
    workspaces: list[ActiveWorkspace]

    @classmethod
    def from_db_workspaces(
        cls, ws: Optional[db_models.ActiveWorkspace]
    ) -> "ListActiveWorkspacesResponse":
        if ws is None:
            return cls(workspaces=[])
        return cls(
            workspaces=[ActiveWorkspace(name=ws.name, is_active=True, last_updated=ws.last_update)]
        )


class CreateOrRenameWorkspaceRequest(pydantic.BaseModel):
    name: str

    # If set, rename the workspace to this name. Note that
    # the 'name' field is still required and the workspace
    # workspace must exist.
    rename_to: Optional[str] = None


class ActivateWorkspaceRequest(pydantic.BaseModel):
    name: str


class ChatMessage(pydantic.BaseModel):
    """
    Represents a chat message.
    """

    message: str
    timestamp: datetime.datetime
    message_id: str


class QuestionAnswer(pydantic.BaseModel):
    """
    Represents a question and answer pair.
    """

    question: ChatMessage
    answer: Optional[ChatMessage]


class QuestionType(str, Enum):
    chat = "chat"
    fim = "fim"


class PartialQuestions(pydantic.BaseModel):
    """
    Represents all user messages obtained from a DB row.
    """

    messages: List[str]
    timestamp: datetime.datetime
    message_id: str
    provider: Optional[str]
    type: QuestionType


class PartialQuestionAnswer(pydantic.BaseModel):
    """
    Represents a partial conversation.
    """

    partial_questions: PartialQuestions
    answer: Optional[ChatMessage]


class Conversation(pydantic.BaseModel):
    """
    Represents a conversation.
    """

    question_answers: List[QuestionAnswer]
    provider: Optional[str]
    type: QuestionType
    chat_id: str
    conversation_timestamp: datetime.datetime


class AlertConversation(pydantic.BaseModel):
    """
    Represents an alert with it's respective conversation.
    """

    conversation: Conversation
    alert_id: str
    code_snippet: Optional[CodeSnippet]
    trigger_string: Optional[Union[str, dict]]
    trigger_type: str
    trigger_category: Optional[str]
    timestamp: datetime.datetime
