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


class ProviderType(str, Enum):
    """
    Represents the different types of providers we support.
    """

    openai = "openai"
    anthropic = "anthropic"
    vllm = "vllm"


class ProviderAuthType(str, Enum):
    """
    Represents the different types of auth we support for providers.
    """

    # No auth required
    none = "none"
    # Whatever the user provides is passed through
    passthrough = "passthrough"
    # API key is required
    api_key = "api_key"


class ProviderEndpoint(pydantic.BaseModel):
    """
    Represents a provider's endpoint configuration. This
    allows us to persist the configuration for each provider,
    so we can use this for muxing messages.
    """

    id: int
    name: str
    description: str = ""
    provider_type: ProviderType
    endpoint: str
    auth_type: ProviderAuthType


class ModelByProvider(pydantic.BaseModel):
    """
    Represents a model supported by a provider.

    Note that these are auto-discovered by the provider.
    """

    name: str
    provider: str

    def __str__(self):
        return f"{self.provider}/{self.name}"


class MuxMatcherType(str, Enum):
    """
    Represents the different types of matchers we support.
    """

    # Match a regular expression for a file path
    # in the prompt. Note that if no file is found,
    # the prompt will be passed through.
    file_regex = "file_regex"

    # Always match this prompt
    catch_all = "catch_all"


class MuxRule(pydantic.BaseModel):
    """
    Represents a mux rule for a provider.
    """

    provider: str
    model: str
    # The type of matcher to use
    matcher_type: MuxMatcherType
    # The actual matcher to use. Note that
    # this depends on the matcher type.
    matcher: Optional[str]
