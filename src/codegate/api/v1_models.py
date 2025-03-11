import datetime
import json
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import pydantic

import codegate.muxing.models as mux_models
from codegate.db import models as db_models
from codegate.extract_snippets.message_extractor import CodeSnippet
from codegate.providers.base import BaseProvider
from codegate.providers.registry import ProviderRegistry


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


class WorkspaceConfig(pydantic.BaseModel):
    custom_instructions: str

    muxing_rules: List[mux_models.MuxRule]


class FullWorkspace(pydantic.BaseModel):
    name: str

    config: Optional[WorkspaceConfig] = None


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


class TokenUsageByModel(pydantic.BaseModel):
    """
    Represents the tokens used by a model.
    """

    provider_type: db_models.ProviderType
    model: str
    token_usage: db_models.TokenUsage


class TokenUsageAggregate(pydantic.BaseModel):
    """
    Represents the tokens used. Includes the information of the tokens used by model.
    `used_tokens` are the total tokens used in the `tokens_by_model` list.
    """

    tokens_by_model: Dict[str, TokenUsageByModel]
    token_usage: db_models.TokenUsage

    def add_model_token_usage(self, model_token_usage: TokenUsageByModel) -> None:
        # Copilot doesn't have a model name and we cannot obtain the tokens used. Skip it.
        if model_token_usage.model == "":
            return

        # Skip if the model has not used any tokens.
        if (
            model_token_usage.token_usage.input_tokens == 0
            and model_token_usage.token_usage.output_tokens == 0
        ):
            return

        if model_token_usage.model in self.tokens_by_model:
            self.tokens_by_model[
                model_token_usage.model
            ].token_usage += model_token_usage.token_usage
        else:
            self.tokens_by_model[model_token_usage.model] = model_token_usage
        self.token_usage += model_token_usage.token_usage


class Alert(pydantic.BaseModel):
    """
    Represents an alert.
    """

    @staticmethod
    def from_db_model(db_model: db_models.Alert) -> "Alert":
        try:
            trigger_string = json.loads(db_model.trigger_string)
        except Exception:
            trigger_string = db_model.trigger_string
        snippet = json.loads(db_model.code_snippet) if db_model.code_snippet else None
        return Alert(
            id=db_model.id,
            prompt_id=db_model.prompt_id,
            code_snippet=snippet,
            trigger_string=trigger_string,
            trigger_type=db_model.trigger_type,
            trigger_category=db_model.trigger_category,
            timestamp=db_model.timestamp,
        )

    id: str
    prompt_id: str
    code_snippet: Optional[CodeSnippet]
    trigger_string: Optional[Union[str, dict]]
    trigger_type: str
    trigger_category: db_models.AlertSeverity
    timestamp: datetime.datetime


class AlertSummary(pydantic.BaseModel):
    """
    Represents a set of summary alerts
    """

    malicious_packages: int
    pii: int
    secrets: int
    total_alerts: int


class PartialQuestionAnswer(pydantic.BaseModel):
    """
    Represents a partial conversation.
    """

    partial_questions: PartialQuestions
    answer: Optional[ChatMessage]
    model_token_usage: TokenUsageByModel


class Conversation(pydantic.BaseModel):
    """
    Represents a conversation.
    """

    question_answers: List[QuestionAnswer]
    provider: Optional[str]
    type: QuestionType
    chat_id: str
    conversation_timestamp: datetime.datetime
    token_usage_agg: Optional[TokenUsageAggregate]
    alerts: Optional[List[Alert]] = []


class ConversationSummary(pydantic.BaseModel):
    """
    Represents a conversation summary.
    """

    chat_id: str
    prompt: ChatMessage
    alerts_summary: AlertSummary
    token_usage_agg: Optional[TokenUsageAggregate]
    provider: Optional[str]
    type: QuestionType
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

    #  This will be set on creation
    id: Optional[str] = ""
    name: str
    description: str = ""
    provider_type: db_models.ProviderType
    endpoint: str = ""  # Some providers have defaults we can leverage
    auth_type: ProviderAuthType = ProviderAuthType.none

    @staticmethod
    def from_db_model(db_model: db_models.ProviderEndpoint) -> "ProviderEndpoint":
        auth_type = (
            ProviderAuthType.none
            if not db_model.auth_type
            else ProviderAuthType(db_model.auth_type)
        )
        return ProviderEndpoint(
            id=db_model.id,
            name=db_model.name,
            description=db_model.description,
            provider_type=db_model.provider_type,
            endpoint=db_model.endpoint,
            auth_type=auth_type,
        )

    def to_db_model(self) -> db_models.ProviderEndpoint:
        return db_models.ProviderEndpoint(
            id=self.id,
            name=self.name,
            description=self.description,
            provider_type=self.provider_type,
            endpoint=self.endpoint,
            auth_type=self.auth_type,
        )

    def get_from_registry(self, registry: ProviderRegistry) -> Optional[BaseProvider]:
        return registry.get_provider(self.provider_type)


class AddProviderEndpointRequest(ProviderEndpoint):
    """
    Represents a request to add a provider endpoint.
    """

    api_key: Optional[str] = None


class ConfigureAuthMaterial(pydantic.BaseModel):
    """
    Represents a request to configure auth material for a provider.
    """

    auth_type: ProviderAuthType
    api_key: Optional[str] = None


class ModelByProvider(pydantic.BaseModel):
    """
    Represents a model supported by a provider.

    Note that these are auto-discovered by the provider.
    """

    name: str
    provider_type: db_models.ProviderType
    provider_name: str

    def __str__(self):
        return f"{self.provider_name} / {self.name}"


class PersonaRequest(pydantic.BaseModel):
    """
    Model for creating a new Persona.
    """

    name: str
    description: str


class PersonaUpdateRequest(pydantic.BaseModel):
    """
    Model for updating a Persona.
    """

    new_name: str
    new_description: str


class PaginatedMessagesResponse(pydantic.BaseModel):
    data: List[ConversationSummary]
    limit: int
    offset: int
    total: int
