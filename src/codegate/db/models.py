import datetime
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, BeforeValidator, ConfigDict, PlainSerializer, StringConstraints


class AlertSeverity(str, Enum):
    INFO = "info"
    CRITICAL = "critical"


class Alert(BaseModel):
    id: str
    prompt_id: str
    code_snippet: Optional[str]
    trigger_string: Optional[str]
    trigger_type: str
    trigger_category: AlertSeverity
    timestamp: datetime.datetime


class Output(BaseModel):
    id: Any
    prompt_id: Any
    timestamp: Any
    output: Any
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    input_cost: Optional[float] = None
    output_cost: Optional[float] = None


class Prompt(BaseModel):
    id: Any
    timestamp: Any
    provider: Optional[Any]
    request: Any
    type: Any
    workspace_id: Optional[str]


class TokenUsage(BaseModel):
    """
    TokenUsage it's not a table, it's a model to represent the token usage.
    The data is stored in the outputs table.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    input_cost: float = 0
    output_cost: float = 0

    @classmethod
    def from_dict(cls, usage_dict: Dict) -> "TokenUsage":
        return cls(
            input_tokens=usage_dict.get("prompt_tokens", 0) or usage_dict.get("input_tokens", 0),
            output_tokens=usage_dict.get("completion_tokens", 0)
            or usage_dict.get("output_tokens", 0),
            input_cost=0,
            output_cost=0,
        )

    @classmethod
    def from_db(
        cls,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
        input_cost: Optional[float],
        output_cost: Optional[float],
    ) -> "TokenUsage":
        return cls(
            input_tokens=0 if not input_tokens else input_tokens,
            output_tokens=0 if not output_tokens else output_tokens,
            input_cost=0 if not input_cost else input_cost,
            output_cost=0 if not output_cost else output_cost,
        )

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            input_cost=self.input_cost + other.input_cost,
            output_cost=self.output_cost + other.output_cost,
        )


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


class ProviderType(str, Enum):
    """
    Represents the different types of providers we support.
    """

    openai = "openai"
    anthropic = "anthropic"
    vllm = "vllm"
    ollama = "ollama"
    lm_studio = "lm_studio"
    llamacpp = "llamacpp"
    openrouter = "openrouter"


class IntermediatePromptWithOutputUsageAlerts(BaseModel):
    """
    An intermediate model to represent the result of a query
    for a prompt and related outputs, usage stats & alerts.
    """

    prompt_id: Any
    prompt_timestamp: Any
    provider: Optional[Any]
    request: Any
    type: Any
    output_id: Optional[Any]
    output: Optional[Any]
    output_timestamp: Optional[Any]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    input_cost: Optional[float]
    output_cost: Optional[float]
    alert_id: Optional[Any]
    code_snippet: Optional[Any]
    trigger_string: Optional[Any]
    trigger_type: Optional[Any]
    trigger_category: Optional[Any]
    alert_timestamp: Optional[Any]


class GetPromptWithOutputsRow(BaseModel):
    id: Any
    timestamp: Any
    provider: Optional[Any]
    request: Any
    type: Any
    output_id: Optional[Any]
    output: Optional[Any]
    output_timestamp: Optional[Any]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    input_cost: Optional[float]
    output_cost: Optional[float]
    alerts: List[Alert] = []


class WorkspaceWithSessionInfo(BaseModel):
    """Returns a workspace ID with an optional
    session ID. If the session ID is None, then
    the workspace is not active.
    """

    id: str
    name: WorkspaceNameStr
    session_id: Optional[str]


class WorkspaceWithModel(BaseModel):
    """Returns a workspace ID with model name"""

    id: str
    name: WorkspaceNameStr
    provider_model_name: str


class ActiveWorkspace(BaseModel):
    """Returns a full active workspace object with the
    with the session information.
    """

    id: str
    name: WorkspaceNameStr
    custom_instructions: Optional[str]
    session_id: str
    last_update: datetime.datetime


class ProviderEndpoint(BaseModel):
    id: str
    name: str
    description: str
    provider_type: str
    endpoint: str
    auth_type: str


class ProviderAuthMaterial(BaseModel):
    provider_endpoint_id: str
    auth_type: str
    auth_blob: str


class ProviderModel(BaseModel):
    provider_endpoint_id: str
    provider_endpoint_name: Optional[str] = None
    name: str


class MuxRule(BaseModel):
    id: str
    provider_endpoint_id: str
    provider_model_name: str
    workspace_id: str
    matcher_type: str
    matcher_blob: str
    priority: int
    created_at: Optional[datetime.datetime] = None
    updated_at: Optional[datetime.datetime] = None


# Pydantic doesn't support numpy arrays out of the box. Defining a custom type
# Reference: https://github.com/pydantic/pydantic/issues/7017
def nd_array_custom_before_validator(x):
    # custome before validation logic
    return x


def nd_array_custom_serializer(x):
    # custome serialization logic
    return str(x)


NdArray = Annotated[
    np.ndarray,
    BeforeValidator(nd_array_custom_before_validator),
    PlainSerializer(nd_array_custom_serializer, return_type=str),
]


class Persona(BaseModel):
    id: str
    name: str
    description: str


class PersonaEmbedding(Persona):
    description_embedding: NdArray  # sqlite-vec will handle numpy arrays directly

    # Part of the workaround to allow numpy arrays in pydantic models
    model_config = ConfigDict(arbitrary_types_allowed=True)


class PersonaDistance(Persona):
    distance: float
