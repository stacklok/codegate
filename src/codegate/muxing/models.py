from enum import Enum
from typing import Optional

import pydantic

from codegate.clients.clients import ClientType
from codegate.db.models import MuxRule as DbMuxRule


class MuxMatcherType(str, Enum):
    """
    Represents the different types of matchers we support.
    """

    # Always match this prompt
    catch_all = "catch_all"
    # Match based on the filename. It will match if there is a filename
    # in the request that matches the matcher either extension or full name (*.py or main.py)
    filename_match = "filename_match"
    # Match based on the request type. It will match if the request type
    # matches the matcher (e.g. FIM or chat)
    request_type_match = "request_type_match"


class MuxRule(pydantic.BaseModel):
    """
    Represents a mux rule for a provider.
    """

    # Used for exportable workspaces
    provider_name: Optional[str] = None
    provider_id: str
    model: str
    # The type of matcher to use
    matcher_type: MuxMatcherType
    # The actual matcher to use. Note that
    # this depends on the matcher type.
    matcher: Optional[str] = None

    @classmethod
    def try_from_db_model(cls, db_model: DbMuxRule) -> "MuxRule":
        try:
            return cls(
                provider_name=db_model.provider_endpoint_name,
                provider_id=db_model.provider_endpoint_id,
                model=db_model.provider_model_name,
                matcher_type=MuxMatcherType(db_model.matcher_type),
                matcher=db_model.matcher_blob,
            )
        except Exception as e:
            raise ValueError(f"Error converting from DbMuxRule: {e}")


class ThingToMatchMux(pydantic.BaseModel):
    """
    Represents the fields we can use to match a mux rule.
    """

    body: dict
    url_request_path: str
    is_fim_request: bool
    client_type: ClientType
