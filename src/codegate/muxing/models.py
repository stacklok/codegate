from enum import Enum
from typing import Optional, Self

import pydantic

from codegate.clients.clients import ClientType
from codegate.db.models import MuxRule as DBMuxRule


class MuxMatcherType(str, Enum):
    """
    Represents the different types of matchers we support.

    The 3 rules present match filenames and request types. They're used in conjunction with the
    matcher field in the MuxRule model.
    E.g.
    - catch_all-> Always match
    - filename_match and match: requests.py -> Match the request if the filename is requests.py
    - fim_filename and match: main.py -> Match the request if the request type is fim
    and the filename is main.py

    NOTE: Removing or updating fields from this enum will require a migration.
    Adding new fields is safe.
    """

    # Always match this prompt
    catch_all = "catch_all"
    # Match based on the filename. It will match if there is a filename
    # in the request that matches the matcher either extension or full name (*.py or main.py)
    filename_match = "filename_match"
    # Match based on fim request type. It will match if the request type is fim
    fim_filename = "fim_filename"
    # Match based on chat request type. It will match if the request type is chat
    chat_filename = "chat_filename"
    # Match the request content to the persona description
    persona_description = "persona_description"


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
    def from_db_mux_rule(cls, db_mux_rule: DBMuxRule) -> Self:
        """
        Convert a DBMuxRule to a MuxRule.
        """
        return MuxRule(
            provider_id=db_mux_rule.id,
            model=db_mux_rule.provider_model_name,
            matcher_type=db_mux_rule.matcher_type,
            matcher=db_mux_rule.matcher_blob,
        )


class ThingToMatchMux(pydantic.BaseModel):
    """
    Represents the fields we can use to match a mux rule.
    """

    body: dict
    url_request_path: str
    is_fim_request: bool
    client_type: ClientType
