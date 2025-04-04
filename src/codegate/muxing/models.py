from enum import Enum
from typing import Any, Optional, Self

import pydantic

from codegate.clients.clients import ClientType
from codegate.db.models import MuxRule as DBMuxRule
from codegate.db.models import ProviderEndpoint as DBProviderEndpoint
from codegate.db.models import ProviderType


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


class MuxRule(pydantic.BaseModel):
    """
    Represents a mux rule for a provider.
    """

    provider_name: str
    provider_type: ProviderType
    model: str
    # The type of matcher to use
    matcher_type: MuxMatcherType
    # The actual matcher to use. Note that
    # this depends on the matcher type.
    matcher: Optional[str] = None

    @classmethod
    def from_db_models(
        cls, db_mux_rule: DBMuxRule, db_provider_endpoint: DBProviderEndpoint
    ) -> Self:
        """
        Convert a DBMuxRule and DBProviderEndpoint to a MuxRule.
        """
        return cls(
            provider_name=db_provider_endpoint.name,
            provider_type=db_provider_endpoint.provider_type,
            model=db_mux_rule.provider_model_name,
            matcher_type=MuxMatcherType(db_mux_rule.matcher_type),
            matcher=db_mux_rule.matcher_blob,
        )

    @classmethod
    def from_mux_rule_with_provider_id(cls, rule: "MuxRuleWithProviderId") -> Self:
        """
        Convert a MuxRuleWithProviderId to a MuxRule.
        """
        return cls(
            provider_name=rule.provider_name,
            provider_type=rule.provider_type,
            model=rule.model,
            matcher_type=rule.matcher_type,
            matcher=rule.matcher,
        )


class MuxRuleWithProviderId(MuxRule):
    """
    Represents a mux rule for a provider with provider ID.
    Used internally for referring to a mux rule.
    """

    provider_id: str

    @classmethod
    def from_db_models(
        cls, db_mux_rule: DBMuxRule, db_provider_endpoint: DBProviderEndpoint
    ) -> Self:
        """
        Convert a DBMuxRule and DBProviderEndpoint to a MuxRuleWithProviderId.
        """
        return cls(
            **MuxRule.from_db_models(db_mux_rule, db_provider_endpoint).model_dump(),
            provider_id=db_mux_rule.provider_endpoint_id,
        )


class ThingToMatchMux(pydantic.BaseModel):
    """
    Represents the fields we can use to match a mux rule.
    """

    body: Any
    url_request_path: str
    is_fim_request: bool
    client_type: ClientType
