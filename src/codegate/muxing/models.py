from enum import Enum
from typing import Optional

import pydantic


class MuxMatcherType(str, Enum):
    """
    Represents the different types of matchers we support.
    """

    # Always match this prompt
    catch_all = "catch_all"


class MuxRule(pydantic.BaseModel):
    """
    Represents a mux rule for a provider.
    """

    provider_id: str
    model: str
    # The type of matcher to use
    matcher_type: MuxMatcherType
    # The actual matcher to use. Note that
    # this depends on the matcher type.
    matcher: Optional[str] = None
