from typing import Any, Dict

from ._legacy_models import LegacyCompletionRequest


class CopilotCompletionRequest(LegacyCompletionRequest):
    nwo: str | None = None
    extra: Dict[str, Any] | None = None
