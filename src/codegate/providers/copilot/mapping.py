from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, HttpUrl
from pydantic_settings import BaseSettings


class CopilotProxyRoute(BaseModel):
    """Pydantic model for proxy route validation"""

    path: str
    target: HttpUrl


class CoPilotMappings(BaseSettings):

    # Proxy routes configuration
    # This code is to ensure that incoming proxy requests are routed to the correct target
    PROXY_ROUTES: List[tuple[str, str]] = [
        ("/github/login", "https://github.com/login"),
        ("/api/github/user", "https://api.github.com"),
        ("/api/github/copilot", "https://api.github.com/copilot_internal"),
        ("/copilot/telemetry", "https://copilot-telemetry.githubusercontent.com"),
        ("/vscode/ab", "https://default.exp-tas.com"),
        ("/copilot/proxy", "https://copilot-proxy.githubusercontent.com"),
        ("/origin-tracker", "https://origin-tracker.githubusercontent.com"),
        ("/copilot/suggestions", "https://githubcopilot.com"),
        ("/copilot_internal/user", "https://api.github.com"),
        ("/copilot_internal/v2/token", "https://api.github.com"),
        ("/telemetry", "https://copilot-telemetry.githubusercontent.com"),
        ("/", "https://github.com"),
        ("/login/oauth/access_token", "https://github.com/login/oauth/access_token"),
        ("/api/copilot", "https://api.github.com/copilot_internal"),
        ("/api/copilot_internal", "https://api.github.com/copilot_internal"),
        ("/v1/completions", "https://copilot-proxy.githubusercontent.com/v1/completions"),
        ("/v1", "https://copilot-proxy.githubusercontent.com/v1"),
    ]


# Create settings instance
mappings = CoPilotMappings()

# Convert routes to validated ProxyRoute objects
VALIDATED_ROUTES: List[CopilotProxyRoute] = [
    CopilotProxyRoute(path=path, target=target) for path, target in mappings.PROXY_ROUTES
]


class PipelineType(Enum):
    FIM = "fim"
    CHAT = "chat"


@dataclass
class PipelineRoute:
    path: str
    pipeline_type: PipelineType
    target_url: Optional[str] = None


PIPELINE_ROUTES = [
    PipelineRoute(
        path="v1/chat/completions",
        # target_url="https://api.openai.com/v1/chat/completions",
        pipeline_type=PipelineType.CHAT,
    ),
    PipelineRoute(path="v1/engines/copilot-codex/completions", pipeline_type=PipelineType.FIM),
    PipelineRoute(path="chat/completions", pipeline_type=PipelineType.CHAT),
]
