from typing import List

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
