from typing import List, Optional

import structlog

from codegate.config import Config
from codegate.llm_utils.llmclient import LLMClient

logger = structlog.get_logger("codegate")


class PackageExtractor:
    """
    Utility class to extract package names from code or queries.
    """

    @staticmethod
    async def extract_packages(
        content: str,
        provider: str,
        model: str = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> List[str]:
        """Extract package names from the given content."""
        system_prompt = Config.get_config().prompts.lookup_packages

        result = await LLMClient.complete(
            content=content,
            system_prompt=system_prompt,
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )

        # Handle both formats: {"packages": [...]} and direct list [...]
        packages = result if isinstance(result, list) else result.get("packages", [])
        logger.info(f"Extracted packages: {packages}")
        return packages
