import asyncio
import json
from pathlib import Path
from typing import Dict, List, Tuple

import structlog

from codegate.db import models as db_models

logger = structlog.get_logger("codegate")


class TokenUsageError(Exception):
    pass


class TokenUsageParser:

    def __init__(self):
        current_dir = Path(__file__).parent
        filemodel_path = (
            current_dir.parent.parent.parent
            / "model_cost_data"
            / "model_prices_and_context_window.json"
        )
        with open(filemodel_path) as file:
            self.model_cost_mapping: Dict[str, Dict] = json.load(file)

        if not self.model_cost_mapping or not isinstance(self.model_cost_mapping, dict):
            raise TokenUsageError("Failed to load model prices and context window.")

    @property
    def mapping_model_to_model_cost(self) -> dict:
        """
        Maps the model name to the model cost name. The model cost name should
        exist in the URL above (model_prices_and_context_window.json).
        """
        return {
            "claude-3-5-sonnet-latest": "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-latest": "claude-3-5-haiku-20241022",
            "claude-3-opus-latest": "claude-3-opus-20240229",
        }

    async def _parse_usage_dict(self, usage_dict: dict) -> db_models.TokenUsage:
        return db_models.TokenUsage.from_dict(usage_dict)

    async def _get_model_cost(self, model: str) -> Tuple[float, float]:
        """
        Get the cost of the tokens for the model.
        """
        if not model:
            return 0, 0

        model_cost_name = self.mapping_model_to_model_cost.get(model, model)
        model_cost = self.model_cost_mapping.get(model_cost_name, {})
        # If the model is not found, return 0. Keys found in the URL above.
        input_cost_per_token = model_cost.get("input_cost_per_token", 0)
        output_cost_per_token = model_cost.get("output_cost_per_token", 0)

        return input_cost_per_token, output_cost_per_token

    async def _get_usage_from_output(self, output: db_models.Output) -> db_models.TokenUsage:
        """
        Parse from an output chunk the token usage.
        """
        try:
            output_dict = json.loads(output.output)
        except json.JSONDecodeError:
            logger.error(f"Failed to decode output: {output.output}")
            return db_models.TokenUsage()

        if not isinstance(output_dict, dict):
            logger.error(f"Output is not a dictionary: {output_dict}")
            return db_models.TokenUsage()

        token_usage = await self._parse_usage_dict(output_dict.get("usage", {}))
        input_token_cost, output_token_cost = await self._get_model_cost(
            output_dict.get("model", "")
        )

        token_usage.input_cost = token_usage.input_tokens * input_token_cost
        token_usage.output_cost = token_usage.output_tokens * output_token_cost

        return token_usage

    async def parse_outputs(self, outputs: List[db_models.Output]) -> db_models.TokenUsage:
        """
        Parse the token usage from the output chunks.
        """
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(self._get_usage_from_output(output)) for output in outputs]

        token_usage = db_models.TokenUsage()
        for task in tasks:
            token_usage += task.result()
        return token_usage
