import copy

from codegate.db import models as db_models


class BodyAdapter:
    """
    Map the body between OpenAI and Anthropic.

    TODO: This are really knaive implementations. We should replace them with more robust ones.
    """

    def _from_openai_to_antrhopic(self, openai_body: dict) -> dict:
        """Map the OpenAI body to the Anthropic body."""
        new_body = copy.deepcopy(openai_body)
        messages = new_body.get("messages", [])
        system_prompt = None
        system_msg_idx = None
        if messages:
            for i_msg, msg in enumerate(messages):
                if msg.get("role", "") == "system":
                    system_prompt = msg.get("content")
                    system_msg_idx = i_msg
                    break
        if system_prompt:
            new_body["system"] = system_prompt
        if system_msg_idx is not None:
            del messages[system_msg_idx]
        return new_body

    def _from_anthropic_to_openai(self, anthropic_body: dict) -> dict:
        """Map the Anthropic body to the OpenAI body."""
        new_body = copy.deepcopy(anthropic_body)
        system_prompt = anthropic_body.get("system")
        messages = new_body.get("messages", [])
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        if "system" in new_body:
            del new_body["system"]
        return new_body

    def _identify_provider(self, data: dict) -> db_models.ProviderType:
        """Identify the request provider."""
        if "system" in data:
            return db_models.ProviderType.anthropic
        else:
            return db_models.ProviderType.openai

    def map_body_to_dest(self, dest_prov: db_models.ProviderType, data: dict) -> dict:
        """
        Map the body to the destination provider.

        We only need to transform the body if the destination or origin provider is Anthropic.
        """
        origin_prov = self._identify_provider(data)
        if dest_prov == db_models.ProviderType.anthropic:
            if origin_prov == db_models.ProviderType.anthropic:
                return data
            else:
                return self._from_openai_to_antrhopic(data)
        else:
            if origin_prov == db_models.ProviderType.anthropic:
                return self._from_anthropic_to_openai(data)
            else:
                return data
