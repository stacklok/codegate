import copy
from abc import ABC, abstractmethod
from asyncio import Lock
from typing import List, Optional

from codegate.db import models as db_models

_muxrules_sgtn = None

_singleton_lock = Lock()


async def get_muxing_rules_registry():
    """Returns a singleton instance of the muxing rules registry."""

    global _muxrules_sgtn

    if _muxrules_sgtn is None:
        async with _singleton_lock:
            if _muxrules_sgtn is None:
                _muxrules_sgtn = MuxingRulesinWorkspaces()

    return _muxrules_sgtn


class ModelRoute:
    """A route for a model."""

    def __init__(
        self,
        model: db_models.ProviderModel,
        endpoint: db_models.ProviderEndpoint,
        auth_material: db_models.ProviderAuthMaterial,
    ):
        self.model = model
        self.endpoint = endpoint
        self.auth_material = auth_material


class MuxingRuleMatcher(ABC):
    """Base class for matching muxing rules."""

    def __init__(self, route: ModelRoute):
        self._route = route

    @abstractmethod
    def match(self, thing_to_match) -> bool:
        """Return True if the rule matches the thing_to_match."""
        pass

    def destination(self) -> ModelRoute:
        """Return the destination of the rule."""

        return self._route


class MuxingMatcherFactory:
    """Factory for creating muxing matchers."""

    @staticmethod
    def create(mux_rule: db_models.MuxRule, route: ModelRoute) -> MuxingRuleMatcher:
        """Create a muxing matcher for the given endpoint and model."""

        factory = {
            "catch_all": CatchAllMuxingRuleMatcher,
        }

        try:
            return factory[mux_rule.matcher_type](route)
        except KeyError:
            raise ValueError(f"Unknown matcher type: {mux_rule.matcher_type}")


class CatchAllMuxingRuleMatcher(MuxingRuleMatcher):
    """A catch all muxing rule matcher."""

    def match(self, thing_to_match) -> bool:
        return True


class MuxingRulesinWorkspaces:
    """A thread safe dictionary to store the muxing rules in workspaces."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = Lock()
        self._active_workspace = ""
        self._ws_rules = {}

    async def get_ws_rules(self, workspace_name: str) -> List[MuxingRuleMatcher]:
        """Get the rules for the given workspace."""
        async with self._lock:
            return copy.deepcopy(self._ws_rules.get(workspace_name, []))

    async def set_ws_rules(self, workspace_name: str, rules: List[MuxingRuleMatcher]) -> None:
        """Set the rules for the given workspace."""
        async with self._lock:
            self._ws_rules[workspace_name] = rules

    async def delete_ws_rules(self, workspace_name: str) -> None:
        """Delete the rules for the given workspace."""
        async with self._lock:
            del self._ws_rules[workspace_name]

    async def set_active_workspace(self, workspace_name: str) -> None:
        """Set the active workspace."""
        self._active_workspace = workspace_name

    async def get_registries(self) -> List[str]:
        """Get the list of workspaces."""
        async with self._lock:
            return list(self._ws_rules.keys())

    async def get_match_for_active_workspace(self, thing_to_match) -> Optional[ModelRoute]:
        """Get the first match for the given thing_to_match."""

        # We iterate over all the rules and return the first match
        # Since we already do a deepcopy in __getitem__, we don't need to lock here
        try:
            rules = await self.get_ws_rules(self._active_workspace)
            for rule in rules:
                if rule.match(thing_to_match):
                    return rule.destination()
            return None
        except KeyError:
            raise RuntimeError("No rules found for the active workspace")
