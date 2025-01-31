import copy
from abc import ABC, abstractmethod
from collections import UserDict
from threading import Lock, RLock
from typing import List, Optional

from codegate.db import models as db_models

_muxrules_sgtn = None

_singleton_lock = Lock()


def get_muxing_rules_registry():
    """Returns a singleton instance of the muxing rules registry."""

    global _muxrules_sgtn

    if _muxrules_sgtn is None:
        with _singleton_lock:
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


class MuxingRulesinWorkspaces(UserDict):
    """A thread safe dictionary to store the muxing rules in workspaces."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = RLock()
        self._active_workspace = ""

    def __getitem__(self, key: str) -> List[MuxingRuleMatcher]:
        with self._lock:
            # We return a copy so concurrent modifications don't affect the original
            return copy.deepcopy(super().__getitem__(key))

    def __setitem__(self, key: str, value: List[MuxingRuleMatcher]) -> None:
        with self._lock:
            super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        with self._lock:
            super().__delitem__(key)

    def set_active_workspace(self, workspace_name: str) -> None:
        """Set the active workspace."""
        self._active_workspace = workspace_name

    def get_match_for_active_workspace(self, thing_to_match) -> Optional[ModelRoute]:
        """Get the first match for the given thing_to_match."""

        # We iterate over all the rules and return the first match
        # Since we already do a deepcopy in __getitem__, we don't need to lock here
        try:
            for rule in self[self._active_workspace]:
                if rule.match(thing_to_match):
                    return rule.destination()
            return None
        except KeyError:
            raise RuntimeError("No rules found for the active workspace")
