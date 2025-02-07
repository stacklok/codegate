from unittest.mock import MagicMock

import pytest

from codegate.db import models as db_models
from codegate.muxing import models as mux_models
from codegate.muxing import rulematcher

mocked_route_openai = rulematcher.ModelRoute(
    db_models.ProviderModel(
        provider_endpoint_id="1", provider_endpoint_name="fake-openai", name="fake-gpt"
    ),
    db_models.ProviderEndpoint(
        id="1",
        name="fake-openai",
        description="fake-openai",
        provider_type="fake-openai",
        endpoint="http://localhost/openai",
        auth_type="api_key",
    ),
    db_models.ProviderAuthMaterial(
        provider_endpoint_id="1", auth_type="api_key", auth_blob="fake-api-key"
    ),
)


@pytest.mark.parametrize(
    "matcher_blob, thing_to_match",
    [
        (None, None),
        ("fake-matcher-blob", None),
        (
            "fake-matcher-blob",
            mux_models.ThingToMatchMux(
                body={},
                url_request_path="/chat/completions",
                is_fim_request=False,
                client_type="generic",
            ),
        ),
    ],
)
def test_catch_all(matcher_blob, thing_to_match):
    muxing_rule_matcher = rulematcher.CatchAllMuxingRuleMatcher(mocked_route_openai, matcher_blob)
    # It should always match
    assert muxing_rule_matcher.match(thing_to_match) is True


@pytest.mark.parametrize(
    "matcher_blob, filenames_to_match, expected_bool",
    [
        (None, [], False),  # Empty filenames and no blob
        (None, ["main.py"], False),  # Empty blob
        (".py", ["main.py"], True),  # Extension match
        ("main.py", ["main.py"], True),  # Full name match
        (".py", ["main.py", "test.py"], True),  # Extension match
        ("main.py", ["main.py", "test.py"], True),  # Full name match
        ("main.py", ["test.py"], False),  # Full name no match
        (".js", ["main.py", "test.py"], False),  # Extension no match
    ],
)
def test_file_matcher(matcher_blob, filenames_to_match, expected_bool):
    muxing_rule_matcher = rulematcher.FileMuxingRuleMatcher(mocked_route_openai, matcher_blob)
    # We mock the _extract_request_filenames method to return a list of filenames
    # The logic to get the correct filenames from snippets is tested in /tests/extract_snippets
    muxing_rule_matcher._extract_request_filenames = MagicMock(return_value=filenames_to_match)
    mocked_thing_to_match = mux_models.ThingToMatchMux(
        body={},
        url_request_path="/chat/completions",
        is_fim_request=False,
        client_type="generic",
    )
    assert muxing_rule_matcher.match(mocked_thing_to_match) == expected_bool


@pytest.mark.parametrize(
    "matcher_blob, thing_to_match, expected_bool",
    [
        (None, None, False),  # Empty blob
        (
            "fim",
            mux_models.ThingToMatchMux(
                body={},
                url_request_path="/chat/completions",
                is_fim_request=False,
                client_type="generic",
            ),
            False,
        ),  # No match
        (
            "fim",
            mux_models.ThingToMatchMux(
                body={},
                url_request_path="/chat/completions",
                is_fim_request=True,
                client_type="generic",
            ),
            True,
        ),  # Match
        (
            "chat",
            mux_models.ThingToMatchMux(
                body={},
                url_request_path="/chat/completions",
                is_fim_request=True,
                client_type="generic",
            ),
            False,
        ),  # No match
        (
            "chat",
            mux_models.ThingToMatchMux(
                body={},
                url_request_path="/chat/completions",
                is_fim_request=False,
                client_type="generic",
            ),
            True,
        ),  # Match
    ],
)
def test_request_type(matcher_blob, thing_to_match, expected_bool):
    muxing_rule_matcher = rulematcher.RequestTypeMuxingRuleMatcher(
        mocked_route_openai, matcher_blob
    )
    assert muxing_rule_matcher.match(thing_to_match) == expected_bool
