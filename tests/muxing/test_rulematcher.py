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
    "matcher, filenames_to_match, expected_bool_filenames",
    [
        (None, [], True),  # Empty filenames and no blob
        (None, ["main.py"], True),  # Empty blob should match
        (".py", ["main.py"], True),  # Extension match
        ("main.py", ["main.py"], True),  # Full name match
        (".py", ["main.py", "test.py"], True),  # Extension match
        ("main.py", ["main.py", "test.py"], True),  # Full name match
        ("main.py", ["test.py"], False),  # Full name no match
        (".js", ["main.py", "test.py"], False),  # Extension no match
        (".ts", ["main.tsx", "test.tsx"], False),  # Extension no match
    ],
)
@pytest.mark.parametrize(
    "is_fim_request, matcher_type, expected_bool_request",
    [
        (False, "fim", False),  # No match
        (True, "fim", True),  # Match
        (False, "chat", True),  # Match
        (True, "chat", False),  # No match
        (True, "catch_all", True),  # Match
        (False, "catch_all", True),  # Match
    ],
)
def test_file_matcher(
    matcher,
    filenames_to_match,
    expected_bool_filenames,
    is_fim_request,
    matcher_type,
    expected_bool_request,
):
    mux_rule = mux_models.MuxRule(
        provider_id="1",
        model="fake-gpt",
        matcher_type=matcher_type,
        matcher=matcher,
    )
    muxing_rule_matcher = rulematcher.RequestTypeAndFileMuxingRuleMatcher(
        mocked_route_openai, mux_rule
    )
    # We mock the _extract_request_filenames method to return a list of filenames
    # The logic to get the correct filenames from snippets is tested in /tests/extract_snippets
    muxing_rule_matcher._extract_request_filenames = MagicMock(return_value=filenames_to_match)
    mocked_thing_to_match = mux_models.ThingToMatchMux(
        body={},
        url_request_path="/chat/completions",
        is_fim_request=is_fim_request,
        client_type="generic",
    )
    assert (
        muxing_rule_matcher._is_request_type_match(mocked_thing_to_match.is_fim_request)
        is expected_bool_request
    )
    assert (
        muxing_rule_matcher._is_matcher_in_filenames(
            mocked_thing_to_match.client_type, mocked_thing_to_match.body
        )
        is expected_bool_filenames
    )
    assert muxing_rule_matcher.match(mocked_thing_to_match) is (
        expected_bool_request and expected_bool_filenames
    )
