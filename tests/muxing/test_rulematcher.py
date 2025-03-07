from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

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


@pytest.mark.asyncio
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
async def test_catch_all(matcher_blob, thing_to_match):
    muxing_rule_matcher = rulematcher.CatchAllMuxingRuleMatcher(mocked_route_openai, matcher_blob)
    # It should always match
    assert await muxing_rule_matcher.match(thing_to_match) is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "matcher, filenames_to_match, expected_bool",
    [
        (None, [], True),  # Empty filenames and no blob
        (None, ["main.py"], True),  # Empty blob should match
        ("*.py", ["main.py"], True),  # Extension match
        ("main.py", ["main.py"], True),  # Full name match
        ("*.py", ["main.py", "test.py"], True),  # Extension match
        ("main.py", ["main.py", "test.py"], True),  # Full name match
        ("main.py", ["test.py"], False),  # Full name no match
        ("*.js", ["main.py", "test.py"], False),  # Extension no match
        ("*.ts", ["main.tsx", "test.tsx"], False),  # Extension no match
    ],
)
async def test_file_matcher(
    matcher,
    filenames_to_match,
    expected_bool,
):
    mux_rule = mux_models.MuxRule(
        provider_id="1",
        model="fake-gpt",
        matcher_type="filename_match",
        matcher=matcher,
    )
    muxing_rule_matcher = rulematcher.FileMuxingRuleMatcher(mocked_route_openai, mux_rule)
    # We mock the _extract_request_filenames method to return a list of filenames
    # The logic to get the correct filenames from snippets is tested in /tests/extract_snippets
    muxing_rule_matcher._extract_request_filenames = MagicMock(return_value=filenames_to_match)
    mocked_thing_to_match = mux_models.ThingToMatchMux(
        body={},
        url_request_path="/chat/completions",
        is_fim_request=False,
        client_type="generic",
    )
    assert await muxing_rule_matcher.match(mocked_thing_to_match) is expected_bool


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "matcher, filenames_to_match, expected_bool_filenames",
    [
        (None, [], True),  # Empty filenames and no blob
        (None, ["main.py"], True),  # Empty blob should match
        ("*.py", ["main.py"], True),  # Extension match
        ("main.py", ["main.py"], True),  # Full name match
        ("*.py", ["main.py", "test.py"], True),  # Extension match
        ("main.py", ["main.py", "test.py"], True),  # Full name match
        ("main.py", ["test.py"], False),  # Full name no match
        ("*.js", ["main.py", "test.py"], False),  # Extension no match
        ("*.ts", ["main.tsx", "test.tsx"], False),  # Extension no match
    ],
)
@pytest.mark.parametrize(
    "is_fim_request, matcher_type, expected_bool_request",
    [
        (False, "fim_filename", False),  # No match
        (True, "fim_filename", True),  # Match
        (False, "chat_filename", True),  # Match
        (True, "chat_filename", False),  # No match
    ],
)
async def test_request_file_matcher(
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
    assert await muxing_rule_matcher.match(mocked_thing_to_match) is (
        expected_bool_request and expected_bool_filenames
    )


# We mock PersonaManager because it's tested in /tests/persona/test_manager.py
MOCK_PERSONA_MANAGER = AsyncMock()
MOCK_PERSONA_MANAGER.check_persona_match.return_value = True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body, expected_queries",
    [
        ({"messages": [{"role": "system", "content": "Youre helpful"}]}, []),
        ({"messages": [{"role": "user", "content": "hello"}]}, ["hello"]),
        (
            {"messages": [{"role": "user", "content": [{"type": "text", "text": "hello_dict"}]}]},
            ["hello_dict"],
        ),
    ],
)
async def test_user_msgs_persona_desc_matcher(body: Dict, expected_queries: List[str]):
    mux_rule = mux_models.MuxRule(
        provider_id="1",
        model="fake-gpt",
        matcher_type="persona_description",
        matcher="foo_persona",
    )
    muxing_rule_matcher = rulematcher.UserMsgsPersonaDescMuxMatcher(mocked_route_openai, mux_rule)

    mocked_thing_to_match = mux_models.ThingToMatchMux(
        body=body,
        url_request_path="/chat/completions",
        is_fim_request=False,
        client_type="generic",
    )

    resulting_queries = muxing_rule_matcher._get_queries_for_persona_match(body)
    assert set(resulting_queries) == set(expected_queries)

    with patch("codegate.muxing.rulematcher.PersonaManager", return_value=MOCK_PERSONA_MANAGER):
        result = await muxing_rule_matcher.match(mocked_thing_to_match)

    if expected_queries:
        assert result is True
    else:
        assert result is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body, expected_queries",
    [
        ({"messages": [{"role": "system", "content": "Youre helpful"}]}, ["Youre helpful"]),
        ({"messages": [{"role": "user", "content": "hello"}]}, []),
        (
            {
                "messages": [
                    {"role": "system", "content": "Youre helpful"},
                    {"role": "user", "content": "hello"},
                ]
            },
            ["Youre helpful"],
        ),
        (
            {"messages": [{"role": "user", "content": "hello"}], "system": "Anthropic system"},
            ["Anthropic system"],
        ),
    ],
)
async def test_sys_prompt_persona_desc_matcher(body: Dict, expected_queries: List[str]):
    mux_rule = mux_models.MuxRule(
        provider_id="1",
        model="fake-gpt",
        matcher_type="sys_prompt_persona_desc",
        matcher="foo_persona",
    )
    muxing_rule_matcher = rulematcher.SysPromptPersonaDescMuxMatcher(mocked_route_openai, mux_rule)

    mocked_thing_to_match = mux_models.ThingToMatchMux(
        body=body,
        url_request_path="/chat/completions",
        is_fim_request=False,
        client_type="generic",
    )

    resulting_queries = muxing_rule_matcher._get_queries_for_persona_match(body)
    assert set(resulting_queries) == set(expected_queries)

    with patch("codegate.muxing.rulematcher.PersonaManager", return_value=MOCK_PERSONA_MANAGER):
        result = await muxing_rule_matcher.match(mocked_thing_to_match)

    if expected_queries:
        assert result is True
    else:
        assert result is False


@pytest.mark.parametrize(
    "matcher_type, expected_class",
    [
        (mux_models.MuxMatcherType.catch_all, rulematcher.CatchAllMuxingRuleMatcher),
        (mux_models.MuxMatcherType.filename_match, rulematcher.FileMuxingRuleMatcher),
        (mux_models.MuxMatcherType.fim_filename, rulematcher.RequestTypeAndFileMuxingRuleMatcher),
        (mux_models.MuxMatcherType.chat_filename, rulematcher.RequestTypeAndFileMuxingRuleMatcher),
        (
            mux_models.MuxMatcherType.persona_description,
            rulematcher.UserMsgsPersonaDescMuxMatcher,
        ),
        (
            mux_models.MuxMatcherType.sys_prompt_persona_desc,
            rulematcher.SysPromptPersonaDescMuxMatcher,
        ),
        ("invalid_matcher", None),
    ],
)
def test_muxing_matcher_factory(matcher_type, expected_class):
    mux_rule = db_models.MuxRule(
        id="1",
        provider_endpoint_id="1",
        provider_model_name="fake-gpt",
        workspace_id="1",
        matcher_type=matcher_type,
        matcher_blob="fake-matcher",
        priority=1,
    )
    if expected_class:
        assert isinstance(
            rulematcher.MuxingMatcherFactory.create(mux_rule, mocked_route_openai), expected_class
        )
    else:
        with pytest.raises(ValueError):
            rulematcher.MuxingMatcherFactory.create(mux_rule, mocked_route_openai)
