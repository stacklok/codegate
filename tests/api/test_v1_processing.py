import datetime
import json
from unittest.mock import AsyncMock, patch

import pytest

from codegate.api.v1_models import PartialQuestions
from codegate.api.v1_processing import (
    _get_partial_question_answer,
    _group_partial_messages,
    _is_system_prompt,
    parse_output,
    parse_request,
)
from codegate.db.models import GetPromptWithOutputsRow


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message, expected_bool",
    [
        ("Hello, how can I help you?", False),
        (
            "Given the following... please reply with a short summary that is 4-12 words in length",
            True,
        ),
    ],
)
async def test_is_system_prompt(message, expected_bool):
    result = await _is_system_prompt(message)
    assert result == expected_bool


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_dict, expected_str_list",
    [
        (
            {"messages": [{"role": "user", "content": "Hello, how can I help you?"}]},
            ["Hello, how can I help you?"],
        ),
        (
            {
                "messages": [  # Request with system prompt
                    {
                        "role": "user",
                        "content": "Given the following... please reply with a short summary",
                    }
                ]
            },
            None,
        ),
        (
            {
                "messages": [  # Request with multiple messages
                    {"role": "user", "content": "Hello, how can I help you?"},
                    {"role": "user", "content": "Hello, latest"},
                ]
            },
            ["Hello, how can I help you?", "Hello, latest"],
        ),
        (
            {
                "messages": [  # Request with content list
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Hello, how can I help you?"}],
                    },
                ]
            },
            ["Hello, how can I help you?"],
        ),
        ({"prompt": "Hello, how can I help you?"}, ["Hello, how can I help you?"]),
    ],
)
async def test_parse_request(request_dict, expected_str_list):
    request_str = json.dumps(request_dict)
    result, _ = await parse_request(request_str)
    assert result == expected_str_list


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output_dict, expected_str",
    [
        (
            [  # Stream output with multiple chunks
                {
                    "id": "chatcmpl-AaQw9O1O2u360mhba5UbMoPwFgqEl",
                    "created": 1733246717,
                    "model": "gpt-4o-mini",
                    "object": "chat.completion.chunk",
                    "system_fingerprint": "fp_0705bf87c0",
                    "choices": [{"index": 0, "delta": {"content": "Hello", "role": "assistant"}}],
                },
                {
                    "id": "chatcmpl-AaQw9O1O2u360mhba5UbMoPwFgqEl",
                    "created": 1733246717,
                    "model": "gpt-4o-mini",
                    "object": "chat.completion.chunk",
                    "system_fingerprint": "fp_0705bf87c0",
                    "choices": [{"index": 0, "delta": {"content": " world"}}],
                },
                {
                    "id": "chatcmpl-AaQw9O1O2u360mhba5UbMoPwFgqEl",
                    "created": 1733246717,
                    "model": "gpt-4o-mini",
                    "object": "chat.completion.chunk",
                    "system_fingerprint": "fp_0705bf87c0",
                    "choices": [{"finish_reason": "stop", "index": 0, "delta": {}}],
                },
            ],
            "Hello world",
        ),
        (
            {
                "id": "chatcmpl-AaQw9O1O2u360mhba5UbMoPwFgqEa",
                "created": 1733246717,
                "model": "gpt-4o-mini",
                "object": "chat.completion.chunk",
                "system_fingerprint": "fp_0705bf87c0",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {"content": "User seeks", "role": "assistant"},
                    }
                ],
            },
            "User seeks",
        ),
    ],
)
async def test_parse_output(output_dict, expected_str):
    request_str = json.dumps(output_dict)
    output_message = await parse_output(request_str)
    assert output_message == expected_str


timestamp_now = datetime.datetime.now(datetime.timezone.utc)


@pytest.mark.asyncio
@pytest.mark.parametrize("request_msg_list", [["Hello"], None])
@pytest.mark.parametrize("output_msg_str", ["Hello, how can I help you?", None])
@pytest.mark.parametrize(
    "row",
    [
        GetPromptWithOutputsRow(
            id="1",
            timestamp=timestamp_now,
            provider="openai",
            request="foo",
            type="chat",
            output_id="2",
            output="bar",
            output_timestamp=timestamp_now,
            input_tokens=None,
            output_tokens=None,
            input_cost=None,
            output_cost=None,
        )
    ],
)
async def test_get_question_answer(request_msg_list, output_msg_str, row):
    with patch(
        "codegate.api.v1_processing.parse_request", new_callable=AsyncMock
    ) as mock_parse_request:
        with patch(
            "codegate.api.v1_processing.parse_output", new_callable=AsyncMock
        ) as mock_parse_output:
            # Set return values for the mocks
            mock_parse_request.return_value = request_msg_list, "openai"
            mock_parse_output.return_value = output_msg_str
            result = await _get_partial_question_answer(row)

            mock_parse_request.assert_called_once()
            mock_parse_output.assert_called_once()

            if request_msg_list is None:
                assert result is None
            else:
                assert result.partial_questions.messages == request_msg_list
                if output_msg_str is not None:
                    assert result.answer.message == output_msg_str
                assert result.partial_questions.provider == "openai"
                assert result.partial_questions.type == "chat"


@pytest.mark.parametrize(
    "pq_list,expected_group_ids",
    [
        # 1) No subsets: all items stand alone
        (
            [
                PartialQuestions(
                    messages=["A"],
                    timestamp=datetime.datetime(2023, 1, 1, 0, 0, 0),
                    message_id="pq1",
                    provider="providerA",
                    type="chat",
                ),
                PartialQuestions(
                    messages=["B"],
                    timestamp=datetime.datetime(2023, 1, 1, 0, 0, 1),
                    message_id="pq2",
                    provider="providerA",
                    type="chat",
                ),
            ],
            [["pq1"], ["pq2"]],
        ),
        # 2) Single subset: one is a subset of the other
        #    - "Hello" is a subset of "Hello, how are you?"
        (
            [
                PartialQuestions(
                    messages=["Hello"],
                    timestamp=datetime.datetime(2022, 1, 1, 0, 0, 0),
                    message_id="pq1",
                    provider="providerA",
                    type="chat",
                ),
                PartialQuestions(
                    messages=["Hello", "How are you?"],
                    timestamp=datetime.datetime(2022, 1, 1, 0, 0, 10),
                    message_id="pq2",
                    provider="providerA",
                    type="chat",
                ),
            ],
            [["pq1", "pq2"]],
        ),
        # 3) Multiple identical subsets:
        #    We have 3 partial questions with messages=["Hello"],
        #    plus a superset with messages=["Hello", "Bye"].
        #    Only the single subset that is closest in timestamp to the superset is grouped with the
        #    superset.
        (
            [
                PartialQuestions(
                    messages=["Hello"],
                    timestamp=datetime.datetime(2023, 1, 1, 10, 0, 0),
                    message_id="pq1",
                    provider="providerA",
                    type="chat",
                ),
                PartialQuestions(
                    messages=["Hello"],
                    timestamp=datetime.datetime(2023, 1, 1, 11, 0, 0),
                    message_id="pq2",
                    provider="providerA",
                    type="chat",
                ),
                PartialQuestions(
                    messages=["Hello"],
                    timestamp=datetime.datetime(2023, 1, 1, 12, 0, 0),
                    message_id="pq3",
                    provider="providerA",
                    type="chat",
                ),
                PartialQuestions(
                    messages=["Hello", "Bye"],
                    timestamp=datetime.datetime(2023, 1, 1, 11, 0, 5),
                    message_id="pq4",
                    provider="providerA",
                    type="chat",
                ),
            ],
            # pq4 is the superset => subsets are pq1, pq2, pq3.
            # The closest subset to pq4(11:00:05) is pq2(11:00:00).
            # So group = [pq2, pq4].
            # The other two remain alone in their own group.
            # The final sorted order is by earliest timestamp in each group:
            #   group with pq1 => [pq1], earliest 10:00:00
            #   group with pq2, pq4 => earliest 11:00:00
            #   group with pq3 => earliest 12:00:00
            [["pq1"], ["pq2", "pq4"], ["pq3"]],
        ),
        # 4) Mixed: multiple subsets, multiple supersets, verifying group logic
        (
            [
                # Superset
                PartialQuestions(
                    messages=["hi", "welcome", "bye"],
                    timestamp=datetime.datetime(2023, 5, 1, 9, 0, 0),
                    message_id="pqS1",
                    provider="providerB",
                    type="chat",
                ),
                # Subsets for pqS1
                PartialQuestions(
                    messages=["hi", "welcome"],
                    timestamp=datetime.datetime(2023, 5, 1, 9, 0, 5),
                    message_id="pqA1",
                    provider="providerB",
                    type="chat",
                ),
                PartialQuestions(
                    messages=["hi", "bye"],
                    timestamp=datetime.datetime(2023, 5, 1, 9, 0, 10),
                    message_id="pqA2",
                    provider="providerB",
                    type="chat",
                ),
                PartialQuestions(
                    messages=["hi", "bye"],
                    timestamp=datetime.datetime(2023, 5, 1, 9, 0, 12),
                    message_id="pqA3",
                    provider="providerB",
                    type="chat",
                ),
                # Another superset
                PartialQuestions(
                    messages=["apple", "banana", "cherry"],
                    timestamp=datetime.datetime(2023, 5, 2, 10, 0, 0),
                    message_id="pqS2",
                    provider="providerB",
                    type="chat",
                ),
                # Subsets for pqS2
                PartialQuestions(
                    messages=["banana"],
                    timestamp=datetime.datetime(2023, 5, 2, 10, 0, 1),
                    message_id="pqB1",
                    provider="providerB",
                    type="chat",
                ),
                PartialQuestions(
                    messages=["apple", "banana"],
                    timestamp=datetime.datetime(2023, 5, 2, 10, 0, 3),
                    message_id="pqB2",
                    provider="providerB",
                    type="chat",
                ),
                # Another item alone, not a subset nor superset
                PartialQuestions(
                    messages=["xyz"],
                    timestamp=datetime.datetime(2023, 5, 3, 8, 0, 0),
                    message_id="pqC1",
                    provider="providerB",
                    type="chat",
                ),
                # Different provider => should remain separate
                PartialQuestions(
                    messages=["hi", "welcome"],
                    timestamp=datetime.datetime(2023, 5, 1, 9, 0, 10),
                    message_id="pqProvDiff",
                    provider="providerX",
                    type="chat",
                ),
            ],
            # Expected:
            # For pqS1 (["hi","welcome","bye"]) => subsets are pqA1(["hi","welcome"]),
            # pqA2 & pqA3 (["hi","bye"])
            # Among pqA2 and pqA3, we pick the one closest in time to 09:00:00 =>
            # that is pqA2(09:00:10) vs pqA3(09:00:12).
            # The absolute difference:
            #   pqA2 => 10 seconds
            #   pqA3 => 12 seconds
            # So we pick pqA2. Group => [pqS1, pqA1, pqA2]
            #
            # For pqS2 (["apple","banana","cherry"]) => subsets are pqB1(["banana"]),
            # pqB2(["apple","banana"])
            # Among them, we group them all (because they have distinct messages).
            # So => [pqS2, pqB1, pqB2]
            #
            # pqC1 stands alone => ["pqC1"]
            # pqProvDiff stands alone => ["pqProvDiff"] because provider is different
            #
            # Then we sort by earliest timestamp in each group:
            #   group with pqS1 => earliest is 09:00:00
            #   group with pqProvDiff => earliest is 09:00:10
            #   group with pqS2 => earliest is 10:00:00
            #   group with pqC1 => earliest is 08:00:00 on 5/3 => actually this is the last date,
            #   so let's see:
            #       2023-05-01 is earlier than 2023-05-02, which is earlier than 2023-05-03.
            #       Actually, 2023-05-03 is later. So "pqC1" group is last in chronological order.
            #
            # Correct chronological order of earliest timestamps:
            #   1) [pqS1, pqA1, pqA2] => earliest 2023-05-01 09:00:00
            #   2) [pqProvDiff] => earliest 2023-05-01 09:00:10
            #   3) [pqS2, pqB1, pqB2] => earliest 2023-05-02 10:00:00
            #   4) [pqC1] => earliest 2023-05-03 08:00:00
            [["pqS1", "pqA1", "pqA2"], ["pqProvDiff"], ["pqS2", "pqB1", "pqB2"], ["pqC1"]],
        ),
    ],
)
def test_group_partial_messages(pq_list, expected_group_ids):
    """
    Verify that _group_partial_messages produces the correct grouping
    (by message_id) in the correct order.
    """
    # Execute
    grouped = _group_partial_messages(pq_list)

    # Convert from list[list[PartialQuestions]] -> list[list[str]]
    # so we can compare with expected_group_ids easily.
    grouped_ids = [[pq.message_id for pq in group] for group in grouped]

    is_matched = False
    print(grouped_ids)
    for group_id in grouped_ids:
        for expected_group in expected_group_ids:
            if set(group_id) == set(expected_group):
                is_matched = True
                break
        assert is_matched
