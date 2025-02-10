import hashlib
import json
from datetime import datetime, timedelta, timezone
from unittest import mock

import pytest

from codegate.db.fim_cache import CachedFim, FimCache
from codegate.db.models import Alert, AlertSeverity
from codegate.pipeline.base import PipelineContext

fim_python_message = """
# Path: folder/testing_file.py
# another_folder/another_file.py

This is a test message
"""

fim_js_message = """<|fim_prefix|>// Path: Untitled.txt
// expected_filepath
// repos/testing_js.ts
// index 502f353..36d28fd 100644
// --- a/src/lib/utils.ts
// +++ b/src/lib/utils.ts
// @@ -33,7 +33,7 @@ export function extractTitleFromMessage(message: string) {
//
//  function getGroup(differenceInMs: number, promptDate: Date): string {
//    if (isToday(promptDate)) {
// -    return "Today";
// +    return "TODAY trigger asdasds";
//    }
//    if (isYesterday(promptDate)) {
//      return "Yesterday";
// src/App.tsx
import { Chat } from "./components/Chat";
import {
  Breadcrumb,
  BreadcrumbList,
} from "./components/ui/breadcrumb";
import { useBreadcrumb } from "./hooks/useBreadcrumb";

function App() {

  useEffect(() => {
    fetchPrompts();
    <|fim_suffix|>
  }, [fetchPrompts]);

  const test = "REDACTED<$dNO+OtQHhYyvq6OtrVQ6kFQKPPADwnRQrd3LNmfYr16/fRCzEGfHcmCjF8jA==>";
  console.log(test);
<|fim_suffix|>
  return (
    <div className="flex w-screen h-screen">
      <Sidebar loading={loading}>
<|fim_middle|>"""


@pytest.mark.parametrize(
    "message, provider, expected_filepath",
    [
        ("This is a test message", "test_provider", None),
        (fim_python_message, "copilot", "folder/testing_file.py"),
        (fim_python_message, "other", "another_folder/another_file.py"),
        (fim_js_message, "other", "src/App.tsx"),
    ],
)
def test_create_hash_key(message, provider, expected_filepath):
    fim_cache = FimCache()

    result_filepath = fim_cache._match_filepath(message, provider)
    assert result_filepath == expected_filepath


@pytest.mark.parametrize(
    "filepath, message, provider, expected_msg_to_hash",
    [
        (None, "Message", "test_provider", "Message-test_provider"),
        ("file", "Message", "test_provider", "file-test_provider"),
    ],
)
def test_calculate_hash_key(filepath, message, provider, expected_msg_to_hash):
    fim_cache = FimCache()
    fim_cache._match_filepath = mock.MagicMock(return_value=filepath)

    expected_hash_key = hashlib.sha256(expected_msg_to_hash.encode()).hexdigest()

    result_hash_key = fim_cache._calculate_hash_key(message, provider)
    assert result_hash_key == expected_hash_key


fim_request = {
    "model": "qwen2.5-coder-1.5b-instruct-q5_k_m",
    "max_tokens": 4096,
    "temperature": 0.33,
    "stream": True,
    "stop": [
        "<|endoftext|>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        "<|fim_pad|>",
        "<|repo_name|>",
        "<|file_sep|>",
        "<|im_start|>",
        "<|im_end|>",
        "/src/",
        "#- coding: utf-8",
        "```",
    ],
    "messages": [{"content": fim_js_message, "role": "user"}],
    "had_prompt_before": True,
}


@pytest.mark.parametrize(
    "test_request, expected_result_content",
    [
        ("This is a test message", None),
        (fim_request, fim_js_message),
    ],
)
def test_extract_message_from_fim_request(test_request, expected_result_content):
    fim_cache = FimCache()
    result_content = fim_cache._extract_message_from_fim_request(json.dumps(test_request))
    assert result_content == expected_result_content


def test_are_new_alerts_present():
    fim_cache = FimCache()
    cached_entry = CachedFim(timestamp=datetime.now(), critical_alerts=[], initial_id="1")
    context = PipelineContext()
    context.alerts_raised = [mock.MagicMock(trigger_category=AlertSeverity.CRITICAL.value)]
    result = fim_cache._are_new_alerts_present(context, cached_entry)
    assert result is True

    populated_cache = CachedFim(
        timestamp=datetime.now(),
        critical_alerts=[
            Alert(
                id="1",
                prompt_id="1",
                trigger_category=AlertSeverity.CRITICAL.value,
                timestamp=datetime.now(),
                trigger_type="test",
                code_snippet=None,
                trigger_string=None,
            )
        ],
        initial_id="2",
    )
    result = fim_cache._are_new_alerts_present(context, populated_cache)
    assert result is False


@pytest.mark.parametrize(
    "cached_entry, is_old",
    [
        (
            CachedFim(
                timestamp=datetime.now(timezone.utc) - timedelta(days=1),
                critical_alerts=[],
                initial_id="1",
            ),
            True,
        ),
        (
            CachedFim(timestamp=datetime.now(timezone.utc), critical_alerts=[], initial_id="2"),
            False,
        ),
    ],
)
def test_is_cached_entry_old(cached_entry, is_old):
    context = PipelineContext()
    context.add_input_request("test", True, "test_provider")  # type: ignore
    fim_cache = FimCache()
    result = fim_cache._is_cached_entry_old(context, cached_entry)
    assert result == is_old
