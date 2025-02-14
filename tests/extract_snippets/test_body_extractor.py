from typing import Dict, List, NamedTuple

import pytest

from codegate.extract_snippets.body_extractor import (
    ClineBodySnippetExtractor,
    ContinueBodySnippetExtractor,
    KoduBodySnippetExtractor,
    OpenInterpreterBodySnippetExtractor,
)


class BodyCodeSnippetTest(NamedTuple):
    input_body_dict: Dict[str, List[Dict[str, str]]]
    expected_count: int
    expected: List[str]


def _evaluate_actual_filenames(filenames: set[str], test_case: BodyCodeSnippetTest):
    assert len(filenames) == test_case.expected_count
    assert filenames == set(test_case.expected)


@pytest.mark.parametrize(
    "test_case",
    [
        # Analyze processed snippets from OpenInterpreter
        BodyCodeSnippetTest(
            input_body_dict={
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "toolu_4",
                                "type": "function",
                                "function": {
                                    "name": "execute",
                                    "arguments": (
                                        '{"language": "python", "code": "\\n'
                                        "# Open and read the contents of the src/codegate/api/v1.py"
                                        " file\\n"
                                        "with open('src/codegate/api/v1.py', 'r') as file:\\n    "
                                        'content = file.read()\\n\\ncontent\\n"}'
                                    ),
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "name": "execute",
                        "content": (
                            "Output truncated.\n\nr as e:\\n    "
                            'raise HTTPException(status_code=400",'
                        ),
                        "tool_call_id": "toolu_4",
                    },
                ]
            },
            expected_count=1,
            expected=["v1.py"],
        ),
    ],
)
def test_body_extract_openinterpreter_snippets(test_case: BodyCodeSnippetTest):
    extractor = OpenInterpreterBodySnippetExtractor()
    filenames = extractor.extract_unique_filenames(test_case.input_body_dict)
    _evaluate_actual_filenames(filenames, test_case)


@pytest.mark.parametrize(
    "test_case",
    [
        # Analyze processed snippets from OpenInterpreter
        BodyCodeSnippetTest(
            input_body_dict={
                "messages": [
                    {"role": "system", "content": "You are Cline, a highly skilled software"},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": '''
        [<task>
now please analyze the folder 'codegate/src/codegate/api/' (see below for folder content)
</task>

<folder_content path="codegate/src/codegate/api/">
├── __init__.py
├── __pycache__/
├── v1.py
├── v1_models.py
└── v1_processing.py

<file_content path="codegate/src/codegate/api/__init__.py">

</file_content>

<file_content path="codegate/src/codegate/api/v1.py">
from typing import List, Optional
from uuid import UUID

import requests
import structlog

v1 = APIRouter()
wscrud = crud.WorkspaceCrud()
pcrud = provendcrud.ProviderCrud()

</file_content>

<file_content path="codegate/src/codegate/api/v1_models.py">
import datetime
from enum import Enum


class Conversation(pydantic.BaseModel):
    """
    Represents a conversation.
    """

    question_answers: List[QuestionAnswer]
    provider: Optional[str]
    type: QuestionType
    chat_id: str
    conversation_timestamp: datetime.datetime
    token_usage_agg: Optional[TokenUsageAggregate]

</file_content>

<file_content path="codegate/src/codegate/api/v1_processing.py">
import asyncio
import json
import re
from collections import defaultdict

async def _process_prompt_output_to_partial_qa(
    prompts_outputs: List[GetPromptWithOutputsRow],
) -> List[PartialQuestionAnswer]:
    """
    Process the prompts and outputs to PartialQuestionAnswer objects.
    """
    # Parse the prompts and outputs in parallel
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(_get_partial_question_answer(row)) for row in prompts_outputs]
    return [task.result() for task in tasks if task.result() is not None]

</file_content>
</folder_content>
        ''',
                            }
                        ],
                    },
                ]
            },
            expected_count=4,
            expected=["__init__.py", "v1.py", "v1_models.py", "v1_processing.py"],
        ),
    ],
)
def test_body_extract_cline_snippets(test_case: BodyCodeSnippetTest):
    extractor = ClineBodySnippetExtractor()
    filenames = extractor.extract_unique_filenames(test_case.input_body_dict)
    _evaluate_actual_filenames(filenames, test_case)


@pytest.mark.parametrize(
    "test_case",
    [
        # Analyze processed snippets from OpenInterpreter
        BodyCodeSnippetTest(
            input_body_dict={
                "messages": [
                    {
                        "role": "user",
                        "content": """
                        ```file:///Users/user/StacklokRepos/testing_file.py
import invokehttp
import fastapi
from fastapi import FastAPI, Request, Response, HTTPException
import numpy

GITHUB_TOKEN="ghp_1J9Z3Z2dfg4dfs23dsfsdf232aadfasdfasfasdf32"

def add(a, b):
        return a + b

def multiply(a, b):
        return a * b



def substract(a, b):

```

please analyze testing_file.py
                        """,
                    }
                ],
                "model": "foo-model-replaced-by-mux",
                "max_tokens": 4096,
                "stream": True,
            },
            expected_count=1,
            expected=["testing_file.py"],
        ),
    ],
)
def test_body_extract_continue_snippets(test_case: BodyCodeSnippetTest):
    extractor = ContinueBodySnippetExtractor()
    filenames = extractor.extract_unique_filenames(test_case.input_body_dict)
    _evaluate_actual_filenames(filenames, test_case)


@pytest.mark.parametrize(
    "test_case",
    [
        # Analyze processed snippets from Kodu
        BodyCodeSnippetTest(
            input_body_dict={
                "messages": [
                    {"role": "system", "content": "You are Kodu, an autonomous coding agent."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """
Here is our task for this conversation, you must remember it all time unless i tell you otherwise.
<task>
please analyze
	<additional-context>
	- Super critical information, the files attached here are part of the task and need to be
	- The URLs attached here need to be scrapped and the information should be used for the
	- The files passed in context are provided to help you understand the task better, the
	<files count="1"><file path="testing_file.py">import invokehttp
import fastapi
from fastapi import FastAPI, Request, Response, HTTPException
import numpy

GITHUB_TOKEN="ghp_1J9Z3Z2dfg4dfs23dsfsdf232aadfasdfasfasdf32"

def add(a, b):
     return a + b

def multiply(a, b):
     return a * b



def substract(a, b):
     </file></files>
	<urls></urls>
	</additional-context>

</task>
        """,
                            }
                        ],
                    },
                    {
                        "type": "text",
                        "text": """
You must use a tool to proceed. Either use attempt_completion if you've completed the task,
or ask_followup_question if you need more information. you must adhere to the tool format
<kodu_action><tool_name><parameter1_name>value1</parameter1_name><parameter2_name>value2
</parameter2_name>... additional parameters as needed in the same format
...</tool_name></kodu_action>
""",
                    },
                ]
            },
            expected_count=1,
            expected=["testing_file.py"],
        ),
    ],
)
def test_body_extract_kodu_snippets(test_case: BodyCodeSnippetTest):
    extractor = KoduBodySnippetExtractor()
    filenames = extractor.extract_unique_filenames(test_case.input_body_dict)
    _evaluate_actual_filenames(filenames, test_case)
