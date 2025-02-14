from typing import List, NamedTuple

import pytest

from codegate.extract_snippets.message_extractor import (
    AiderCodeSnippetExtractor,
    ClineCodeSnippetExtractor,
    CodeSnippet,
    DefaultCodeSnippetExtractor,
    KoduCodeSnippetExtractor,
    OpenInterpreterCodeSnippetExtractor,
)


class CodeSnippetTest(NamedTuple):
    input_message: str
    expected_count: int
    expected: List[CodeSnippet]


def _evaluate_actual_snippets(actual_snips: List[CodeSnippet], expected_snips: CodeSnippetTest):
    assert len(actual_snips) == expected_snips.expected_count

    for expected, actual in zip(expected_snips.expected, actual_snips):
        assert actual.language == expected.language
        assert actual.filepath == expected.filepath
        assert actual.file_extension == expected.file_extension
        assert expected.code in actual.code


@pytest.mark.parametrize(
    "test_case",
    [
        # Single Python code block without filename
        CodeSnippetTest(
            input_message=""":
        Here's a Python snippet:
        ```
        def hello():
            print("Hello, world!")
        ```
        """,
            expected_count=1,
            expected=[
                CodeSnippet(
                    language=None, filepath=None, code='print("Hello, world!")', file_extension=None
                ),
            ],
        ),
        # output code snippet with no filename
        CodeSnippetTest(
            input_message=""":
        ```python
        @app.route('/')
        def hello():
            GITHUB_TOKEN="ghp_RjzIRljYij9CznoS7QAnD5RaFF6yH32073uI"
            if __name__ == '__main__':
                app.run()
        return "Hello, Moon!"
        ```
        """,
            expected_count=1,
            expected=[
                CodeSnippet(
                    language="python",
                    filepath=None,
                    code="Hello, Moon!",
                    file_extension=None,
                ),
            ],
        ),
        # Single Python code block
        CodeSnippetTest(
            input_message="""
        Here's a Python snippet:
        ```hello_world.py (8-13)
        def hello():
            print("Hello, world!")
        ```
        """,
            expected_count=1,
            expected=[
                CodeSnippet(
                    language="python",
                    filepath="hello_world.py",
                    code='print("Hello, world!")',
                    file_extension=".py",
                ),
            ],
        ),
        # Single Python code block with a language identifier
        CodeSnippetTest(
            input_message="""
        Here's a Python snippet:
        ```py goodbye_world.py (8-13)
        def hello():
            print("Goodbye, world!")
        ```
        """,
            expected_count=1,
            expected=[
                CodeSnippet(
                    language="python",
                    filepath="goodbye_world.py",
                    code='print("Goodbye, world!")',
                    file_extension=".py",
                ),
            ],
        ),
        # Multiple code blocks with different languages
        CodeSnippetTest(
            input_message="""
        Python snippet:
        ```main.py
        def hello():
            print("Hello")
        ```

        JavaScript snippet:
        ```script.js (1-3)
        function greet() {
            console.log("Hi");
        }
        ```
        """,
            expected_count=2,
            expected=[
                CodeSnippet(language="python", filepath="main.py", code='print("Hello")'),
                CodeSnippet(
                    language="javascript",
                    filepath="script.js",
                    code='console.log("Hi");',
                    file_extension=".js",
                ),
            ],
        ),
        # No code blocks
        CodeSnippetTest(
            input_message="Just a plain text message",
            expected_count=0,
            expected=[],
        ),
        # unknown language
        CodeSnippetTest(
            input_message=""":
        Here's a Perl snippet:
        ```hello_world.pl
        I'm a Perl script
        ```
        """,
            expected_count=1,
            expected=[
                CodeSnippet(
                    language=None,
                    filepath="hello_world.pl",
                    code="I'm a Perl script",
                    file_extension=".pl",
                ),
            ],
        ),
    ],
)
def test_extract_snippets(test_case: CodeSnippetTest):
    extractor = DefaultCodeSnippetExtractor()
    snippets = extractor.extract_snippets(test_case.input_message)
    _evaluate_actual_snippets(snippets, test_case)


@pytest.mark.parametrize(
    "test_case",
    [
        # Single snippet from Continue
        CodeSnippetTest(
            input_message="""


        ```py testing_file.py (1-17)
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
        analyze this file
            """,
            expected_count=1,
            expected=[
                CodeSnippet(
                    language="python",
                    filepath="testing_file.py",
                    code="def multiply(a, b):",
                    file_extension=".py",
                ),
            ],
        ),
        # Two snippets from Continue, one inserting with CTRL+L and another one with @
        CodeSnippetTest(
            input_message='''
```/Users/user/StacklokRepos/codegate/tests/pipeline/extract_snippets/test_extract_snippets.py
from typing import List, NamedTuple

import pytest

from codegate.pipeline.extract_snippets.extract_snippets import CodeSnippet, CodeSnippetExtractor


class CodeSnippetTest(NamedTuple):
    input_message: str
    expected_count: int
    expected: List[CodeSnippet]

@pytest.mark.parametrize(
    "filepath",
    [
        # No extension
        "README",
        "script",
        "README.txt",
        # Unknown extensions
        "file.xyz",
        "unknown.extension",
    ],
)
def test_no_or_unknown_extensions(filepath):
    extractor = CodeSnippetExtractor()
    assert extractor._ecosystem_from_filepath(filepath) is None

```



```py codegate/src/codegate/pipeline/extract_snippets/extract_snippets.py (24-50)
class CodeSnippet(BaseModel):
    """
    Represents a code snippet with its programming language.

    Args:
        language: The programming language identifier (e.g., 'python', 'javascript')
        code: The actual code content
    """

    code: str
    language: Optional[str]
    filepath: Optional[str]
    libraries: List[str] = []
    file_extension: Optional[str] = None
```
analyze this file with respect to test_extract_snippets.py
            ''',
            expected_count=2,
            expected=[
                CodeSnippet(
                    language="python",
                    filepath="/Users/user/StacklokRepos/codegate/tests/pipeline/extract_snippets/test_extract_snippets.py",
                    code="def test_no_or_unknown_extensions(filepath):",
                    file_extension=".py",
                ),
                CodeSnippet(
                    language="python",
                    filepath="codegate/src/codegate/pipeline/extract_snippets/extract_snippets.py",
                    code="class CodeSnippet(BaseModel)",
                    file_extension=".py",
                ),
            ],
        ),
        # Two snippets from Continue, one inserting with CTRL+L and another one with @
        CodeSnippetTest(
            input_message="""
```/Users/foo_user/StacklokRepos/src/README.MD
# Handling changes

Changes are not immediate

### Example

On the response class "Package", changing "description" "repo_description":

From:
```
@dataclass
class Package:
    id: str
    name: str
```

To:
```
@dataclass
class Package:
    id: str
    name: str
```

And finally

```
@dataclass
class Package:
    id: str
    name: str
    type: str
```
```

README.MD and that file?
            """,
            expected_count=1,
            expected=[
                CodeSnippet(
                    language=None,
                    filepath="/Users/foo_user/StacklokRepos/src/README.MD",
                    code="Changes are not immediate",
                    file_extension=".md",
                ),
            ],
        ),
    ],
)
def test_extract_snippets_require_filepath(test_case: CodeSnippetTest):
    extractor = DefaultCodeSnippetExtractor()
    snippets = extractor.extract_snippets(test_case.input_message, require_filepath=True)
    _evaluate_actual_snippets(snippets, test_case)


@pytest.mark.parametrize(
    "test_case",
    [
        # Analyze folder from Cline
        CodeSnippetTest(
            input_message="""
[TASK RESUMPTION] This task was interrupted 1 day ago.
It may or may not be complete, so please reassess the task context.
Be aware that the project state may have changed since then.
The current working directory is now '/Users/aponcedeleonch/StacklokRepos'.
If the task has not been completed, retry the last step before interruption
and proceed with completing the task.

Note: If you previously attempted a tool use that the user did not provide a result for,
you should assume the tool use was not successful and assess whether you should retry.
If the last tool was a browser_action, the browser has been closed and you must launch a new
browser if needed.

New instructions for task continuation:
<user_message>
please evaluate my folder 'codegate/tests/pipeline/extract_snippets/' (see below for folder content)
  and suggest improvements
</user_message>

<folder_content path="codegate/tests/pipeline/extract_snippets/">
├── __pycache__/
└── test_extract_snippets.py

<file_content path="codegate/tests/pipeline/extract_snippets/test_extract_snippets.py">
from typing import List, NamedTuple

import pytest

from codegate.pipeline.extract_snippets.extract_snippets import CodeSnippet, CodeSnippetExtractor


class CodeSnippetTest(NamedTuple):
    input_message: str
    expected_count: int
    expected: List[CodeSnippet]


def _evaluate_actual_snippets(actual_snips: List[CodeSnippet], expected_snips: CodeSnippetTest):
    assert len(actual_snips) == expected_snips.expected_count

    for expected, actual in zip(expected_snips.expected, actual_snips):
        assert actual.language == expected.language
        assert actual.filepath == expected.filepath
        assert actual.file_extension == expected.file_extension
        assert expected.code in actual.code

</file_content>
</folder_content>
            """,
            expected_count=1,
            expected=[
                CodeSnippet(
                    language="python",
                    filepath="codegate/tests/pipeline/extract_snippets/test_extract_snippets.py",
                    code="def _evaluate_actual_snippets",
                    file_extension=".py",
                ),
            ],
        ),
        # Several snippets from Cline
        CodeSnippetTest(
            input_message='''
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
            expected_count=4,
            expected=[
                CodeSnippet(
                    language="python",
                    filepath="codegate/src/codegate/api/__init__.py",
                    code="",
                    file_extension=".py",
                ),
                CodeSnippet(
                    language="python",
                    filepath="codegate/src/codegate/api/v1.py",
                    code="v1 = APIRouter()",
                    file_extension=".py",
                ),
                CodeSnippet(
                    language="python",
                    filepath="codegate/src/codegate/api/v1_models.py",
                    code="class Conversation(pydantic.BaseModel):",
                    file_extension=".py",
                ),
                CodeSnippet(
                    language="python",
                    filepath="codegate/src/codegate/api/v1_processing.py",
                    code="async def _process_prompt_output_to_partial_qa(",
                    file_extension=".py",
                ),
            ],
        ),
    ],
)
def test_extract_cline_snippets(test_case: CodeSnippetTest):
    extractor = ClineCodeSnippetExtractor()
    snippets = extractor.extract_snippets(test_case.input_message, require_filepath=True)
    _evaluate_actual_snippets(snippets, test_case)


@pytest.mark.parametrize(
    "test_case",
    [
        # Analyze summary extracts from Aider
        CodeSnippetTest(
            input_message='''
Here are summaries of some files present in my git repository.
Do not propose changes to these files, treat them as *read-only*.
If you need to edit any of these files, ask me to *add them to the chat* first.

src/codegate/codegate_logging.py:
⋮...
│def serialize_for_logging(obj: Any) -> Any:
⋮...

src/codegate/config.py:
⋮...
│@dataclass
│class Config:
│    """Application configuration with priority resolution."""
│
⋮...
│    @classmethod
│    def from_file(cls, config_path: Union[str, Path]) -> "Config":
⋮...
│    @classmethod
│    def load(
│        cls,
│        config_path: Optional[Union[str, Path]] = None,
│        prompts_path: Optional[Union[str, Path]] = None,
│        cli_port: Optional[int] = None,
│        cli_proxy_port: Optional[int] = None,
│        cli_host: Optional[str] = None,
│        cli_log_level: Optional[str] = None,
│        cli_log_format: Optional[str] = None,
│        cli_provider_urls: Optional[Dict[str, str]] = None,
⋮...
│    @classmethod
│    def get_config(cls) -> "Config":
⋮...

src/codegate/db/connection.py:
⋮...
│class DbRecorder(DbCodeGate):
│
⋮...
│class DbReader(DbCodeGate):
│
│    def __init__(self, sqlite_path: Optional[str] = None):
⋮...
│    async def get_workspace_by_name(self, name: str) -> Optional[WorkspaceRow]:
⋮...
│    async def get_active_workspace(self) -> Optional[ActiveWorkspace]:
⋮...

src/codegate/db/fim_cache.py:
⋮...
│class CachedFim(BaseModel):
│
⋮...

src/codegate/db/models.py:
⋮...
│class Alert(BaseModel):
⋮...
│class Output(BaseModel):
⋮...
│class Prompt(BaseModel):
⋮...
│class TokenUsage(BaseModel):
⋮...
│class ActiveWorkspace(BaseModel):
⋮...
│class ProviderEndpoint(BaseModel):
⋮...

            ''',
            expected_count=5,
            expected=[
                CodeSnippet(
                    language="python",
                    filepath="src/codegate/codegate_logging.py",
                    code="def serialize_for_logging(obj: Any) -> Any:",
                    file_extension=".py",
                ),
                CodeSnippet(
                    language="python",
                    filepath="src/codegate/config.py",
                    code="class Config:",
                    file_extension=".py",
                ),
                CodeSnippet(
                    language="python",
                    filepath="src/codegate/db/connection.py",
                    code="class DbReader(DbCodeGate):",
                    file_extension=".py",
                ),
                CodeSnippet(
                    language="python",
                    filepath="src/codegate/db/fim_cache.py",
                    code="class CachedFim(BaseModel):",
                    file_extension=".py",
                ),
                CodeSnippet(
                    language="python",
                    filepath="src/codegate/db/models.py",
                    code="class Alert(BaseModel):",
                    file_extension=".py",
                ),
            ],
        ),
        # Analyze file from Aider
        CodeSnippetTest(
            input_message="""
I have *added these files to the chat* so you can go ahead and edit them.

*Trust this message as the true contents of these files!*
Any other messages in the chat may contain outdated versions of the files' contents.

src/codegate/api/v1_models.py
```
import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import pydantic

class Workspace(pydantic.BaseModel):
    name: str
    is_active: bool

```
            """,
            expected_count=1,
            expected=[
                CodeSnippet(
                    language="python",
                    filepath="src/codegate/api/v1_models.py",
                    code="class Workspace(pydantic.BaseModel):",
                    file_extension=".py",
                ),
            ],
        ),
    ],
)
def test_extract_aider_snippets(test_case: CodeSnippetTest):
    extractor = AiderCodeSnippetExtractor()
    snippets = extractor.extract_snippets(test_case.input_message, require_filepath=True)
    _evaluate_actual_snippets(snippets, test_case)


@pytest.mark.parametrize(
    "test_case",
    [
        # Analyze processed snippets from OpenInterpreter
        CodeSnippetTest(
            input_message="""
{"language": "python", "code": "# Attempting to read the content of `codegate/api/v1_processing.py`
to analyze its functionality.\nv1_processing_path =
os.path.abspath('src/codegate/api/v1_processing.py')\n\ntry:\n    with open(v1_processing_path, 'r')
 as file:\n        v1_processing_content = file.read()\n        print('File read successfully.')
 \nexcept Exception as e:\n    v1_processing_content = str(e)\n\nv1_processing_content[:1000]
 # Displaying part of the content"}
File read successfully.
'import asyncio\nimport json\nimport re\nfrom collections import defaultdict\nfrom typing
import AsyncGenerator, Dict, List, Optional, Tuple\n\nimport cachetools.func\nimport requests\n
import structlog\n\nfrom codegate.api.v1_models import (\n    AlertConversation,\n    ChatMessage,\n
    \ndef fetch_latest_version() -> st'
            """,
            expected_count=1,
            expected=[
                CodeSnippet(
                    language="python",
                    filepath="codegate/api/v1_processing.py",
                    code="from codegate.api.v1_models import",
                    file_extension=".py",
                ),
            ],
        ),
        # Analyze processed snippets from OpenInterpreter when setting -y option
        CodeSnippetTest(
            input_message="""
{"language": "python", "code": "\n# Open and read the contents of the src/codegate/api/v1.py file\n
with open('src/codegate/api/v1.py', 'r') as file:\n    content = file.read()\n\ncontent\n"}
Output truncated. Showing the last 2800 characters.
You should try again and use computer.ai.summarize(output) over the output, or break it down into
smaller steps. Run `get_last_output()[0:2800]` to see the first page.

r as e:\n        raise HTTPException(status_code=400, detail=str(e))\n    except Exception:\n
logger.exception("Error while setting muxes")\n
raise HTTPException(status_code=500, detail="Internal server error")\n\n
return Response(status_code=204)\n\n\n@v1.get("/alerts_notification", tags=["Dashboard"]
            """,
            expected_count=1,
            expected=[
                CodeSnippet(
                    language="python",
                    filepath="src/codegate/api/v1.py",
                    code="raise HTTPException",
                    file_extension=".py",
                ),
            ],
        ),
    ],
)
def test_extract_openinterpreter_snippets(test_case: CodeSnippetTest):
    extractor = OpenInterpreterCodeSnippetExtractor()
    snippets = extractor.extract_snippets(test_case.input_message, require_filepath=True)
    _evaluate_actual_snippets(snippets, test_case)


@pytest.mark.parametrize(
    "test_case",
    [
        # Analyze processed snippets from OpenInterpreter
        CodeSnippetTest(
            input_message="""
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
            expected_count=1,
            expected=[
                CodeSnippet(
                    language="python",
                    filepath="testing_file.py",
                    code="def multiply(a, b):",
                    file_extension=".py",
                ),
            ],
        ),
    ],
)
def test_extract_kodu_snippets(test_case: CodeSnippetTest):
    extractor = KoduCodeSnippetExtractor()
    snippets = extractor.extract_snippets(test_case.input_message, require_filepath=True)
    _evaluate_actual_snippets(snippets, test_case)


@pytest.mark.parametrize(
    "filepath,expected",
    [
        # Standard extensions
        ("file.py", "python"),
        ("script.js", "javascript"),
        ("code.go", "go"),
        ("app.ts", "typescript"),
        ("component.tsx", "typescript"),
        ("program.rs", "rust"),
        ("App.java", "java"),
        # Case insensitive
        ("FILE.PY", "python"),
        ("SCRIPT.JS", "javascript"),
        # Full paths
        ("/path/to/file.rs", "rust"),
        ("C:\\Users\\name\\file.java", "java"),
    ],
)
def test_valid_extensions(filepath, expected):
    extractor = DefaultCodeSnippetExtractor()
    assert extractor._ecosystem_from_filepath(filepath) == expected


@pytest.mark.parametrize(
    "filepath",
    [
        # No extension
        "README",
        "script",
        "README.txt",
        # Unknown extensions
        "file.xyz",
        "unknown.extension",
    ],
)
def test_no_or_unknown_extensions(filepath):
    extractor = DefaultCodeSnippetExtractor()
    assert extractor._ecosystem_from_filepath(filepath) is None


@pytest.mark.parametrize(
    "expected_filenames,message",
    [
        # Single snippet
        (
            ["main.py"],
            """
                    ```main.py
                    foo
                    ```
                    """,
        ),
        # Repeated snippet
        (
            ["main.py"],
            """
                    ```main.py
                    foo
                    ```

                    ```main.py
                    bar
                    ```
                    """,
        ),
        # Multiple snippets
        (
            ["main.py", "snippets.py"],
            """
                    ```main.py
                    foo
                    ```

                    ```src/codegate/snippets.py
                    bar
                    ```
                    """,
        ),
    ],
)
def test_extract_unique_snippets(expected_filenames: List[str], message: str):
    extractor = DefaultCodeSnippetExtractor()
    snippets = extractor.extract_unique_snippets(message)

    actual_code_hashes = snippets.keys()
    assert len(actual_code_hashes) == len(expected_filenames)
    assert set(actual_code_hashes) == set(expected_filenames)
