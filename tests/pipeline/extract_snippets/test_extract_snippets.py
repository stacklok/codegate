from typing import List, NamedTuple

import pytest

from codegate.pipeline.base import CodeSnippet
from codegate.pipeline.extract_snippets.extract_snippets import (
    ecosystem_from_filepath,
    extract_snippets,
)


class CodeSnippetTest(NamedTuple):
    input_message: str
    expected_count: int
    expected: List[CodeSnippet]


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
                CodeSnippet(language=None, filepath=None, code='print("Hello, world!")'),
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
                    language="python", filepath="hello_world.py", code='print("Hello, world!")'
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
                    language="python", filepath="goodbye_world.py", code='print("Goodbye, world!")'
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
                ),
            ],
        ),
    ],
)
def test_extract_snippets(test_case):
    snippets = extract_snippets(test_case.input_message)

    assert len(snippets) == test_case.expected_count

    for expected, actual in zip(test_case.expected, snippets):
        assert actual.language == expected.language
        assert actual.filepath == expected.filepath
        assert expected.code in actual.code


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
    assert ecosystem_from_filepath(filepath) == expected


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
    assert ecosystem_from_filepath(filepath) is None
