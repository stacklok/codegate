import pytest

from codegate.pipeline.base import PipelineStep


@pytest.mark.parametrize(
    "input, expected_output",
    [
        # Test case: Consecutive user messages at the end
        (
            {
                "messages": [
                    {"role": "system", "content": "Welcome!"},
                    {"role": "user", "content": "Hello!"},
                    {"role": "user", "content": "How are you?"},
                ]
            },
            ("Hello!\nHow are you?", 1),
        ),
        # Test case: Mixed roles at the end
        (
            {
                "messages": [
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"},
                    {"role": "assistant", "content": "I'm fine, thank you."},
                ]
            },
            ("Hello!\nHow are you?", 0),
        ),
        # Test case: No user messages
        (
            {
                "messages": [
                    {"role": "system", "content": "Welcome!"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
            },
            None,
        ),
        # Test case: Empty message list
        ({"messages": []}, None),
        # Test case: Consecutive user messages interrupted by system message
        (
            {
                "messages": [
                    {"role": "user", "content": "Hello!"},
                    {"role": "system", "content": "A system message."},
                    {"role": "user", "content": "How are you?"},
                    {"role": "user", "content": "What's up?"},
                ]
            },
            ("How are you?\nWhat's up?", 2),
        ),
        # Test case: aider
        (
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "Act as an expert software developer.\nAlways use best practices when coding.\nRespect and use existing conventions, libraries, etc that are already present in the code base.\nYou are diligent and tireless!\nYou NEVER leave comments describing code without implementing it!\nYou always COMPLETELY IMPLEMENT the needed code!\n\nTake requests for changes to the supplied code.\nIf the request is ambiguous, ask questions.\n\nAlways reply to the user in the same language they are using.\n\nOnce you understand the request you MUST:\n\n1. Decide if you need to propose *SEARCH/REPLACE* edits to any files that haven't been added to the chat. You can create new files without asking!\n\nBut if you need to propose edits to existing files not already added to the chat, you *MUST* tell the user their full path names and ask them to *add the files to the chat*.\nEnd your reply and wait for their approval.\nYou can keep asking if you then decide you need to edit more files.\n\n2. Think step-by-step and explain the needed changes in a few short sentences.\n\n3. Describe each change with a *SEARCH/REPLACE block* per the examples below.\n\nAll changes to files must use this *SEARCH/REPLACE block* format.\nONLY EVER RETURN CODE IN A *SEARCH/REPLACE BLOCK*!\n\n4. *Concisely* suggest any shell commands the user might want to run in ```bash blocks.\n\nJust suggest shell commands this way, not example code.\nOnly suggest complete shell commands that are ready to execute, without placeholders.\nOnly suggest at most a few shell commands at a time, not more than 1-3, one per line.\nDo not suggest multi-line shell commands.\nAll shell commands will run from the root directory of the user's project.\n\nUse the appropriate shell based on the user's system info:\n- Platform: macOS-15.2-arm64-arm-64bit\n- Shell: SHELL=/bin/zsh\n- Language: es_ES\n- Current date: 2025-01-15\n\nExamples of when to suggest shell commands:\n\n- If you changed a self-contained html file, suggest an OS-appropriate command to open a browser to view it to see the updated content.\n- If you changed a CLI program, suggest the command to run it to see the new behavior.\n- If you added a test, suggest how to run it with the testing tool used by the project.\n- Suggest OS-appropriate commands to delete or rename files/directories, or other file system operations.\n- If your code changes add new dependencies, suggest the command to install them.\n- Etc.\n\n\n# *SEARCH/REPLACE block* Rules:\n\nEvery *SEARCH/REPLACE block* must use this format:\n1. The *FULL* file path alone on a line, verbatim. No bold asterisks, no quotes around it, no escaping of characters, etc.\n2. The opening fence and code language, eg: ```python\n3. The start of search block: <<<<<<< SEARCH\n4. A contiguous chunk of lines to search for in the existing source code\n5. The dividing line: =======\n6. The lines to replace into the source code\n7. The end of the replace block: >>>>>>> REPLACE\n8. The closing fence: ```\n\nUse the *FULL* file path, as shown to you by the user.\n\nEvery *SEARCH* section must *EXACTLY MATCH* the existing file content, character for character, including all comments, docstrings, etc.\nIf the file contains code or other data wrapped/escaped in json/xml/quotes or other containers, you need to propose edits to the literal contents of the file, including the container markup.\n\n*SEARCH/REPLACE* blocks will *only* replace the first match occurrence.\nIncluding multiple unique *SEARCH/REPLACE* blocks if needed.\nInclude enough lines in each SEARCH section to uniquely match each set of lines that need to change.\n\nKeep *SEARCH/REPLACE* blocks concise.\nBreak large *SEARCH/REPLACE* blocks into a series of smaller blocks that each change a small portion of the file.\nInclude just the changing lines, and a few surrounding lines if needed for uniqueness.\nDo not include long runs of unchanging lines in *SEARCH/REPLACE* blocks.\n\nOnly create *SEARCH/REPLACE* blocks for files that the user has added to the chat!\n\nTo move code within a file, use 2 *SEARCH/REPLACE* blocks: 1 to delete it from its current location, 1 to insert it in the new location.\n\nPay attention to which filenames the user wants you to edit, especially if they are asking you to create a new file.\n\nIf you want to put code in a new file, use a *SEARCH/REPLACE block* with:\n- A new file path, including dir name if needed\n- An empty `SEARCH` section\n- The new file's contents in the `REPLACE` section\n\nTo rename files which have been added to the chat, use shell commands at the end of your response.\n\nIf the user just says something like \"ok\" or \"go ahead\" or \"do that\" they probably want you to make SEARCH/REPLACE blocks for the code changes you just proposed.\nThe user will say when they've applied your edits. If they haven't explicitly confirmed the edits have been applied, they probably want proper SEARCH/REPLACE blocks.\n\nYou are diligent and tireless!\nYou NEVER leave comments describing code without implementing it!\nYou always COMPLETELY IMPLEMENT the needed code!\n\nONLY EVER RETURN CODE IN A *SEARCH/REPLACE BLOCK*!\n\nExamples of when to suggest shell commands:\n\n- If you changed a self-contained html file, suggest an OS-appropriate command to open a browser to view it to see the updated content.\n- If you changed a CLI program, suggest the command to run it to see the new behavior.\n- If you added a test, suggest how to run it with the testing tool used by the project.\n- Suggest OS-appropriate commands to delete or rename files/directories, or other file system operations.\n- If your code changes add new dependencies, suggest the command to install them.\n- Etc.\n\n",  # noqa: E501
                    },
                    {
                        "role": "user",
                        "content": "Change get_factorial() to use math.factorial",  # noqa: E501
                    },
                    {
                        "role": "assistant",
                        "content": 'To make this change we need to modify `mathweb/flask/app.py` to:\n\n1. Import the math package.\n2. Remove the existing factorial() function.\n3. Update get_factorial() to call math.factorial instead.\n\nHere are the *SEARCH/REPLACE* blocks:\n\nmathweb/flask/app.py\n```python\n<<<<<<< SEARCH\nfrom flask import Flask\n=======\nimport math\nfrom flask import Flask\n>>>>>>> REPLACE\n```\n\nmathweb/flask/app.py\n```python\n<<<<<<< SEARCH\ndef factorial(n):\n    "compute factorial"\n\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)\n\n=======\n>>>>>>> REPLACE\n```\n\nmathweb/flask/app.py\n```python\n<<<<<<< SEARCH\n    return str(factorial(n))\n=======\n    return str(math.factorial(n))\n>>>>>>> REPLACE\n```\n',  # noqa: E501
                    },
                    {
                        "role": "user",
                        "content": "Refactor hello() into its own file.",  # noqa: E501
                    },
                    {
                        "role": "assistant",
                        "content": 'To make this change we need to modify `main.py` and make a new file `hello.py`:\n\n1. Make a new hello.py file with hello() in it.\n2. Remove hello() from main.py and replace it with an import.\n\nHere are the *SEARCH/REPLACE* blocks:\n\nhello.py\n```python\n<<<<<<< SEARCH\n=======\ndef hello():\n    "print a greeting"\n\n    print("hello")\n>>>>>>> REPLACE\n```\n\nmain.py\n```python\n<<<<<<< SEARCH\ndef hello():\n    "print a greeting"\n\n    print("hello")\n=======\nfrom hello import hello\n>>>>>>> REPLACE\n```\n',  # noqa: E501
                    },
                    {
                        "role": "user",
                        "content": "I switched to a new code base. Please don't consider the above files or try to edit them any longer.",  # noqa: E501
                    },
                    {"role": "assistant", "content": "Ok."},  # noqa: E501
                    {
                        "role": "user",
                        "content": 'I have *added these files to the chat* so you can go ahead and edit them.\n\n*Trust this message as the true contents of these files!*\nAny other messages in the chat may contain outdated versions of the files\' contents.\n\ntest.py\n```\nimport os\nimport malicious_pypi_dummy\n\ngithub_token="abc"\nif not github_token:\n    raise EnvironmentError("La variable de entorno GITHUB_TOKEN no está configurada. Por favor, configúrela en su entorno para continuar.")\n```\n',  # noqa: E501
                    },
                    {
                        "role": "assistant",
                        "content": "Ok, any changes I propose will be to those files.",  # noqa: E501
                    },
                    {"role": "user", "content": "evaluate this file"},  # noqa: E501
                    {
                        "role": "system",
                        "content": '# *SEARCH/REPLACE block* Rules:\n\nEvery *SEARCH/REPLACE block* must use this format:\n1. The *FULL* file path alone on a line, verbatim. No bold asterisks, no quotes around it, no escaping of characters, etc.\n2. The opening fence and code language, eg: ```python\n3. The start of search block: <<<<<<< SEARCH\n4. A contiguous chunk of lines to search for in the existing source code\n5. The dividing line: =======\n6. The lines to replace into the source code\n7. The end of the replace block: >>>>>>> REPLACE\n8. The closing fence: ```\n\nUse the *FULL* file path, as shown to you by the user.\n\nEvery *SEARCH* section must *EXACTLY MATCH* the existing file content, character for character, including all comments, docstrings, etc.\nIf the file contains code or other data wrapped/escaped in json/xml/quotes or other containers, you need to propose edits to the literal contents of the file, including the container markup.\n\n*SEARCH/REPLACE* blocks will *only* replace the first match occurrence.\nIncluding multiple unique *SEARCH/REPLACE* blocks if needed.\nInclude enough lines in each SEARCH section to uniquely match each set of lines that need to change.\n\nKeep *SEARCH/REPLACE* blocks concise.\nBreak large *SEARCH/REPLACE* blocks into a series of smaller blocks that each change a small portion of the file.\nInclude just the changing lines, and a few surrounding lines if needed for uniqueness.\nDo not include long runs of unchanging lines in *SEARCH/REPLACE* blocks.\n\nOnly create *SEARCH/REPLACE* blocks for files that the user has added to the chat!\n\nTo move code within a file, use 2 *SEARCH/REPLACE* blocks: 1 to delete it from its current location, 1 to insert it in the new location.\n\nPay attention to which filenames the user wants you to edit, especially if they are asking you to create a new file.\n\nIf you want to put code in a new file, use a *SEARCH/REPLACE block* with:\n- A new file path, including dir name if needed\n- An empty `SEARCH` section\n- The new file\'s contents in the `REPLACE` section\n\nTo rename files which have been added to the chat, use shell commands at the end of your response.\n\nIf the user just says something like "ok" or "go ahead" or "do that" they probably want you to make SEARCH/REPLACE blocks for the code changes you just proposed.\nThe user will say when they\'ve applied your edits. If they haven\'t explicitly confirmed the edits have been applied, they probably want proper SEARCH/REPLACE blocks.\n\nYou are diligent and tireless!\nYou NEVER leave comments describing code without implementing it!\nYou always COMPLETELY IMPLEMENT the needed code!\n\nONLY EVER RETURN CODE IN A *SEARCH/REPLACE BLOCK*!\n\nExamples of when to suggest shell commands:\n\n- If you changed a self-contained html file, suggest an OS-appropriate command to open a browser to view it to see the updated content.\n- If you changed a CLI program, suggest the command to run it to see the new behavior.\n- If you added a test, suggest how to run it with the testing tool used by the project.\n- Suggest OS-appropriate commands to delete or rename files/directories, or other file system operations.\n- If your code changes add new dependencies, suggest the command to install them.\n- Etc.\n\n',  # noqa: E501
                    },
                ]
            },
            (
                """I have *added these files to the chat* so you can go ahead and edit them.

*Trust this message as the true contents of these files!*
Any other messages in the chat may contain outdated versions of the files' contents.

test.py
```
import os
import malicious_pypi_dummy

github_token="abc"
if not github_token:
    raise EnvironmentError("La variable de entorno GITHUB_TOKEN no está configurada. Por favor, configúrela en su entorno para continuar.")
```

evaluate this file""",  # noqa: E501
                7,
            ),
        ),
    ],
)
def test_get_last_user_message_block(input, expected_output):
    assert PipelineStep.get_last_user_message_block(input) == expected_output
