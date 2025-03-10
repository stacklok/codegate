import pytest

from codegate.pipeline.base import PipelineStep
from codegate.types.openai import ChatCompletionRequest


@pytest.mark.parametrize(
    "input, expected_output",
    [
        # Test case: Consecutive user messages at the end
        (
            {
                "model": "model",
                "messages": [
                    {"role": "system", "content": "Welcome!"},
                    {"role": "user", "content": "Hello!"},
                    {"role": "user", "content": "How are you?"},
                ],
            },
            ("Hello!\nHow are you?", 1),
        ),
        # Test case: Assistant message at the end returns an empty block
        (
            {
                "model": "model",
                "messages": [
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"},
                    {"role": "assistant", "content": "I'm fine, thank you."},
                ],
            },
            None,
        ),
        # Test case: No user messages
        (
            {
                "model": "model",
                "messages": [
                    {"role": "system", "content": "Welcome!"},
                    {"role": "assistant", "content": "Hi there!"},
                ],
            },
            None,
        ),
        # Test case: Empty message list
        ({"model": "model", "messages": []}, None),
        # Test case: Consecutive user messages interrupted by system
        # message. This is all a single user block.
        (
            {
                "model": "model",
                "messages": [
                    {"role": "user", "content": "Hello!"},
                    {"role": "system", "content": "A system message."},
                    {"role": "user", "content": "How are you?"},
                    {"role": "user", "content": "What's up?"},
                ],
            },
            ("Hello!\nHow are you?\nWhat's up?", 0),
        ),
        #         # Test case: aider
        #         (
        #             {
        #                 "model": "model",
        #                 "messages": [
        #                     {
        #                         "role": "system",
        #                         "content": "Act as an expert software developer.\nAlways use best practices when coding.\nRespect and use existing conventions, libraries, etc that are already present in the code base.\nYou are diligent and tireless!\nYou NEVER leave comments describing code without implementing it!\nYou always COMPLETELY IMPLEMENT the needed code!\n\nTake requests for changes to the supplied code.\nIf the request is ambiguous, ask questions.\n\nAlways reply to the user in the same language they are using.\n\nOnce you understand the request you MUST:\n\n1. Decide if you need to propose *SEARCH/REPLACE* edits to any files that haven't been added to the chat. You can create new files without asking!\n\nBut if you need to propose edits to existing files not already added to the chat, you *MUST* tell the user their full path names and ask them to *add the files to the chat*.\nEnd your reply and wait for their approval.\nYou can keep asking if you then decide you need to edit more files.\n\n2. Think step-by-step and explain the needed changes in a few short sentences.\n\n3. Describe each change with a *SEARCH/REPLACE block* per the examples below.\n\nAll changes to files must use this *SEARCH/REPLACE block* format.\nONLY EVER RETURN CODE IN A *SEARCH/REPLACE BLOCK*!\n\n4. *Concisely* suggest any shell commands the user might want to run in ```bash blocks.\n\nJust suggest shell commands this way, not example code.\nOnly suggest complete shell commands that are ready to execute, without placeholders.\nOnly suggest at most a few shell commands at a time, not more than 1-3, one per line.\nDo not suggest multi-line shell commands.\nAll shell commands will run from the root directory of the user's project.\n\nUse the appropriate shell based on the user's system info:\n- Platform: macOS-15.2-arm64-arm-64bit\n- Shell: SHELL=/bin/zsh\n- Language: es_ES\n- Current date: 2025-01-15\n\nExamples of when to suggest shell commands:\n\n- If you changed a self-contained html file, suggest an OS-appropriate command to open a browser to view it to see the updated content.\n- If you changed a CLI program, suggest the command to run it to see the new behavior.\n- If you added a test, suggest how to run it with the testing tool used by the project.\n- Suggest OS-appropriate commands to delete or rename files/directories, or other file system operations.\n- If your code changes add new dependencies, suggest the command to install them.\n- Etc.\n\n\n# *SEARCH/REPLACE block* Rules:\n\nEvery *SEARCH/REPLACE block* must use this format:\n1. The *FULL* file path alone on a line, verbatim. No bold asterisks, no quotes around it, no escaping of characters, etc.\n2. The opening fence and code language, eg: ```python\n3. The start of search block: <<<<<<< SEARCH\n4. A contiguous chunk of lines to search for in the existing source code\n5. The dividing line: =======\n6. The lines to replace into the source code\n7. The end of the replace block: >>>>>>> REPLACE\n8. The closing fence: ```\n\nUse the *FULL* file path, as shown to you by the user.\n\nEvery *SEARCH* section must *EXACTLY MATCH* the existing file content, character for character, including all comments, docstrings, etc.\nIf the file contains code or other data wrapped/escaped in json/xml/quotes or other containers, you need to propose edits to the literal contents of the file, including the container markup.\n\n*SEARCH/REPLACE* blocks will *only* replace the first match occurrence.\nIncluding multiple unique *SEARCH/REPLACE* blocks if needed.\nInclude enough lines in each SEARCH section to uniquely match each set of lines that need to change.\n\nKeep *SEARCH/REPLACE* blocks concise.\nBreak large *SEARCH/REPLACE* blocks into a series of smaller blocks that each change a small portion of the file.\nInclude just the changing lines, and a few surrounding lines if needed for uniqueness.\nDo not include long runs of unchanging lines in *SEARCH/REPLACE* blocks.\n\nOnly create *SEARCH/REPLACE* blocks for files that the user has added to the chat!\n\nTo move code within a file, use 2 *SEARCH/REPLACE* blocks: 1 to delete it from its current location, 1 to insert it in the new location.\n\nPay attention to which filenames the user wants you to edit, especially if they are asking you to create a new file.\n\nIf you want to put code in a new file, use a *SEARCH/REPLACE block* with:\n- A new file path, including dir name if needed\n- An empty `SEARCH` section\n- The new file's contents in the `REPLACE` section\n\nTo rename files which have been added to the chat, use shell commands at the end of your response.\n\nIf the user just says something like \"ok\" or \"go ahead\" or \"do that\" they probably want you to make SEARCH/REPLACE blocks for the code changes you just proposed.\nThe user will say when they've applied your edits. If they haven't explicitly confirmed the edits have been applied, they probably want proper SEARCH/REPLACE blocks.\n\nYou are diligent and tireless!\nYou NEVER leave comments describing code without implementing it!\nYou always COMPLETELY IMPLEMENT the needed code!\n\nONLY EVER RETURN CODE IN A *SEARCH/REPLACE BLOCK*!\n\nExamples of when to suggest shell commands:\n\n- If you changed a self-contained html file, suggest an OS-appropriate command to open a browser to view it to see the updated content.\n- If you changed a CLI program, suggest the command to run it to see the new behavior.\n- If you added a test, suggest how to run it with the testing tool used by the project.\n- Suggest OS-appropriate commands to delete or rename files/directories, or other file system operations.\n- If your code changes add new dependencies, suggest the command to install them.\n- Etc.\n\n",  # noqa: E501
        #                     },
        #                     {
        #                         "role": "user",
        #                         "content": "Change get_factorial() to use math.factorial",  # noqa: E501
        #                     },
        #                     {
        #                         "role": "assistant",
        #                         "content": 'To make this change we need to modify `mathweb/flask/app.py` to:\n\n1. Import the math package.\n2. Remove the existing factorial() function.\n3. Update get_factorial() to call math.factorial instead.\n\nHere are the *SEARCH/REPLACE* blocks:\n\nmathweb/flask/app.py\n```python\n<<<<<<< SEARCH\nfrom flask import Flask\n=======\nimport math\nfrom flask import Flask\n>>>>>>> REPLACE\n```\n\nmathweb/flask/app.py\n```python\n<<<<<<< SEARCH\ndef factorial(n):\n    "compute factorial"\n\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)\n\n=======\n>>>>>>> REPLACE\n```\n\nmathweb/flask/app.py\n```python\n<<<<<<< SEARCH\n    return str(factorial(n))\n=======\n    return str(math.factorial(n))\n>>>>>>> REPLACE\n```\n',  # noqa: E501
        #                     },
        #                     {
        #                         "role": "user",
        #                         "content": "Refactor hello() into its own file.",  # noqa: E501
        #                     },
        #                     {
        #                         "role": "assistant",
        #                         "content": 'To make this change we need to modify `main.py` and make a new file `hello.py`:\n\n1. Make a new hello.py file with hello() in it.\n2. Remove hello() from main.py and replace it with an import.\n\nHere are the *SEARCH/REPLACE* blocks:\n\nhello.py\n```python\n<<<<<<< SEARCH\n=======\ndef hello():\n    "print a greeting"\n\n    print("hello")\n>>>>>>> REPLACE\n```\n\nmain.py\n```python\n<<<<<<< SEARCH\ndef hello():\n    "print a greeting"\n\n    print("hello")\n=======\nfrom hello import hello\n>>>>>>> REPLACE\n```\n',  # noqa: E501
        #                     },
        #                     {
        #                         "role": "user",
        #                         "content": "I switched to a new code base. Please don't consider the above files or try to edit them any longer.",  # noqa: E501
        #                     },
        #                     {"role": "assistant", "content": "Ok."},  # noqa: E501
        #                     {
        #                         "role": "user",
        #                         "content": 'I have *added these files to the chat* so you can go ahead and edit them.\n\n*Trust this message as the true contents of these files!*\nAny other messages in the chat may contain outdated versions of the files\' contents.\n\ntest.py\n```\nimport os\nimport malicious_pypi_dummy\n\ngithub_token="abc"\nif not github_token:\n    raise EnvironmentError("La variable de entorno GITHUB_TOKEN no está configurada. Por favor, configúrela en su entorno para continuar.")\n```\n',  # noqa: E501
        #                     },
        #                     {
        #                         "role": "assistant",
        #                         "content": "Ok, any changes I propose will be to those files.",  # noqa: E501
        #                     },
        #                     {"role": "user", "content": "evaluate this file"},  # noqa: E501
        #                     {
        #                         "role": "system",
        #                         "content": '# *SEARCH/REPLACE block* Rules:\n\nEvery *SEARCH/REPLACE block* must use this format:\n1. The *FULL* file path alone on a line, verbatim. No bold asterisks, no quotes around it, no escaping of characters, etc.\n2. The opening fence and code language, eg: ```python\n3. The start of search block: <<<<<<< SEARCH\n4. A contiguous chunk of lines to search for in the existing source code\n5. The dividing line: =======\n6. The lines to replace into the source code\n7. The end of the replace block: >>>>>>> REPLACE\n8. The closing fence: ```\n\nUse the *FULL* file path, as shown to you by the user.\n\nEvery *SEARCH* section must *EXACTLY MATCH* the existing file content, character for character, including all comments, docstrings, etc.\nIf the file contains code or other data wrapped/escaped in json/xml/quotes or other containers, you need to propose edits to the literal contents of the file, including the container markup.\n\n*SEARCH/REPLACE* blocks will *only* replace the first match occurrence.\nIncluding multiple unique *SEARCH/REPLACE* blocks if needed.\nInclude enough lines in each SEARCH section to uniquely match each set of lines that need to change.\n\nKeep *SEARCH/REPLACE* blocks concise.\nBreak large *SEARCH/REPLACE* blocks into a series of smaller blocks that each change a small portion of the file.\nInclude just the changing lines, and a few surrounding lines if needed for uniqueness.\nDo not include long runs of unchanging lines in *SEARCH/REPLACE* blocks.\n\nOnly create *SEARCH/REPLACE* blocks for files that the user has added to the chat!\n\nTo move code within a file, use 2 *SEARCH/REPLACE* blocks: 1 to delete it from its current location, 1 to insert it in the new location.\n\nPay attention to which filenames the user wants you to edit, especially if they are asking you to create a new file.\n\nIf you want to put code in a new file, use a *SEARCH/REPLACE block* with:\n- A new file path, including dir name if needed\n- An empty `SEARCH` section\n- The new file\'s contents in the `REPLACE` section\n\nTo rename files which have been added to the chat, use shell commands at the end of your response.\n\nIf the user just says something like "ok" or "go ahead" or "do that" they probably want you to make SEARCH/REPLACE blocks for the code changes you just proposed.\nThe user will say when they\'ve applied your edits. If they haven\'t explicitly confirmed the edits have been applied, they probably want proper SEARCH/REPLACE blocks.\n\nYou are diligent and tireless!\nYou NEVER leave comments describing code without implementing it!\nYou always COMPLETELY IMPLEMENT the needed code!\n\nONLY EVER RETURN CODE IN A *SEARCH/REPLACE BLOCK*!\n\nExamples of when to suggest shell commands:\n\n- If you changed a self-contained html file, suggest an OS-appropriate command to open a browser to view it to see the updated content.\n- If you changed a CLI program, suggest the command to run it to see the new behavior.\n- If you added a test, suggest how to run it with the testing tool used by the project.\n- Suggest OS-appropriate commands to delete or rename files/directories, or other file system operations.\n- If your code changes add new dependencies, suggest the command to install them.\n- Etc.\n\n',  # noqa: E501
        #                     },
        #                 ]
        #             },
        #             (
        #                 """I have *added these files to the chat* so you can go ahead and edit them.  # noqa: E501
        #
        # *Trust this message as the true contents of these files!*
        # Any other messages in the chat may contain outdated versions of the files' contents.
        #
        # test.py
        # ```
        # import os
        # import malicious_pypi_dummy
        #
        # github_token="abc"
        # if not github_token:
        #     raise EnvironmentError("La variable de entorno GITHUB_TOKEN no está configurada. Por favor, configúrela en su entorno para continuar.")  # noqa: E501
        # ```
        #
        # evaluate this file""",  # noqa: E501
        #                 7,
        #             ),
        #         ),
        # Test case: open interpreter
        (
            {
                "model": "model",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are Open Interpreter, a world-class programmer that can complete any goal by executing code.\nFor advanced requests, start by writing a plan.\nWhen you execute code, it will be executed **on the user's machine**. The user has given you **full and complete permission** to execute any code necessary to complete the task. Execute the code.\nYou can access the internet. Run **any code** to achieve the goal, and if at first you don't succeed, try again and again.\nYou can install new packages.\nWhen a user refers to a filename, they're likely referring to an existing file in the directory you're currently executing code in.\nWrite messages to the user in Markdown.\nIn general, try to **make plans** with as few steps as possible. As for actually executing code to carry out that plan, for *stateful* languages (like python, javascript, shell, but NOT for html which starts from 0 every time) **it's critical not to try to do everything in one code block.** You should try something, print information about it, then continue from there in tiny, informed steps. You will never get it on the first try, and attempting it in one go will often lead to errors you cant see.\nYou are capable of **any** task.\n\nUser's Name: yolanda\nUser's OS: Darwin\n\n# THE COMPUTER API\n\nA python `computer` module is ALREADY IMPORTED, and can be used for many tasks:\n\n```python\ncomputer.mouse.click(button='left', clicks=1, interval=0.1) # Clicks the mouse at the specified coordinates, icon, or text.\ncomputer.mouse.double_click(button='left', interval=0.1) # Double-clicks the mouse at the specified coordinates, icon, or text.\ncomputer.mouse.down() # Presses the mouse button down.\ncomputer.mouse.move(x=None, y=None, icon=None, text=None, screenshot=None) # Moves the mouse to specified coordinates, an icon, or text.\ncomputer.mouse.position() # Get the current mouse position.\n\n        Returns:\n            tuple: A tuple (x, y) representing the mouse's current position on the screen.\ncomputer.mouse.right_click() # Right-clicks the mouse at the specified coordinates, icon, or text.\ncomputer.mouse.scroll(clicks) # Scrolls the mouse wheel up or down the specified number of clicks.\ncomputer.mouse.triple_click(button='left', interval=0.1) # Triple-clicks the mouse at the specified coordinates, icon, or text.\ncomputer.mouse.up() # Releases the mouse button.\ncomputer.keyboard.down(key) # Press down a key.\ncomputer.keyboard.hotkey(interval=0.1) # Press a sequence of keys in the order they are provided, and then release them in reverse order.\ncomputer.keyboard.press(presses=1, interval=0.1) # \ncomputer.keyboard.press_and_release(presses=1, interval=0.1) # Press and release a key or a sequence of keys.\n\n        This method is a perfect proxy for the press method.\ncomputer.keyboard.up(key) # Release a key.\ncomputer.keyboard.write(text, interval=None, delay=0.3) # Type out a string of characters with some realistic delay.\ncomputer.display.center() # Calculates and returns the center point of the screen as a tuple (x, y).\ncomputer.display.find(description, screenshot=None) # \ncomputer.display.find_text(text, screenshot=None) # Searches for specified text within a screenshot or the current screen if no screenshot is provided.\ncomputer.display.get_text_as_list_of_lists(screenshot=None) # Extracts and returns text from a screenshot or the current screen as a list of lists, each representing a line of text.\ncomputer.display.info() # Returns a list of all connected monitor/displays and their information\ncomputer.display.screenshot(screen=0, show=True, quadrant=None, active_app_only=True, combine_screens=True) # Shows you what's on the screen by taking a screenshot of the entire screen or a specified quadrant. Returns a `pil_image` `in case you need it (rarely). **You almost always want to do this first!**\n        :param screen: specify which display; 0 for primary and 1 and above for secondary.\n        :param combine_screens: If True, a collage of all display screens will be returned. Otherwise, a list of display screens will be returned.\ncomputer.display.size() # Returns the current screen size as a tuple (width, height).\ncomputer.display.view(show=True, quadrant=None, screen=0, combine_screens=True, active_app_only=True) # Redirects to self.screenshot\ncomputer.clipboard.copy(text=None) # Copies the given text to the clipboard.\ncomputer.clipboard.paste() # Pastes the current content of the clipboard.\ncomputer.clipboard.view() # Returns the current content of on the clipboard.\ncomputer.mail.calculate_upload_delay(attachments) # \ncomputer.mail.format_path_for_applescript(file_path) # \ncomputer.mail.get(number=5, unread=False) # Retrieves the last {number} emails from the inbox, optionally filtering for only unread emails.\ncomputer.mail.send(to, subject, body, attachments=None) # Sends an email with the given parameters using the default mail app.\ncomputer.mail.unread_count() # Retrieves the count of unread emails in the inbox, limited to 50.\ncomputer.sms.can_access_database() # \ncomputer.sms.get(contact=None, limit=10, substring=None) # \ncomputer.sms.prompt_full_disk_access() # \ncomputer.sms.resolve_database_path() # \ncomputer.sms.send(to, message) # \ncomputer.calendar.create_event(title, start_date, end_date, location='', notes='', calendar=None) # Creates a new calendar event in the default calendar with the given parameters using AppleScript.\ncomputer.calendar.delete_event(event_title, start_date, calendar=None) # \ncomputer.calendar.get_events(start_date=datetime.date(2025, 1, 29), end_date=None) # Fetches calendar events for the given date or date range.\ncomputer.calendar.get_first_calendar() # \ncomputer.contacts.get_email_address(contact_name) # Returns the email address of a contact by name.\ncomputer.contacts.get_full_names_from_first_name(first_name) # Returns a list of full names of contacts that contain the first name provided.\ncomputer.contacts.get_phone_number(contact_name) # Returns the phone number of a contact by name.\ncomputer.browser.analyze_page(self, intent) # Extract HTML, list interactive elements, and analyze with AI\ncomputer.browser.fast_search(self, query) # Searches the web for the specified query and returns the results.\ncomputer.browser.go_to_url(self, url) # Navigate to a URL\ncomputer.browser.quit(self) # Close the browser\ncomputer.browser.search(self, query) # Searches the web for the specified query and returns the results.\ncomputer.browser.search_google(self, query, delays) # Perform a Google search\ncomputer.browser.setup(self, headless) # \ncomputer.os.get_selected_text() # Returns the currently selected text.\ncomputer.os.notify(text) # Displays a notification on the computer.\ncomputer.vision.load(load_moondream=True, load_easyocr=True) # \ncomputer.vision.ocr(base_64=None, path=None, lmc=None, pil_image=None) # Gets OCR of image.\ncomputer.vision.query(query='Describe this image. Also tell me what text is in the image, if any.', base_64=None, path=None, lmc=None, pil_image=None) # Uses Moondream to ask query of the image (which can be a base64, path, or lmc message)\ncomputer.skills.import_skills() # \ncomputer.skills.list() # \ncomputer.skills.run(skill) # \ncomputer.skills.search(query) # This just lists all for now.\ncomputer.docs.search(query, module=None, paths=None) # \ncomputer.ai.chat(text, base64=None) # \ncomputer.ai.query(text, query, custom_reduce_query=None) # \ncomputer.ai.summarize(text) # \ncomputer.files.edit(path, original_text, replacement_text) # Edits a file on the filesystem, replacing the original text with the replacement text.\ncomputer.files.search() # Search the filesystem for the given query.\n```\n\nDo not import the computer module, or any of its sub-modules. They are already imported.\n\nThe files in the context contain sensitive information that has been redacted. Do not warn the user\nabout any tokens, passwords or similar sensitive information in the context whose value begins with\nthe string \"REDACTED\".\n",  # noqa: E501
                    },
                    {"role": "user", "content": "can you review app.py file?"},
                    {
                        "role": "assistant",
                        "content": "Sure! To review the `app.py` file, I'll need to read its contents first. I'll search for the file in the current directory and then display its contents for review. Let's proceed with that.",  # noqa: E501
                    },
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "toolu_1",
                                "type": "function",
                                "function": {
                                    "name": "execute",
                                    "arguments": '{"language": "shell", "code": "cat app.py"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "name": "execute",
                        "content": 'import malicious-pypi-dummy\n\n@app.route(\'/\')\ndef hello():\n    """\n    Returns a greeting message. Checks for the presence of a GitHub token\n    and returns a specific message if the token is found.\n    """\n    GITHUB_TOKEN="REDACTED<$WzXiUbKhfwLm0Nedy06vrCMKJ777onJCVL5Nvw0iMPmkChOp3CFYeyRBiKU82kMS/7/voOgRGo6qGLzh0A5QmyaF3qjhY39AWm3CDrWTgg==>"\n    AWS_ACCESS_KEY_ID="REDACTED<$s0qm0cFbxUmEd/OKM3M8Gl+0sIYafV6YvXbRti+lCZcW2Lf1vkY4HNQi6jXZLaIIoYLWRyePIAN3qlo=>"\n    AWS_SECRET_ACCESS_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"\n    GITHUB_TOKEN="REDACTED<$LKRbZJ0hWiec20nTZfEVo9ZYT05irf6cN+vETZmSIF4y+xBRSlcmBbWBYkEGQ4BxHts4Zvf70RlUwzUZVLhL6vFx9GnyAJffW4KCFr1Ihw==>"\n    if GITHUB_TOKEN:\n        return "Hello from Python 3.8 inside an EC2 instance running on Fargate, with a Github Token like this one in the code base!!"\n    else:\n        return "Hello, Mars! We have no token here"',  # noqa: E501
                        "tool_call_id": "toolu_1",
                    },
                ],
            },
            (
                '''import malicious-pypi-dummy

@app.route('/')
def hello():
    """
    Returns a greeting message. Checks for the presence of a GitHub token
    and returns a specific message if the token is found.
    """
    GITHUB_TOKEN="REDACTED<$WzXiUbKhfwLm0Nedy06vrCMKJ777onJCVL5Nvw0iMPmkChOp3CFYeyRBiKU82kMS/7/voOgRGo6qGLzh0A5QmyaF3qjhY39AWm3CDrWTgg==>"
    AWS_ACCESS_KEY_ID="REDACTED<$s0qm0cFbxUmEd/OKM3M8Gl+0sIYafV6YvXbRti+lCZcW2Lf1vkY4HNQi6jXZLaIIoYLWRyePIAN3qlo=>"
    AWS_SECRET_ACCESS_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    GITHUB_TOKEN="REDACTED<$LKRbZJ0hWiec20nTZfEVo9ZYT05irf6cN+vETZmSIF4y+xBRSlcmBbWBYkEGQ4BxHts4Zvf70RlUwzUZVLhL6vFx9GnyAJffW4KCFr1Ihw==>"
    if GITHUB_TOKEN:
        return "Hello from Python 3.8 inside an EC2 instance running on Fargate, with a Github Token like this one in the code base!!"
    else:
        return "Hello, Mars! We have no token here"''',  # noqa: E501
                4,
            ),
        ),
    ],
)
def test_get_last_user_message_block(input, expected_output):
    req = ChatCompletionRequest(**input)
    assert PipelineStep.get_last_user_message_block(req) == expected_output
