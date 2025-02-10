import asyncio
import json
import os
import re
import sys
from typing import Optional

import requests
import structlog
import yaml
from checks import CheckLoader
from dotenv import find_dotenv, load_dotenv
from requesters import RequesterFactory

logger = structlog.get_logger("codegate")


class CodegateTestRunner:
    def __init__(self):
        self.requester_factory = RequesterFactory()
        self.failed_tests = []  # Track failed tests

    def call_codegate(
        self, url: str, headers: dict, data: dict, provider: str
    ) -> Optional[requests.Response]:
        logger.debug(f"Creating requester for provider: {provider}")
        requester = self.requester_factory.create_requester(provider)
        logger.debug(f"Using requester type: {requester.__class__.__name__}")

        logger.debug(f"Making request to URL: {url}")
        logger.debug(f"Headers: {headers}")
        logger.debug(f"Data: {data}")

        response = requester.make_request(url, headers, data)

        # Enhanced response logging
        if response is not None:

            if response.status_code != 200:
                logger.debug(f"Response error status: {response.status_code}")
                logger.debug(f"Response error headers: {dict(response.headers)}")
                try:
                    error_content = response.json()
                    logger.error(f"Request error as JSON: {error_content}")
                except ValueError:
                    # If not JSON, try to get raw text
                    logger.error(f"Raw request error: {response.text}")
        else:
            logger.error("No response received")

        return response

    @staticmethod
    def parse_response_message(response, streaming=True):
        response_message = ""
        try:
            if streaming:
                for line in response.iter_lines():
                    decoded_line = line.decode("utf-8").strip()
                    if (
                        not decoded_line
                        or decoded_line.startswith("event:")
                        or "message_start" in decoded_line
                    ):
                        continue

                    if "DONE" in decoded_line or "message_stop" in decoded_line:
                        break

                    decoded_line = decoded_line.replace("data:", "").strip()
                    json_line = json.loads(decoded_line)
                    message_content = None
                    # Handle both chat and FIM responses
                    if "choices" in json_line:
                        choice = json_line["choices"][0]
                        # Break if the conversation is over
                        if choice.get("finish_reason") == "stop":
                            break
                        # Handle chat responses
                        if "delta" in choice:
                            delta = choice["delta"]
                            if "content" in delta and delta["content"] is not None:
                                message_content = delta["content"]
                        # Handle FIM responses
                        elif "text" in choice:
                            text = choice["text"]
                            if text is not None:
                                message_content = text
                    elif "delta" in json_line:
                        message_content = json_line["delta"].get("text", "")
                    elif "message" in json_line:
                        message_content = json_line["message"].get("content", "")
                    elif "response" in json_line:
                        message_content = json_line.get("response", "")

                    if message_content is not None:
                        response_message += message_content
            else:
                if "choices" in response.json():
                    response_message = response.json()["choices"][0]["message"].get("content", "")
                elif "content" in response.json():
                    response_message = response.json()["content"][0].get("text", "")

        except Exception as e:
            logger.exception("An error occurred: %s", e)

        # Remove any trailing newlines and return
        return response_message.strip()

    @staticmethod
    def replace_env_variables(input_string, env):
        """
        Replace occurrences of strings starting with ENV in the input string
        with their corresponding values from the env dictionary.

        Args:
            input_string (str): The string containing patterns starting with ENV.
            env (dict): A dictionary mapping ENV keys to their replacement values.

        Returns:
            str: The modified string with ENV patterns replaced.
        """

        def replacement(match):
            key = match.group(0)  # Full match (e.g., ENV_VLLM_KEY)
            return env.get(key, key)  # Replace with value if key exists, else keep original

        # Match patterns starting with ENV (alphanumeric and underscore after ENV)
        pattern = r"ENV\w*"
        return re.sub(pattern, replacement, input_string)

    async def run_test(self, test: dict, test_headers: dict) -> bool:
        test_name = test["name"]
        url = test["url"]
        data = json.loads(test["data"])
        streaming = data.get("stream", False)
        provider = test["provider"]

        response = self.call_codegate(url, test_headers, data, provider)
        if not response:
            logger.error(f"Test {test_name} failed: No response received")
            return False

        # Debug response info
        logger.debug(f"Response status: {response.status_code}")
        logger.debug(f"Response headers: {dict(response.headers)}")

        try:
            parsed_response = self.parse_response_message(response, streaming=streaming)
            logger.debug(f"Response message: {parsed_response}")

            # Load appropriate checks for this test
            checks = CheckLoader.load(test)

            # Run all checks
            all_passed = True
            for check in checks:
                passed_check = await check.run_check(parsed_response, test)
                if not passed_check:
                    all_passed = False

            if not all_passed:
                self.failed_tests.append(test_name)

            logger.info(f"Test {test_name} {'passed' if all_passed else 'failed'}")
            return all_passed

        except Exception as e:
            logger.exception("Could not parse response: %s", e)
            self.failed_tests.append(test_name)
            return False

    async def run_tests(
        self,
        testcases_file: str,
        providers: Optional[list[str]] = None,
        test_names: Optional[list[str]] = None,
    ) -> bool:
        with open(testcases_file, "r") as f:
            tests = yaml.safe_load(f)

        headers = tests["headers"]
        testcases = tests["testcases"]

        if providers or test_names:
            filtered_testcases = {}

            for test_id, test_data in testcases.items():
                if providers:
                    if test_data.get("provider", "").lower() not in [p.lower() for p in providers]:
                        continue

                if test_names:
                    if test_data.get("name", "").lower() not in [t.lower() for t in test_names]:
                        continue

                filtered_testcases[test_id] = test_data

            testcases = filtered_testcases

            if not testcases:
                filter_msg = []
                if providers:
                    filter_msg.append(f"providers: {', '.join(providers)}")
                if test_names:
                    filter_msg.append(f"test names: {', '.join(test_names)}")
                logger.warning(f"No tests found for {' and '.join(filter_msg)}")
                return True  # No tests is not a failure

        test_count = len(testcases)
        filter_msg = []
        if providers:
            filter_msg.append(f"providers: {', '.join(providers)}")
        if test_names:
            filter_msg.append(f"test names: {', '.join(test_names)}")

        logger.info(
            f"Running {test_count} tests"
            + (f" for {' and '.join(filter_msg)}" if filter_msg else "")
        )

        all_tests_passed = True
        for test_id, test_data in testcases.items():
            test_headers = headers.get(test_data["provider"], {})
            test_headers = {
                k: self.replace_env_variables(v, os.environ) for k, v in test_headers.items()
            }
            test_passed = await self.run_test(test_data, test_headers)
            if not test_passed:
                all_tests_passed = False

        if not all_tests_passed:
            logger.error(f"The following tests failed: {', '.join(self.failed_tests)}")

        return all_tests_passed


async def main():
    load_dotenv(find_dotenv())
    test_runner = CodegateTestRunner()

    # Get providers and test names from environment variables
    providers_env = os.environ.get("CODEGATE_PROVIDERS")
    test_names_env = os.environ.get("CODEGATE_TEST_NAMES")

    # Base directory for all test cases
    base_test_dir = "./tests/integration"

    # Get list of provider directories
    available_providers = []
    try:
        available_providers = [
            d for d in os.listdir(base_test_dir) if os.path.isdir(os.path.join(base_test_dir, d))
        ]
    except FileNotFoundError:
        logger.error(f"Test directory {base_test_dir} not found")
        sys.exit(1)

    # Filter providers if specified in environment
    selected_providers = None
    if providers_env:
        selected_providers = [p.strip() for p in providers_env.split(",") if p.strip()]
        # Validate selected providers exist
        invalid_providers = [p for p in selected_providers if p not in available_providers]
        if invalid_providers:
            logger.error(f"Invalid providers specified: {', '.join(invalid_providers)}")
            logger.error(f"Available providers: {', '.join(available_providers)}")
            sys.exit(1)
    else:
        selected_providers = available_providers

    # Get test names if specified
    test_names = None
    if test_names_env:
        test_names = [t.strip() for t in test_names_env.split(",") if t.strip()]

    # Run tests for each provider
    all_tests_passed = True
    for provider in selected_providers:
        provider_test_file = os.path.join(base_test_dir, provider, "testcases.yaml")

        if not os.path.exists(provider_test_file):
            logger.warning(f"No testcases.yaml found for provider {provider}")
            continue

        logger.info(f"Running tests for provider: {provider}")
        provider_tests_passed = await test_runner.run_tests(
            provider_test_file,
            providers=[provider],  # Only run tests for current provider
            test_names=test_names,
        )
        all_tests_passed = all_tests_passed and provider_tests_passed

    # Exit with status code 1 if any tests failed
    if not all_tests_passed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
