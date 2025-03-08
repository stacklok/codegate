import asyncio
import copy
import json
import os
import re
import sys
from typing import Any, Dict, Optional, Tuple

import requests
import structlog
import yaml
from checks import CheckLoader, CodeGateEnrichment
from dotenv import find_dotenv, load_dotenv
from requesters import RequesterFactory

logger = structlog.get_logger("codegate")


class CodegateTestRunner:
    def __init__(self):
        self.requester_factory = RequesterFactory()
        self.failed_tests = []  # Track failed tests

    def call_provider(
        self, url: str, headers: dict, data: dict, provider: str, method: str = "POST"
    ) -> Optional[requests.Response]:
        logger.debug(f"Creating requester for provider: {provider}")
        requester = self.requester_factory.create_requester(provider)
        logger.debug(f"Using requester type: {requester.__class__.__name__}")

        logger.debug(f"Making request to URL: {url}")
        logger.debug(f"Headers: {headers}")
        logger.debug(f"Data: {data}")

        response = requester.make_request(url, headers, data, method=method)

        # Enhanced response logging
        if response is not None:

            if response.status_code not in [200, 201, 204]:
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
                    elif "message" in json_line and isinstance(json_line["message"], str):
                         # "messages" is a raw string
                        message_content = json_line["message"]
                    elif "message" in json_line:
                         # "messages" is a structured object
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
        data = json.loads(test["data"])
        codegate_url = test["url"]
        streaming = data.get("stream", False)
        provider = test["provider"]
        logger.info(f"Starting test: {test_name}")

        # Call Codegate
        response = self.call_provider(codegate_url, test_headers, data, provider)
        if not response:
            logger.error(f"Test {test_name} failed: No response received")
            return False

        # Call model directly if specified
        direct_response = None
        if test.get(CodeGateEnrichment.KEY) is not None:
            direct_provider_url = test.get(CodeGateEnrichment.KEY)["provider_url"]
            direct_response = self.call_provider(
                direct_provider_url, test_headers, data, "not-codegate"
            )
            if not direct_response:
                logger.error(f"Test {test_name} failed: No direct response received")
                return False

        # Debug response info
        logger.debug(f"Response status: {response.status_code}")
        logger.debug(f"Response headers: {dict(response.headers)}")

        try:
            parsed_response = self.parse_response_message(response, streaming=streaming)
            logger.debug(f"Response message: {parsed_response}")

            if direct_response:
                # Dirty hack to pass direct response to checks
                test["direct_response"] = self.parse_response_message(
                    direct_response, streaming=streaming
                )
                logger.debug(f"Direct response message: {test['direct_response']}")

            # Load appropriate checks for this test
            checks = CheckLoader.load(test)

            # Run all checks
            all_passed = True
            for check in checks:
                logger.info(f"Running check: {check.__class__.__name__}")
                passed_check = await check.run_check(parsed_response, test)
                logger.info(
                    f"Check {check.__class__.__name__} {'passed' if passed_check else 'failed'}"
                )
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

    async def _get_testcases(
        self, testcases_dict: Dict, test_names: Optional[list[str]] = None
    ) -> Dict[str, Dict[str, str]]:
        testcases: Dict[str, Dict[str, str]] = testcases_dict["testcases"]

        # Filter testcases by provider and test names
        if test_names:
            filtered_testcases = {}

            # Iterate over the original testcases and only keep the ones that match the
            # specified test names
            for test_id, test_data in testcases.items():
                if test_data.get("name", "").lower() not in [t.lower() for t in test_names]:
                    continue

                filtered_testcases[test_id] = test_data

            testcases = filtered_testcases
        return testcases

    async def _setup_muxing(
        self, provider: str, muxing_config: Optional[Dict]
    ) -> Optional[Tuple[str, str]]:
        """
        Muxing setup. Create the provider endpoints and the muxing rules

        Return
        """
        # The muxing section was not found in the testcases.yaml file. Nothing to do.
        if not muxing_config:
            return

        # Create the provider endpoint
        provider_endpoint = muxing_config.get("provider_endpoint")
        try:
            data_with_api_keys = self.replace_env_variables(provider_endpoint["data"], os.environ)
            response_create_provider = self.call_provider(
                provider=provider,
                url=provider_endpoint["url"],
                headers=provider_endpoint["headers"],
                data=json.loads(data_with_api_keys),
            )
            created_provider_endpoint = response_create_provider.json()
        except Exception as e:
            logger.warning(f"Could not setup provider endpoint for muxing: {e}")
            return
        logger.info("Created provider endpoint for muixing")

        muxes_rules: Dict[str, Any] = muxing_config.get("muxes", {})
        try:
            # We need to first update all the muxes with the provider_id
            for mux in muxes_rules.get("rules", []):
                mux["provider_id"] = created_provider_endpoint["id"]

            # The endpoint actually takes a list
            self.call_provider(
                provider=provider,
                url=muxes_rules["url"],
                headers=muxes_rules["headers"],
                data=muxes_rules.get("rules", []),
                method="PUT",
            )
        except Exception as e:
            logger.warning(f"Could not setup muxing rules: {e}")
            return
        logger.info("Created muxing rules")

        return muxing_config["mux_url"], muxing_config["trimm_from_testcase_url"]

    async def _augment_testcases_with_muxing(
        self, testcases: Dict, mux_url: str, trimm_from_testcase_url: str
    ) -> Dict:
        """
        Augment the testcases with the muxing information. Copy the testcases
        and execute them through the muxing endpoint.
        """
        test_cases_with_muxing = copy.deepcopy(testcases)
        for test_id, test_data in testcases.items():
            # Replace the provider in the URL with the muxed URL
            rest_of_path = test_data["url"].replace(trimm_from_testcase_url, "")
            new_url = f"{mux_url}{rest_of_path}"
            new_test_data = copy.deepcopy(test_data)
            new_test_data["url"] = new_url
            new_test_id = f"{test_id}_muxed"
            test_cases_with_muxing[new_test_id] = new_test_data

        logger.info("Augmented testcases with muxing")
        return test_cases_with_muxing

    async def _setup(
        self, testcases_file: str, provider: str, test_names: Optional[list[str]] = None
    ) -> Tuple[Dict, Dict]:
        with open(testcases_file, "r") as f:
            testcases_dict: Dict = yaml.safe_load(f)

        headers = testcases_dict["headers"]
        testcases = await self._get_testcases(testcases_dict, test_names)
        muxing_result = await self._setup_muxing(provider, testcases_dict.get("muxing", {}))
        # We don't have any muxing setup, return the headers and testcases
        if not muxing_result:
            return headers, testcases

        mux_url, trimm_from_testcase_url = muxing_result
        test_cases_with_muxing = await self._augment_testcases_with_muxing(
            testcases, mux_url, trimm_from_testcase_url
        )

        return headers, test_cases_with_muxing

    async def run_tests(
        self,
        testcases_file: str,
        provider: str,
        test_names: Optional[list[str]] = None,
    ) -> bool:
        headers, testcases = await self._setup(testcases_file, provider, test_names)
        if not testcases:
            logger.warning(
                f"No tests found for provider {provider} in file: {testcases_file} "
                f"and specific testcases: {test_names}"
            )
            return True  # No tests is not a failure

        test_count = len(testcases)
        logging_msg = f"Running {test_count} tests for provider {provider}"
        if test_names:
            logging_msg += f" and test names: {', '.join(test_names)}"
        logger.info(logging_msg)

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

        # Run tests for the provider. The provider has already been selected when
        # reading the testcases.yaml file.
        logger.info(f"Running tests for provider: {provider}")
        provider_tests_passed = await test_runner.run_tests(
            provider_test_file,
            provider=provider,
            test_names=test_names,
        )
        all_tests_passed = all_tests_passed and provider_tests_passed

    # Exit with status code 1 if any tests failed
    if not all_tests_passed:
        sys.exit(1)
    logger.info("All tests passed")


if __name__ == "__main__":
    asyncio.run(main())
