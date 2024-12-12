import json
from abc import ABC, abstractmethod
from typing import Dict

import structlog
from litellm.types.llms.openai import ChatCompletionRequest

from codegate.pipeline.base import PipelineContext
from codegate.providers.normalizer.completion import CompletionNormalizer

logger = structlog.get_logger("codegate")


class CopilotPipeline(ABC):
    """
    A CopilotPipeline puts together a normalizer to be able to pass
    a request to the pipeline in a normalized format, and a pipeline
    factory to create the pipeline itself and run the request
    """

    def __init__(self, pipeline_factory):
        self.pipeline_factory = pipeline_factory
        self.normalizer = self._create_normalizer()
        self.provider_name = "copilot"

    @abstractmethod
    def _create_normalizer(self):
        """Each strategy defines which normalizer to use"""
        pass

    @abstractmethod
    def create_pipeline(self):
        """Each strategy defines which pipeline to create"""
        pass

    @staticmethod
    def _request_id(headers: list[str]) -> str:
        """Extracts the request ID from the headers"""
        for header in headers:
            if header.startswith("x-request-id"):
                print(f"Request ID found in headers: {header}")
                return header.split(":")[1].strip()
        print("No request ID found in headers")
        return ""

    @staticmethod
    def _get_copilot_headers(headers: Dict[str, str]) -> Dict[str, str]:
        copilot_header_names = [
            "copilot-integration-id",
            "editor-plugin-version",
            "editor-version",
            "openai-intent",
            "openai-organization",
            "user-agent",
            "vscode-machineid",
            "vscode-sessionid",
            "x-github-api-version",
            "x-request-id",
        ]
        copilot_headers = {}
        for a_name in copilot_header_names:
            copilot_headers[a_name] = headers.get(a_name, "")

        return copilot_headers

    async def process_body(self, headers: list[str], body: bytes) -> (bytes, PipelineContext):
        """Common processing logic for all strategies"""
        try:
            normalized_body = self.normalizer.normalize(body)

            headers_dict = {}
            for header in headers:
                try:
                    name, value = header.split(":", 1)
                    headers_dict[name.strip().lower()] = value.strip()
                except ValueError:
                    continue

            pipeline = self.create_pipeline()
            result = await pipeline.process_request(
                request=normalized_body,
                provider=self.provider_name,
                prompt_id=self._request_id(headers),
                model=normalized_body.get("model", "gpt-4o-mini"),
                api_key=headers_dict.get("authorization", "").replace("Bearer ", ""),
                api_base="https://" + headers_dict.get("host", ""),
                extra_headers=CopilotPipeline._get_copilot_headers(headers_dict),
            )

            if result.request:
                # the pipeline did modify the request, return to the user
                # in the original LLM format
                body = self.normalizer.denormalize(result.request)
                logger.info(f"Pipeline processed request: {body}")

            return body, result.context
        except Exception as e:
            logger.error(f"Pipeline processing error: {e}")
            return body, None


class CopilotFimNormalizer:
    """
    A custom normalizer for the FIM format used by Copilot
    We reuse the normalizer for "prompt" format, but we need to
    load the body first and then encode on the way back.
    """

    def __init__(self):
        self._completion_normalizer = CompletionNormalizer()

    def normalize(self, body: bytes) -> ChatCompletionRequest:
        json_body = json.loads(body)
        return self._completion_normalizer.normalize(json_body)

    def denormalize(self, request_from_pipeline: ChatCompletionRequest) -> bytes:
        normalized_json_body = self._completion_normalizer.denormalize(request_from_pipeline)
        return json.dumps(normalized_json_body).encode()


class CopilotChatNormalizer:
    """
    A custom normalizer for the chat format used by Copilot
    The requests are already in the OpenAI format, we just need
    to unmarshall them and marshall them back.
    """

    def normalize(self, body: bytes) -> ChatCompletionRequest:
        json_body = json.loads(body)
        return ChatCompletionRequest(**json_body)

    def denormalize(self, request_from_pipeline: ChatCompletionRequest) -> bytes:
        return json.dumps(request_from_pipeline).encode()


class CopilotFimPipeline(CopilotPipeline):
    """
    A pipeline for the FIM format used by Copilot. Combines the normalizer for the FIM
    format and the FIM pipeline used by all providers.
    """

    def _create_normalizer(self):
        return CopilotFimNormalizer()

    def create_pipeline(self):
        return self.pipeline_factory.create_fim_pipeline()


class CopilotChatPipeline(CopilotPipeline):
    """
    A pipeline for the Chat format used by Copilot. Combines the normalizer for the FIM
    format and the FIM pipeline used by all providers.
    """

    def _create_normalizer(self):
        return CopilotChatNormalizer()

    def create_pipeline(self):
        return self.pipeline_factory.create_input_pipeline()
