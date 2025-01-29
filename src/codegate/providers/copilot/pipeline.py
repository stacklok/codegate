import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import structlog
from litellm import ModelResponse
from litellm.types.llms.openai import ChatCompletionRequest
from litellm.types.utils import Delta, StreamingChoices

from codegate.clients.clients import ClientType
from codegate.pipeline.base import PipelineContext, PipelineResult, SequentialPipelineProcessor
from codegate.pipeline.factory import PipelineFactory
from codegate.providers.normalizer.completion import CompletionNormalizer

logger = structlog.get_logger("codegate")


class CopilotPipeline(ABC):
    """
    A CopilotPipeline puts together a normalizer to be able to pass
    a request to the pipeline in a normalized format, and a pipeline
    factory to create the pipeline itself and run the request
    """

    def __init__(self, pipeline_factory: PipelineFactory):
        self.pipeline_factory = pipeline_factory
        self.instance = self._create_pipeline()
        self.normalizer = self._create_normalizer()
        self.provider_name = "openai"

    @abstractmethod
    def _create_normalizer(self):
        """Each strategy defines which normalizer to use"""
        pass

    @abstractmethod
    def _create_pipeline(self) -> SequentialPipelineProcessor:
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

    @staticmethod
    def _create_shortcut_response(result: PipelineResult, model: str) -> bytes:
        response = ModelResponse(
            choices=[
                StreamingChoices(
                    finish_reason="stop",
                    index=0,
                    delta=Delta(content=result.response.content, role="assistant"),
                )
            ],
            created=int(time.time()),
            model=model,
            stream=True,
        )
        body = response.model_dump_json(exclude_none=True, exclude_unset=True).encode()
        return body

    async def process_body(
        self,
        headers: list[str],
        body: bytes,
    ) -> Tuple[bytes, PipelineContext | None]:
        """Common processing logic for all strategies"""
        try:
            normalized_body = self.normalizer.normalize(body)
        except Exception as e:
            logger.error(f"Pipeline processing error: {e}")
            return body, None

        headers_dict = {}
        for header in headers:
            try:
                name, value = header.split(":", 1)
                headers_dict[name.strip().lower()] = value.strip()
            except ValueError:
                continue

        try:
            result = await self.instance.process_request(
                request=normalized_body,
                provider=self.provider_name,
                model=normalized_body.get("model", "gpt-4o-mini"),
                api_key=headers_dict.get("authorization", "").replace("Bearer ", ""),
                api_base="https://" + headers_dict.get("host", ""),
                extra_headers=CopilotPipeline._get_copilot_headers(headers_dict),
            )
        except Exception as e:
            logger.error(f"Pipeline processing error: {e}")
            return body, None

        if result.context.shortcut_response:
            try:
                # Return shortcut response to the user
                body = CopilotPipeline._create_shortcut_response(
                    result, normalized_body.get("model", "gpt-4o-mini")
                )
                logger.info(f"Pipeline created shortcut response: {body}")
                return body, result.context
            except Exception as e:
                logger.error(f"Pipeline processing error: {e}")
                return body, None

        elif result.request:
            try:
                # the pipeline did modify the request, return to the user
                # in the original LLM format
                body = self.normalizer.denormalize(result.request)
                # Uncomment the below to debug the request
                # logger.debug(f"Pipeline processed request: {body}")

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
        normalized_data = ChatCompletionRequest(**json_body)

        # This would normally be the required to get the token usage with OpenAI models.
        # However the response comes back empty with Copilot. Commenting for the moment.
        # It's not critical since Copilot charges a fixed rate and not based in tokens.
        # if normalized_data.get("stream", False):
        #     normalized_data["stream_options"] = {"include_usage": True}

        return normalized_data

    def denormalize(self, request_from_pipeline: ChatCompletionRequest) -> bytes:
        return json.dumps(request_from_pipeline).encode()


class CopilotFimPipeline(CopilotPipeline):
    """
    A pipeline for the FIM format used by Copilot. Combines the normalizer for the FIM
    format and the FIM pipeline used by all providers.
    """

    def __init__(self, pipeline_factory: PipelineFactory):
        super().__init__(pipeline_factory)

    def _create_normalizer(self):
        return CopilotFimNormalizer()

    def _create_pipeline(self) -> SequentialPipelineProcessor:
        return self.pipeline_factory.create_fim_pipeline(ClientType.COPILOT)


class CopilotChatPipeline(CopilotPipeline):
    """
    A pipeline for the Chat format used by Copilot. Combines the normalizer for the FIM
    format and the FIM pipeline used by all providers.
    """

    def __init__(self, pipeline_factory: PipelineFactory):
        super().__init__(pipeline_factory)

    def _create_normalizer(self):
        return CopilotChatNormalizer()

    def _create_pipeline(self) -> SequentialPipelineProcessor:
        return self.pipeline_factory.create_input_pipeline(ClientType.COPILOT)
