import datetime
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import structlog
from pydantic import BaseModel

from codegate.clients.clients import ClientType
from codegate.db.models import Alert, AlertSeverity, Output, Prompt
from codegate.extract_snippets.message_extractor import CodeSnippet
from codegate.pipeline.secrets.manager import SecretsManager
from codegate.types.common import ChatCompletionRequest, ModelResponse

logger = structlog.get_logger("codegate")


@dataclass
class PipelineSensitiveData:
    manager: SecretsManager
    session_id: str
    api_key: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    api_base: Optional[str] = None

    def secure_cleanup(self):
        """Securely cleanup sensitive data for this session"""
        if self.manager is None or self.session_id == "":
            return

        self.manager.cleanup_session(self.session_id)
        self.session_id = ""

        # Securely wipe the API key using the same method as secrets manager
        if self.api_key is not None:
            api_key_bytes = bytearray(self.api_key.encode())
            self.manager.crypto.wipe_bytearray(api_key_bytes)
            self.api_key = None

        self.model = None


@dataclass
class PipelineContext:
    metadata: Dict[str, Any] = field(default_factory=dict)
    sensitive: Optional[PipelineSensitiveData] = field(default_factory=lambda: None)
    alerts_raised: List[Alert] = field(default_factory=list)
    prompt_id: Optional[str] = field(default_factory=lambda: None)
    input_request: Optional[Prompt] = field(default_factory=lambda: None)
    output_responses: List[Output] = field(default_factory=list)
    shortcut_response: bool = False
    bad_packages_found: bool = False
    secrets_found: bool = False
    client: ClientType = ClientType.GENERIC

    def add_alert(
        self,
        step_name: str,
        severity_category: AlertSeverity = AlertSeverity.INFO,
        code_snippet: Optional[CodeSnippet] = None,
        trigger_string: Optional[str] = None,
    ) -> None:
        """
        Add an alert to the pipeline step alerts_raised.
        """
        if self.prompt_id is None:
            self.prompt_id = str(uuid.uuid4())

        if not code_snippet and not trigger_string:
            logger.warning("No code snippet or trigger string provided for alert. Will not create")
            return

        code_snippet_str = code_snippet.model_dump_json() if code_snippet else None

        self.alerts_raised.append(
            Alert(
                id=str(uuid.uuid4()),
                prompt_id=self.prompt_id,
                code_snippet=code_snippet_str,
                trigger_string=trigger_string,
                trigger_type=step_name,
                trigger_category=severity_category.value,
                timestamp=datetime.datetime.now(datetime.timezone.utc),
            )
        )
        # Uncomment the below to debug
        # logger.debug(f"Added alert to context: {self.alerts_raised[-1]}")

    def add_input_request(
        self, normalized_request: ChatCompletionRequest, is_fim_request: bool, provider: str
    ) -> None:
        try:
            if self.prompt_id is None:
                self.prompt_id = str(uuid.uuid4())

            request_str = normalized_request.json()

            self.input_request = Prompt(
                id=self.prompt_id,
                timestamp=datetime.datetime.now(datetime.timezone.utc),
                provider=provider,
                type="fim" if is_fim_request else "chat",
                request=request_str,
                workspace_id=None,
            )
            # Uncomment the below to debug the input
            # logger.debug(f"Added input request to context: {self.input_request}")
        except Exception as e:
            logger.warning(f"Failed to serialize input request: {normalized_request}", error=str(e))

    def add_output(self, model_response: ModelResponse) -> None:
        try:
            if self.prompt_id is None:
                logger.warning(f"Tried to record output without response: {model_response}")
                return

            if isinstance(model_response, BaseModel):
                output_str = model_response.model_dump_json(exclude_none=True, exclude_unset=True)
            else:
                output_str = json.dumps(model_response)

            self.output_responses.append(
                Output(
                    id=str(uuid.uuid4()),
                    prompt_id=self.prompt_id,
                    timestamp=datetime.datetime.now(datetime.timezone.utc),
                    output=output_str,
                )
            )
            # Uncomment the below to debug the responses
            # logger.debug(f"Added output to context: {self.output_responses[-1]}")
        except Exception as e:
            logger.error(f"Failed to serialize output: {model_response}", error=str(e))
            return


@dataclass
class PipelineResponse:
    """Response generated by a pipeline step"""

    content: str
    step_name: str  # The name of the pipeline step that generated this response
    model: str  # Taken from the original request's model field


@dataclass
class PipelineResult:
    """
    Represents the result of a pipeline operation.
    Either contains a modified request to continue processing,
    or a response to return to the client.
    """

    request: Optional[ChatCompletionRequest] = None
    response: Optional[PipelineResponse] = None
    context: Optional[PipelineContext] = None
    error_message: Optional[str] = None

    def shortcuts_processing(self) -> bool:
        """Returns True if this result should end pipeline processing"""
        return self.response is not None or self.error_message is not None

    @property
    def success(self) -> bool:
        """Returns True if the pipeline step completed without errors"""
        return self.error_message is None


class PipelineStep(ABC):
    """Base class for all pipeline steps in the processing chain."""

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Returns the name of the pipeline step.

        Returns:
            str: A unique identifier for this pipeline step
        """
        pass

    @staticmethod
    def get_last_user_message(
        request: ChatCompletionRequest,
    ) -> Optional[tuple[str, int]]:
        """
        Get the last user message and its index from the request.

        Args:
            request (ChatCompletionRequest): The chat completion request to process

        Returns:
            Optional[tuple[str, int]]: A tuple containing the message content and
                                       its index, or None if no user message is found
        """
        msg = request.last_user_message()

        if msg is None:
            return None

        # unpack the tuple
        msg, idx = msg
        return "".join([txt for txt in msg.text()]), idx


    @staticmethod
    def get_last_user_message_block(
        request: ChatCompletionRequest,
        client: ClientType = ClientType.GENERIC,
    ) -> Optional[tuple[str, int]]:
        """
        Get the last block of consecutive 'user' messages from the request.

        Args:
            request (ChatCompletionRequest): The chat completion request to process
            client (ClientType): The client type to consider when processing the request

        Returns:
            Optional[str, int]: A string containing all consecutive user messages in the
                        last user message block, separated by newlines, or None if
                        no user message block is found.
                        Index of the first message detected in the block.
        """
        user_messages = []
        last_idx = -1
        for msg, idx in request.last_user_block():
            for content in msg.text():
                user_messages.append(content)
                last_idx = idx

        if user_messages == []:
            return None
        return "\n".join(reversed(user_messages)), last_idx

    @abstractmethod
    async def process(
        self, request: ChatCompletionRequest, context: PipelineContext
    ) -> PipelineResult:
        """Process a request and return either modified request or response stream"""
        pass


class InputPipelineInstance:
    def __init__(
        self,
        pipeline_steps: List[PipelineStep],
        secret_manager: SecretsManager,
        is_fim: bool,
        client: ClientType = ClientType.GENERIC,
    ):
        self.pipeline_steps = pipeline_steps
        self.secret_manager = secret_manager
        self.is_fim = is_fim
        self.context = PipelineContext(client=client)

        # we create the sesitive context here so that it is not shared between individual requests
        # TODO: could we get away with just generating the session ID for an instance?
        self.context.sensitive = PipelineSensitiveData(
            manager=self.secret_manager,
            session_id=str(uuid.uuid4()),
        )
        self.context.metadata["is_fim"] = is_fim

    async def process_request(
        self,
        request: ChatCompletionRequest,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> PipelineResult:
        """Process a request through all pipeline steps"""
        self.context.metadata["extra_headers"] = extra_headers
        current_request = request

        self.context.sensitive.api_key = api_key
        self.context.sensitive.model = model
        self.context.sensitive.provider = provider
        self.context.sensitive.api_base = api_base

        # For Copilot provider=openai. Use a flag to not clash with other places that may use that.
        provider_db = provider
        if self.context.client == ClientType.COPILOT:
            provider_db = "copilot"

        for step in self.pipeline_steps:
            result = await step.process(current_request, self.context)
            if result is None:
                continue

            if result.shortcuts_processing():
                # Also record the input when shortchutting
                self.context.add_input_request(
                    current_request, is_fim_request=self.is_fim, provider=provider_db
                )
                return result

            if result.request is not None:
                current_request = result.request

            if result.context is not None:
                self.context = result.context

        # Create the input request at the end so we make sure the secrets are obfuscated
        self.context.add_input_request(
            current_request, is_fim_request=self.is_fim, provider=provider_db
        )
        return PipelineResult(request=current_request, context=self.context)


class SequentialPipelineProcessor:
    def __init__(
        self,
        pipeline_steps: List[PipelineStep],
        secret_manager: SecretsManager,
        client_type: ClientType,
        is_fim: bool,
    ):
        self.pipeline_steps = pipeline_steps
        self.secret_manager = secret_manager
        self.is_fim = is_fim
        self.instance = self._create_instance(client_type)

    def _create_instance(self, client_type: ClientType) -> InputPipelineInstance:
        """Create a new pipeline instance for processing a request"""
        return InputPipelineInstance(
            self.pipeline_steps,
            self.secret_manager,
            self.is_fim,
            client_type,
        )

    async def process_request(
        self,
        request: ChatCompletionRequest,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> PipelineResult:
        """Create a new pipeline instance and process the request"""
        return await self.instance.process_request(
            request,
            provider,
            model,
            api_key,
            api_base,
            extra_headers,
        )
