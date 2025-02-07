import copy
import json
import uuid
from typing import Union

import structlog
from fastapi.responses import JSONResponse, StreamingResponse
from litellm import ModelResponse
from litellm.types.utils import Delta, StreamingChoices
from ollama import ChatResponse

from codegate.db import models as db_models
from codegate.muxing import rulematcher
from codegate.providers.ollama.adapter import OLlamaToModel

logger = structlog.get_logger("codegate")


class MuxingAdapterError(Exception):
    pass


class BodyAdapter:
    """
    Map the body between OpenAI and Anthropic.

    TODO: This are really knaive implementations. We should replace them with more robust ones.
    """

    def _from_openai_to_antrhopic(self, openai_body: dict) -> dict:
        """Map the OpenAI body to the Anthropic body."""
        new_body = copy.deepcopy(openai_body)
        messages = new_body.get("messages", [])
        system_prompt = None
        system_msg_idx = None
        if messages:
            for i_msg, msg in enumerate(messages):
                if msg.get("role", "") == "system":
                    system_prompt = msg.get("content")
                    system_msg_idx = i_msg
                    break
        if system_prompt:
            new_body["system"] = system_prompt
        if system_msg_idx is not None:
            del messages[system_msg_idx]
        return new_body

    def _from_anthropic_to_openai(self, anthropic_body: dict) -> dict:
        """Map the Anthropic body to the OpenAI body."""
        new_body = copy.deepcopy(anthropic_body)
        system_prompt = anthropic_body.get("system")
        messages = new_body.get("messages", [])
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        if "system" in new_body:
            del new_body["system"]
        return new_body

    def _get_provider_formatted_url(self, model_route: rulematcher.ModelRoute) -> str:
        """Get the provider formatted URL to use in base_url. Note this value comes from DB"""
        if model_route.endpoint.provider_type in [
            db_models.ProviderType.openai,
            db_models.ProviderType.openrouter,
        ]:
            return f"{model_route.endpoint.endpoint}/v1"
        return model_route.endpoint.endpoint

    def _set_destination_info(self, data: dict, model_route: rulematcher.ModelRoute) -> dict:
        """Set the destination provider info."""
        new_data = copy.deepcopy(data)
        new_data["model"] = model_route.model.name
        new_data["base_url"] = self._get_provider_formatted_url(model_route)
        return new_data

    def _identify_provider(self, data: dict) -> db_models.ProviderType:
        """Identify the request provider."""
        if "system" in data:
            return db_models.ProviderType.anthropic
        else:
            return db_models.ProviderType.openai

    def map_body_to_dest(self, model_route: rulematcher.ModelRoute, data: dict) -> dict:
        """
        Map the body to the destination provider.

        We only need to transform the body if the destination or origin provider is Anthropic.
        """
        origin_prov = self._identify_provider(data)
        if model_route.endpoint.provider_type == db_models.ProviderType.anthropic:
            if origin_prov != db_models.ProviderType.anthropic:
                data = self._from_openai_to_antrhopic(data)
        else:
            if origin_prov == db_models.ProviderType.anthropic:
                data = self._from_anthropic_to_openai(data)
        return self._set_destination_info(data, model_route)


class StreamChunkFormatter:
    """
    Format a single chunk from a stream to OpenAI format.
    We need to configure the client to expect the OpenAI format.
    In Continue this means setting "provider": "openai" in the config json file.
    """

    def __init__(self):
        self.provider_to_func = {
            db_models.ProviderType.ollama: self._format_ollama,
            db_models.ProviderType.openai: self._format_openai,
            db_models.ProviderType.anthropic: self._format_antropic,
            # Our Lllamacpp provider emits OpenAI chunks
            db_models.ProviderType.llamacpp: self._format_openai,
            # OpenRouter is a dialect of OpenAI
            db_models.ProviderType.openrouter: self._format_openai,
        }

    def _format_ollama(self, chunk: str) -> str:
        """Format the Ollama chunk to OpenAI format."""
        try:
            chunk_dict = json.loads(chunk)
            ollama_chunk = ChatResponse(**chunk_dict)
            open_ai_chunk = OLlamaToModel.normalize_chunk(ollama_chunk)
            return open_ai_chunk.model_dump_json(exclude_none=True, exclude_unset=True)
        except Exception:
            return chunk

    def _format_openai(self, chunk: str) -> str:
        """The chunk is already in OpenAI format. To standarize remove the "data:" prefix."""
        cleaned_chunk = chunk.split("data:")[1].strip()
        try:
            chunk_dict = json.loads(cleaned_chunk)
            open_ai_chunk = ModelResponse(**chunk_dict)
            return open_ai_chunk.model_dump_json(exclude_none=True, exclude_unset=True)
        except Exception:
            return cleaned_chunk

    def _format_antropic(self, chunk: str) -> str:
        """Format the Anthropic chunk to OpenAI format."""
        cleaned_chunk = chunk.split("data:")[1].strip()
        try:
            chunk_dict = json.loads(cleaned_chunk)
            msg_type = chunk_dict.get("type", "")

            finish_reason = None
            if msg_type == "message_stop":
                finish_reason = "stop"

            # In type == "content_block_start" the content comes in "content_block"
            # In type == "content_block_delta" the content comes in "delta"
            msg_content_dict = chunk_dict.get("delta", {}) or chunk_dict.get("content_block", {})
            # We couldn't obtain the content from the chunk. Skip it.
            if not msg_content_dict:
                return ""

            msg_content = msg_content_dict.get("text", "")
            open_ai_chunk = ModelResponse(
                id=f"anthropic-chat-{str(uuid.uuid4())}",
                model="anthropic-muxed-model",
                object="chat.completion.chunk",
                choices=[
                    StreamingChoices(
                        finish_reason=finish_reason,
                        index=0,
                        delta=Delta(content=msg_content, role="assistant"),
                        logprobs=None,
                    )
                ],
            )
            return open_ai_chunk.model_dump_json(exclude_none=True, exclude_unset=True)
        except Exception:
            return cleaned_chunk.strip()

    def format(self, chunk: str, dest_prov: db_models.ProviderType) -> ModelResponse:
        """Format the chunk to OpenAI format."""
        # Get the format function
        format_func = self.provider_to_func.get(dest_prov)
        if format_func is None:
            raise MuxingAdapterError(f"Provider {dest_prov} not supported.")
        return format_func(chunk)


class ResponseAdapter:

    def __init__(self):
        self.stream_formatter = StreamChunkFormatter()

    def _format_as_openai_chunk(self, formatted_chunk: str) -> str:
        """Format the chunk as OpenAI chunk. This is the format how the clients expect the data."""
        return f"data:{formatted_chunk}\n\n"

    async def _format_streaming_response(
        self, response: StreamingResponse, dest_prov: db_models.ProviderType
    ):
        """Format the streaming response to OpenAI format."""
        async for chunk in response.body_iterator:
            openai_chunk = self.stream_formatter.format(chunk, dest_prov)
            # Sometimes for Anthropic we couldn't get content from the chunk. Skip it.
            if not openai_chunk:
                continue
            yield self._format_as_openai_chunk(openai_chunk)

    def format_response_to_client(
        self, response: Union[StreamingResponse, JSONResponse], dest_prov: db_models.ProviderType
    ) -> Union[StreamingResponse, JSONResponse]:
        """Format the response to the client."""
        if isinstance(response, StreamingResponse):
            return StreamingResponse(
                self._format_streaming_response(response, dest_prov),
                status_code=response.status_code,
                headers=response.headers,
                background=response.background,
                media_type=response.media_type,
            )
        else:
            raise MuxingAdapterError("Only streaming responses are supported.")
