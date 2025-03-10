import copy
import json
import uuid
from abc import ABC, abstractmethod
from typing import Callable, Dict, Union
from urllib.parse import urljoin

import structlog
from fastapi.responses import JSONResponse, StreamingResponse
from litellm import ModelResponse
from litellm.types.utils import Delta, StreamingChoices
from ollama import ChatResponse, GenerateResponse

from codegate.db import models as db_models
from codegate.muxing import rulematcher
from codegate.providers.ollama.adapter import OLlamaToModel
from codegate.types.ollama import StreamingChatCompletion as OllamaStreamingChatCompletion
from codegate.types.ollama import StreamingGenerateCompletion as OllamaStreamingGenerateCompletion
from codegate.muxing.ollama_mappers import (
    openai_chunk_from_ollama_chat,
    openai_chunk_from_ollama_generate,
)
from codegate.types.openai import StreamingChatCompletion as OpenAIStreamingChatCompletion

logger = structlog.get_logger("codegate")


class MuxingAdapterError(Exception):
    pass


class BodyAdapter:
    """
    Format the body to the destination provider format.

    We expect the body to always be in OpenAI format. We need to configure the client
    to send and expect OpenAI format. Here we just need to set the destination provider info.
    """

    def _get_provider_formatted_url(self, model_route: rulematcher.ModelRoute) -> str:
        """Get the provider formatted URL to use in base_url. Note this value comes from DB"""
        if model_route.endpoint.provider_type in [
            db_models.ProviderType.openai,
            db_models.ProviderType.vllm,
        ]:
            return urljoin(model_route.endpoint.endpoint, "/v1")
        if model_route.endpoint.provider_type == db_models.ProviderType.openrouter:
            return urljoin(model_route.endpoint.endpoint, "/api/v1")
        return model_route.endpoint.endpoint

    def get_destination_info(self, model_route: rulematcher.ModelRoute) -> dict:
        """Set the destination provider info."""
        return model_route.model.name, self._get_provider_formatted_url(model_route)


class OutputFormatter(ABC):

    @property
    @abstractmethod
    def provider_format_funcs(self) -> Dict[str, Callable]:
        """
        Return the provider specific format functions. All providers format functions should
        return the chunk in OpenAI format.
        """
        pass

    @abstractmethod
    def format(
        self, response: Union[StreamingResponse, JSONResponse], dest_prov: db_models.ProviderType
    ) -> Union[StreamingResponse, JSONResponse]:
        """Format the response to the client."""
        pass


class StreamChunkFormatter(OutputFormatter):
    """
    Format a single chunk from a stream to OpenAI format.
    We need to configure the client to expect the OpenAI format.
    In Continue this means setting "provider": "openai" in the config json file.
    """

    @property
    @abstractmethod
    def provider_format_funcs(self) -> Dict[str, Callable]:
        """
        Return the provider specific format functions. All providers format functions should
        return the chunk in OpenAI format.
        """
        pass

    def _clean_chunk(self, chunk: str) -> str:
        """Clean the chunk from the "data:" and any extra characters."""
        # Find the first position of 'data:' and add 5 characters to skip 'data:'
        start_pos = chunk.find("data:") + 5
        cleaned_chunk = chunk[start_pos:].strip()
        return cleaned_chunk

    def _format_openai(self, chunk: str) -> str:
        """
        The chunk is already in OpenAI format. To standarize remove the "data:" prefix.

        This function is used by both chat and FIM formatters
        """
        return self._clean_chunk(chunk)

    def _format_antropic(self, chunk: str) -> str:
        """
        Format the Anthropic chunk to OpenAI format.

        This function is used by both chat and FIM formatters
        """
        cleaned_chunk = self._clean_chunk(chunk)
        try:
            # Use `strict=False` to allow the JSON payload to contain
            # newlines, tabs and other valid characters that might
            # come from Anthropic returning code.
            chunk_dict = json.loads(cleaned_chunk, strict=False)
        except Exception as e:
            logger.warning(f"Error parsing Anthropic chunk: {chunk}. Error: {e}")
            return cleaned_chunk.strip()

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

        try:
            return open_ai_chunk.model_dump_json(exclude_none=True, exclude_unset=True)
        except Exception as e:
            logger.warning(f"Error serializing Anthropic chunk: {chunk}. Error: {e}")
            return cleaned_chunk.strip()

    def _format_as_openai_chunk(self, formatted_chunk: str) -> str:
        """Format the chunk as OpenAI chunk. This is the format how the clients expect the data."""
        chunk_to_send = f"data: {formatted_chunk}\n\n"
        return chunk_to_send

    async def _format_streaming_response(
        self, response: StreamingResponse, dest_prov: db_models.ProviderType
    ):
        """Format the streaming response to OpenAI format."""
        format_func = self.provider_format_funcs.get(dest_prov)
        openai_chunk = None
        try:
            async for chunk in response.body_iterator:
                openai_chunk = format_func(chunk)
                # Sometimes for Anthropic we couldn't get content from the chunk. Skip it.
                if not openai_chunk:
                    continue
                yield self._format_as_openai_chunk(openai_chunk)
        except Exception as e:
            logger.error(f"Error sending chunk in muxing: {e}")
            yield self._format_as_openai_chunk(str(e))
        finally:
            # Make sure the last chunk is always [DONE]
            if openai_chunk and "[DONE]" not in openai_chunk:
                yield self._format_as_openai_chunk("[DONE]")

    def format(
        self, response: StreamingResponse, dest_prov: db_models.ProviderType
    ) -> StreamingResponse:
        """Format the response to the client."""
        return StreamingResponse(
            self._format_streaming_response(response, dest_prov),
            status_code=response.status_code,
            headers=response.headers,
            background=response.background,
            media_type=response.media_type,
        )


class ChatStreamChunkFormatter(StreamChunkFormatter):
    """
    Format a single chunk from a stream to OpenAI format given that the request was a chat.
    """

    @property
    def provider_format_funcs(self) -> Dict[str, Callable]:
        """
        Return the provider specific format functions. All providers format functions should
        return the chunk in OpenAI format.
        """
        return {
            db_models.ProviderType.ollama: self._format_ollama,
            db_models.ProviderType.openai: self._format_openai,
            db_models.ProviderType.anthropic: self._format_antropic,
            # Our Lllamacpp provider emits OpenAI chunks
            db_models.ProviderType.llamacpp: self._format_openai,
            # OpenRouter is a dialect of OpenAI
            db_models.ProviderType.openrouter: self._format_openai,
            # VLLM is a dialect of OpenAI
            db_models.ProviderType.vllm: self._format_openai,
        }

    def _format_ollama(self, chunk: str) -> str:
        """Format the Ollama chunk to OpenAI format."""
        try:
            chunk_dict = json.loads(chunk)
            ollama_chunk = OllamaStreamingChatCompletion.model_validate(chunk_dict)
            open_ai_chunk = openai_chunk_from_ollama_chat(ollama_chunk)
            return open_ai_chunk.model_dump_json(exclude_none=True, exclude_unset=True)
        except Exception as e:
            # Sometimes we receive an OpenAI formatted chunk from ollama. Specifically when
            # talking to Cline or Kodu. If that's the case we use the format_openai function.
            if "data:" in chunk:
                return self._format_openai(chunk)
            logger.warning(f"Error formatting Ollama chunk: {chunk}. Error: {e}")
            return chunk


class FimStreamChunkFormatter(StreamChunkFormatter):

    @property
    def provider_format_funcs(self) -> Dict[str, Callable]:
        """
        Return the provider specific format functions. All providers format functions should
        return the chunk in OpenAI format.
        """
        return {
            db_models.ProviderType.ollama: self._format_ollama,
            db_models.ProviderType.openai: self._format_openai,
            # Our Lllamacpp provider emits OpenAI chunks
            db_models.ProviderType.llamacpp: self._format_openai,
            # OpenRouter is a dialect of OpenAI
            db_models.ProviderType.openrouter: self._format_openai,
            # VLLM is a dialect of OpenAI
            db_models.ProviderType.vllm: self._format_openai,
            db_models.ProviderType.anthropic: self._format_antropic,
        }

    def _format_ollama(self, chunk: str) -> str:
        """Format the Ollama chunk to OpenAI format."""
        try:
            chunk_dict = json.loads(chunk)
            ollama_chunk = OllamaStreamingGenerateCompletion.model_validate(chunk_dict)
            open_ai_chunk = openai_chunk_from_ollama_generate(ollama_chunk)
            return open_ai_chunk.model_dump_json(exclude_none=True, exclude_unset=True)
        except Exception as e:
            print("Error formatting Ollama chunk: ", chunk, e)
            return chunk


class ResponseAdapter:

    def _get_formatter(
        self, response: Union[StreamingResponse, JSONResponse], is_fim_request: bool
    ) -> OutputFormatter:
        """Get the formatter based on the request type."""
        if isinstance(response, StreamingResponse):
            if is_fim_request:
                return FimStreamChunkFormatter()
            return ChatStreamChunkFormatter()
        raise MuxingAdapterError("Only streaming responses are supported.")

    def format_response_to_client(
        self,
        response: Union[StreamingResponse, JSONResponse],
        dest_prov: db_models.ProviderType,
        is_fim_request: bool,
    ) -> Union[StreamingResponse, JSONResponse]:
        """Format the response to the client."""
        stream_formatter = self._get_formatter(response, is_fim_request)
        return stream_formatter.format(response, dest_prov)
