from typing import Iterator, List, Union

import structlog
from llama_cpp import (
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
    CreateCompletionResponse,
    CreateCompletionStreamResponse,
    Llama,
)

logger = structlog.get_logger("codegate")


class LlamaCppInferenceEngine:
    """
    A wrapper class for llama.cpp models

    Attributes:
        __inference_engine: Singleton instance of this class
    """

    __inference_engine = None

    def __new__(cls):
        if cls.__inference_engine is None:
            cls.__inference_engine = super().__new__(cls)
        return cls.__inference_engine

    def __init__(self):
        if not hasattr(self, "models"):
            self.__models = {}

    def __del__(self):
        self._close_models()

    def _close_models(self):
        """
        Closes all open models and samplers
        """
        for _, model in self.__models.items():
            if model._sampler:
                model._sampler.close()
            model.close()

    async def __get_model(
        self, model_path: str, embedding: bool = False, n_ctx: int = 512, n_gpu_layers: int = 0
    ) -> Llama:
        """
        Returns Llama model object from __models if present. Otherwise, the model
        is loaded and added to __models and returned.
        """
        if model_path not in self.__models:
            logger.info(
                f"Loading model from {model_path} with parameters "
                f"n_gpu_layers={n_gpu_layers} and n_ctx={n_ctx}"
            )
            self.__models[model_path] = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
                n_ctx=n_ctx,
                embedding=embedding,
            )

        return self.__models[model_path]

    async def complete(
        self, model_path: str, n_ctx: int = 512, n_gpu_layers: int = 0, **completion_request
    ) -> Union[CreateCompletionResponse, Iterator[CreateCompletionStreamResponse]]:
        """
        Generates a chat completion using the specified model and request parameters.
        """
        model = await self.__get_model(
            model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers
        )
        return model.create_completion(**completion_request)

    async def chat(
        self, model_path: str, n_ctx: int = 512, n_gpu_layers: int = 0, **chat_completion_request
    ) -> Union[CreateChatCompletionResponse, Iterator[CreateChatCompletionStreamResponse]]:
        """
        Generates a chat completion using the specified model and request parameters.
        """
        model = await self.__get_model(
            model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers
        )
        return model.create_chat_completion(**chat_completion_request)

    async def embed(self, model_path: str, content: List[str], n_gpu_layers=0) -> List[List[float]]:
        """
        Generates an embedding for the given content using the specified model.
        """
        logger.debug(
            "Generating embedding",
            model=model_path.split("/")[-1],
            content=content[0][0 : min(100, len(content[0]))],
            content_length=len(content[0]) if content else 0,
        )

        model = await self.__get_model(
            model_path=model_path, embedding=True, n_gpu_layers=n_gpu_layers
        )
        embedding = model.embed(content)

        logger.debug(
            "Generated embedding",
            model=model_path.split("/")[-1],
            vector_length=len(embedding[0]) if embedding else 0,
        )

        return embedding
