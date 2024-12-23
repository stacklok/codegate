import json

import structlog
from litellm import ChatCompletionRequest

from codegate.pipeline.base import (
    AlertSeverity,
    PipelineContext,
    PipelineResult,
    PipelineStep,
)
from codegate.storage.storage_engine import StorageEngine
from codegate.utils.utils import generate_vector_string

logger = structlog.get_logger("codegate")


class CodegateContextRetriever(PipelineStep):
    """
    Pipeline step that adds a context message to the completion request when it detects
    the word "codegate" in the user message.
    """

    @property
    def name(self) -> str:
        """
        Returns the name of this pipeline step.
        """
        return "codegate-context-retriever"

    def generate_context_str(self, objects: list[object], context: PipelineContext) -> str:
        context_str = ""
        for obj in objects:
            # generate dictionary from object
            package_obj = {
                "name": obj.properties["name"],
                "type": obj.properties["type"],
                "status": obj.properties["status"],
                "description": obj.properties["description"],
            }
            # Add one alert for each package found
            context.add_alert(
                self.name,
                trigger_string=json.dumps(package_obj),
                severity_category=AlertSeverity.CRITICAL,
            )
            package_str = generate_vector_string(package_obj)
            context_str += package_str + "\n"
        return context_str

    async def process(
        self, request: ChatCompletionRequest, context: PipelineContext
    ) -> PipelineResult:
        """
        Use RAG DB to add context to the user request
        """

        # Get the latest user messages
        user_messages = self.get_latest_user_messages(request)

        # Nothing to do if the user_messages string is empty
        if len(user_messages) == 0:
            return PipelineResult(request=request)

        context_str = "CodeGate did not find any malicious or archived packages."

        # Vector search to find bad packages
        storage_engine = StorageEngine()
        searched_objects = await storage_engine.search(
            query=user_messages, distance=0.8, limit=100
        )

        logger.info(
            f"Found {len(searched_objects)} matches in the database",
            searched_objects=searched_objects,
        )

        # Generate context string using the searched objects
        logger.info(f"Adding {len(searched_objects)} packages to the context")

        if len(searched_objects) > 0:
            context_str = self.generate_context_str(searched_objects, context)

        last_user_idx = self.get_last_user_message_idx(request)

        # Make a copy of the request
        new_request = request.copy()

        # Add the context to the last user message
        # Format: "Context: {context_str} \n Query: {last user message content}"
        message = new_request["messages"][last_user_idx]
        context_msg = f'Context: {context_str} \n\n Query: {message["content"]}'
        message["content"] = context_msg

        return PipelineResult(request=new_request, context=context)
