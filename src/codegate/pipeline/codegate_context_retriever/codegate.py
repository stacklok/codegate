import json

import structlog
from litellm import ChatCompletionRequest

from codegate.llm_utils.extractor import PackageExtractor
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

    async def get_objects_from_search(
        self, search: str, ecosystem, packages: list[str] = None
    ) -> list[object]:
        storage_engine = StorageEngine()
        objects = await storage_engine.search(
            search, distance=0.8, ecosystem=ecosystem, packages=packages
        )
        return objects

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

    async def __lookup_packages(self, user_query: str, context: PipelineContext):
        # Use PackageExtractor to extract packages from the user query
        packages = await PackageExtractor.extract_packages(
            content=user_query,
            provider=context.sensitive.provider,
            model=context.sensitive.model,
            api_key=context.sensitive.api_key,
            base_url=context.sensitive.api_base,
        )

        logger.info(f"Packages in user query: {packages}")
        return packages

    async def __lookup_ecosystem(self, user_query: str, context: PipelineContext):
        # Use PackageExtractor to extract ecosystem from the user query
        ecosystem = await PackageExtractor.extract_ecosystem(
            content=user_query,
            provider=context.sensitive.provider,
            model=context.sensitive.model,
            api_key=context.sensitive.api_key,
            base_url=context.sensitive.api_base,
        )

        logger.info(f"Ecosystem in user query: {ecosystem}")
        return ecosystem

    async def process(
        self, request: ChatCompletionRequest, context: PipelineContext
    ) -> PipelineResult:
        """
        Use RAG DB to add context to the user request
        """

        # Get the last user message
        last_user_message = self.get_last_user_message(request)

        # Nothing to do if the last user message is none
        if last_user_message is None:
            return PipelineResult(request=request)

        # Extract packages from the user message
        last_user_message_str, last_user_idx = last_user_message
        ecosystem = await self.__lookup_ecosystem(last_user_message_str, context)
        packages = await self.__lookup_packages(last_user_message_str, context)
        packages = [pkg.lower() for pkg in packages]

        # If user message does not reference any packages, then just return
        if len(packages) == 0:
            return PipelineResult(request=request)

        # Look for matches in vector DB using list of packages as filter
        searched_objects = await self.get_objects_from_search(
            last_user_message_str, ecosystem, packages
        )

        logger.info(
            f"Found {len(searched_objects)} matches in the database",
            searched_objects=searched_objects,
        )

        # Remove searched objects that are not in packages. This is needed
        # since Weaviate performs substring match in the filter.
        updated_searched_objects = []
        for searched_object in searched_objects:
            if searched_object.properties["name"].lower() in packages:
                updated_searched_objects.append(searched_object)
        searched_objects = updated_searched_objects

        # Generate context string using the searched objects
        logger.info(f"Adding {len(searched_objects)} packages to the context")

        if len(searched_objects) > 0:
            context_str = self.generate_context_str(searched_objects, context)
        else:
            context_str = "Codegate did not find any malicious or archived packages."

        # Make a copy of the request
        new_request = request.copy()

        # Add the context to the last user message
        # Format: "Context: {context_str} \n Query: {last user message conent}"
        # Handle the two cases: (a) message content is str, (b)message content
        # is list
        message = new_request["messages"][last_user_idx]
        if isinstance(message["content"], str):
            context_msg = f'Context: {context_str} \n\n Query: {message["content"]}'
            message["content"] = context_msg
        elif isinstance(message["content"], (list, tuple)):
            for item in message["content"]:
                if isinstance(item, dict) and item.get("type") == "text":
                    context_msg = f'Context: {context_str} \n\n Query: {item["text"]}'
                    item["text"] = context_msg

                return PipelineResult(request=new_request, context=context)

        # Fall through
        return PipelineResult(request=request, context=context)
