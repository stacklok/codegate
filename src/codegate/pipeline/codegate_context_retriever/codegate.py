import itertools
import json

import regex as re
import structlog

from codegate.clients.clients import ClientType
from codegate.db.models import AlertSeverity
from codegate.extract_snippets.factory import MessageCodeExtractorFactory
from codegate.pipeline.base import (
    PipelineContext,
    PipelineResult,
    PipelineStep,
)
from codegate.storage.storage_engine import StorageEngine
from codegate.types.common import ChatCompletionRequest
from codegate.utils.package_extractor import PackageExtractor
from codegate.utils.utils import generate_vector_string

logger = structlog.get_logger("codegate")


# Pre-compiled regex patterns for performance
markdown_code_block = re.compile(r"```.*?```", flags=re.DOTALL)
markdown_file_listing = re.compile(r"⋮...*?⋮...\n\n", flags=re.DOTALL)
environment_details = re.compile(r"<environment_details>.*?</environment_details>", flags=re.DOTALL)


class CodegateContextRetriever(PipelineStep):
    """
    Pipeline step that adds a context message to the completion request when it detects
    the word "codegate" in the user message.
    """

    def __init__(
            self,
            storage_engine: StorageEngine | None = None,
            package_extractor: PackageExtractor | None = None,
        ):
        """
        Initialize the CodegateContextRetriever with optional dependencies.

        Args:
            storage_engine: Optional StorageEngine instance for package searching
            package_extractor: Optional PackageExtractor class for package extraction
        """
        self.storage_engine = storage_engine or StorageEngine()
        self.package_extractor = package_extractor or PackageExtractor

    @property
    def name(self) -> str:
        """
        Returns the name of this pipeline step.
        """
        return "codegate-context-retriever"

    def generate_context_str(
        self, objects: list[object], context: PipelineContext, snippet_map: dict
    ) -> str:
        context_str = ""
        matched_packages = []
        for obj in objects:
            # The object is already a dictionary with 'properties'
            package_obj = obj["properties"]  # type: ignore
            matched_packages.append(f"{package_obj['name']} ({package_obj['type']})")

            # Retrieve the related snippet if it exists
            code_snippet = snippet_map.get(package_obj["name"])

            # Add one alert for each package found
            context.add_alert(
                self.name,
                trigger_string=json.dumps(package_obj),
                severity_category=AlertSeverity.CRITICAL,
                code_snippet=code_snippet,
            )
            package_str = generate_vector_string(package_obj)
            context_str += package_str + "\n"

        if matched_packages:
            logger.debug(
                "Found matching packages in sqlite-vec database", matched_packages=matched_packages
            )
        return context_str

    async def process(  # noqa: C901
        self, request: ChatCompletionRequest, context: PipelineContext
    ) -> PipelineResult:
        """
        Use RAG DB to add context to the user request
        """
        # Get the latest user message
        last_message = self.get_last_user_message_block(request)
        if not last_message:
            return PipelineResult(request=request)
        user_message, last_user_idx = last_message

        # Extract any code snippets
        extractor = MessageCodeExtractorFactory.create_snippet_extractor(context.client)
        snippets = extractor.extract_snippets(user_message)

        bad_snippet_packages = []
        snippet_map = {}
        if snippets and len(snippets) > 0:
            snippet_language = snippets[0].language
            # Collect all packages referenced in the snippets
            snippet_packages = []
            for snippet in snippets:
                extracted_packages = PackageExtractor.extract_packages(
                    snippet.code, snippet.language
                )
                snippet_packages.extend(extracted_packages)
                for package in extracted_packages:
                    snippet_map[package] = snippet

            logger.info(
                f"Found {len(snippet_packages)} packages "
                f"for language {snippet_language} in code snippets."
            )
            # Find bad packages in the snippets
            bad_snippet_packages = await self.storage_engine.search(
                language=snippet_language, packages=snippet_packages
            )  # type: ignore
            logger.info(f"Found {len(bad_snippet_packages)} bad packages in code snippets.")

        # Remove code snippets and file listing from the user messages and search for bad packages
        # in the rest of the user query/messsages
        user_messages = markdown_code_block.sub("", user_message)
        user_messages = markdown_file_listing.sub("", user_messages)
        user_messages = environment_details.sub("", user_messages)

        # split messages into double newlines, to avoid passing so many content in the search
        split_messages = re.split(r"</?task>|\n|\\n", user_messages)
        collected_bad_packages = []
        for item_message in filter(None, map(str.strip, split_messages)):
            # Vector search to find bad packages
            bad_packages = await self.storage_engine.search(
                query=item_message,
                distance=0.5,
                limit=100,
            )
            if bad_packages and len(bad_packages) > 0:
                collected_bad_packages.extend(bad_packages)

        # All bad packages
        all_bad_packages = bad_snippet_packages + collected_bad_packages

        logger.info(f"Adding {len(all_bad_packages)} bad packages to the context.")

        # Generate context string using the searched objects
        context_str = "CodeGate did not find any malicious or archived packages."

        # Nothing to do if no bad packages are found
        if len(all_bad_packages) == 0:
            return PipelineResult(request=request, context=context)
        else:
            # Add context for bad packages
            context_str = self.generate_context_str(all_bad_packages, context, snippet_map)
            context.bad_packages_found = True

            # perform replacement in all the messages starting from this index
            messages = request.get_messages()
            filtered = itertools.dropwhile(lambda x: x[0] < last_user_idx, enumerate(messages))
            for i, message in filtered:
                message_str = ""
                for content in message.get_content():
                    txt = content.get_text()
                    if not txt:
                        logger.debug(f"content has no text: {content}")
                        continue
                    message_str += txt
                context_msg = message_str
                # Add the context to the last user message
                if context.client in [ClientType.CLINE, ClientType.KODU]:
                    match = re.search(r"<task>\s*(.*?)\s*</task>(.*)", message_str, re.DOTALL)
                    if match:
                        task_content = match.group(1)  # Content within <task>...</task>
                        rest_of_message = match.group(
                            2
                        ).strip()  # Content after </task>, if any

                        # Embed the context into the task block
                        updated_task_content = (
                            f"<task>Context: {context_str}"
                            + f"Query: {task_content.strip()}</task>"
                        )

                        # Combine updated task content with the rest of the message
                        context_msg = updated_task_content + rest_of_message
                else:
                    context_msg = f"Context: {context_str} \n\n Query: {message_str}"
                content = next(message.get_content())
                content.set_text(context_msg)
                logger.debug("Final context message", context_message=context_msg)

            return PipelineResult(request=request, context=context)
