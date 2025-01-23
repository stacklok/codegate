import json
import re

import structlog
from litellm import ChatCompletionRequest

from codegate.pipeline.base import (
    AlertSeverity,
    PipelineContext,
    PipelineResult,
    PipelineStep,
)
from codegate.pipeline.extract_snippets.extract_snippets import extract_snippets
from codegate.storage.storage_engine import StorageEngine
from codegate.utils.package_extractor import PackageExtractor
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
        matched_packages = []
        for obj in objects:
            # The object is already a dictionary with 'properties'
            package_obj = obj["properties"]  # type: ignore
            matched_packages.append(f"{package_obj['name']} ({package_obj['type']})")
            # Add one alert for each package found
            context.add_alert(
                self.name,
                trigger_string=json.dumps(package_obj),
                severity_category=AlertSeverity.CRITICAL,
            )
            package_str = generate_vector_string(package_obj)
            context_str += package_str + "\n"

        if matched_packages:
            logger.debug(
                "Found matching packages in sqlite-vec database", matched_packages=matched_packages
            )
        return context_str

    async def process(
        self, request: ChatCompletionRequest, context: PipelineContext
    ) -> PipelineResult:
        """
        Use RAG DB to add context to the user request
        """
        # Get the latest user message
        last_message = self.get_last_user_message_block(request)
        if not last_message:
            return PipelineResult(request=request)
        user_message, _ = last_message

        # Create storage engine object
        storage_engine = StorageEngine()

        # Extract any code snippets
        snippets = extract_snippets(user_message)

        bad_snippet_packages = []
        if len(snippets) > 0:
            snippet_language = snippets[0].language
            # Collect all packages referenced in the snippets
            snippet_packages = []
            for snippet in snippets:
                snippet_packages.extend(
                    PackageExtractor.extract_packages(snippet.code, snippet.language)  # type: ignore
                )

            logger.info(
                f"Found {len(snippet_packages)} packages "
                f"for language {snippet_language} in code snippets."
            )
            # Find bad packages in the snippets
            bad_snippet_packages = await storage_engine.search(
                language=snippet_language, packages=snippet_packages
            )  # type: ignore
            logger.info(f"Found {len(bad_snippet_packages)} bad packages in code snippets.")

        # Remove code snippets and file listing from the user messages and search for bad packages
        # in the rest of the user query/messsages
        user_messages = re.sub(r"```.*?```", "", user_message, flags=re.DOTALL)
        user_messages = re.sub(r"⋮...*?⋮...\n\n", "", user_messages, flags=re.DOTALL)
        user_messages = re.sub(
            r"<environment_details>.*?</environment_details>", "", user_messages, flags=re.DOTALL
        )

        # split messages into double newlines, to avoid passing so many content in the search
        split_messages = re.split(r"</?task>|(\n\n)", user_messages)
        collected_bad_packages = []
        for item_message in split_messages:
            # Vector search to find bad packages
            bad_packages = await storage_engine.search(query=item_message, distance=0.5, limit=100)
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
            context_str = self.generate_context_str(all_bad_packages, context)
            context.bad_packages_found = True

            last_user_idx = self.get_last_user_message_idx(request)

            # Make a copy of the request
            new_request = request.copy()

            # Format: "Context: {context_str} \n Query: {last user message content}"
            message = new_request["messages"][last_user_idx]
            message_str = str(message["content"])  # type: ignore
            # Add the context to the last user message
            if message_str.strip().startswith("<task>"):
                # formatting of cline
                match = re.match(r"(<task>)(.*?)(</task>)(.*)", message_str, re.DOTALL)
                if match:
                    task_start, task_content, task_end, rest_of_message = match.groups()

                # Embed the context into the task block
                updated_task_content = (
                    f"{task_start}Context: {context_str}\n"
                    + f"Query: {task_content.strip()}</details>{task_end}"
                )

                # Combine the updated task block with the rest of the message
                context_msg = updated_task_content + rest_of_message
            else:
                context_msg = f"Context: {context_str} \n\n Query: {message_str}"  # type: ignore
            message["content"] = context_msg

            logger.debug("Final context message", context_message=context_msg)

            return PipelineResult(request=new_request, context=context)
