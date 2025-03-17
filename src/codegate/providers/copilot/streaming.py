from typing import List

import structlog
from pydantic import ValidationError

from codegate.types.openai import StreamingChatCompletion

logger = structlog.get_logger("codegate")


class SSEProcessor:
    def __init__(self):
        self.buffer = ""
        self.initial_chunk = True
        self.chunk_size = None  # Store the original chunk size
        self.size_written = False

    def process_chunk(self, chunk: bytes) -> List[StreamingChatCompletion]:
        # Skip any chunk size lines (hex number followed by \r\n)
        try:
            chunk_str = chunk.decode("utf-8")
            lines = chunk_str.split("\r\n")
            for line in lines:
                if all(c in "0123456789abcdefABCDEF" for c in line.strip()):
                    continue
                self.buffer += line
        except UnicodeDecodeError:
            logger.error("Failed to decode chunk")

        records: List[StreamingChatCompletion] = []
        while True:
            record_end = self.buffer.find("\n\n")
            if record_end == -1:
                break

            record = self.buffer[:record_end]
            self.buffer = self.buffer[record_end + 2 :]

            if record.startswith("data: "):
                data_content = record[6:]
                if data_content.strip() == "[DONE]":
                    # We don't actually need to do anything with this message as the caller relies
                    # on the stop_reason
                    logger.debug("Received DONE message")
                else:
                    try:
                        record = StreamingChatCompletion.model_validate_json(data_content)
                        records.append(record)
                    except ValidationError as e:
                        logger.debug(f"Failed to parse JSON: {data_content}: {e}")

        return records

    def get_pending(self):
        """Return any pending incomplete data in the buffer"""
        return self.buffer
