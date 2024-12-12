import json

import structlog

logger = structlog.get_logger("codegate")


class SSEProcessor:
    def __init__(self):
        self.buffer = ""
        self.initial_chunk = True
        self.chunk_size = None  # Store the original chunk size
        self.size_written = False

    def process_chunk(self, chunk: bytes) -> list:
        print("BUFFER AT START")
        print(self.buffer)
        print("BUFFER AT START - END")
        # Skip any chunk size lines (hex number followed by \r\n)
        try:
            chunk_str = chunk.decode("utf-8")
            lines = chunk_str.split("\r\n")
            for line in lines:
                if all(c in "0123456789abcdefABCDEF" for c in line.strip()):
                    continue
                self.buffer += line
        except UnicodeDecodeError:
            print("Failed to decode chunk")

        records = []
        while True:
            record_end = self.buffer.find("\n\n")
            if record_end == -1:
                print(f"REMAINING BUFFER {self.buffer}")
                break

            record = self.buffer[:record_end]
            self.buffer = self.buffer[record_end + 2 :]

            if record.startswith("data: "):
                data_content = record[6:]
                if data_content.strip() == "[DONE]":
                    records.append({"type": "done"})
                else:
                    try:
                        data = json.loads(data_content)
                        records.append({"type": "data", "content": data})
                    except json.JSONDecodeError:
                        print(f"Failed to parse JSON: {data_content}")

        return records

    def get_pending(self):
        """Return any pending incomplete data in the buffer"""
        return self.buffer
