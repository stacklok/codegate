import asyncio
import datetime
import os
import re
import ssl
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import unquote, urljoin, urlparse

import structlog
from litellm.types.utils import Delta, ModelResponse, StreamingChoices

from codegate.ca.codegate_ca import CertificateAuthority, TLSCertDomainManager
from codegate.codegate_logging import setup_logging
from codegate.config import Config
from codegate.pipeline.base import PipelineContext
from codegate.pipeline.factory import PipelineFactory
from codegate.pipeline.output import OutputPipelineInstance
from codegate.pipeline.secrets.manager import SecretsManager
from codegate.providers.copilot.mapping import PIPELINE_ROUTES, VALIDATED_ROUTES, PipelineType
from codegate.providers.copilot.pipeline import (
    CopilotChatPipeline,
    CopilotFimPipeline,
    CopilotPipeline,
)
from codegate.providers.copilot.streaming import SSEProcessor

setup_logging()
logger = structlog.get_logger("codegate").bind(origin="copilot_proxy")


TEMPDIR = None
if os.getenv("CODEGATE_DUMP_DIR"):
    basedir = os.getenv("CODEGATE_DUMP_DIR")
    TEMPDIR = tempfile.TemporaryDirectory(prefix="codegate-", dir=basedir, delete=False)


def _dump_data(suffix, func):
    if os.getenv("CODEGATE_DUMP_DIR"):
        buf = bytearray(b"")

        def inner(self, data: bytes):
            nonlocal buf
            func(self, data)
            buf.extend(data)

            if data == b"0\r\n\r\n":
                ts = datetime.datetime.now()
                fname = os.path.join(TEMPDIR.name, ts.strftime(f"{suffix}-%Y%m%dT%H%M%S%f.txt"))
                with open(fname, mode="wb") as fd:
                    fd.write(buf)
                buf = bytearray()

        return inner
    return func


def _dump_request(func):
    return _dump_data("request", func)


def _dump_response(func):
    return _dump_data("response", func)


# Constants
MAX_BUFFER_SIZE = 10 * 1024 * 1024  # 10MB
CHUNK_SIZE = 64 * 1024  # 64KB
HTTP_STATUS_MESSAGES = {
    400: "Bad Request",
    404: "Not Found",
    413: "Request Entity Too Large",
    502: "Bad Gateway",
}


@dataclass
class HttpRequest:
    """Data class to store HTTP request details"""

    method: str
    path: str
    version: str
    headers: List[str]
    original_path: str
    target: Optional[str] = None
    body: Optional[bytes] = None

    def reconstruct(self) -> bytes:
        """Reconstruct HTTP request from stored details"""
        headers = "\r\n".join(self.headers)
        request_line = f"{self.method} /{self.path} {self.version}\r\n"
        header_block = f"{request_line}{headers}\r\n\r\n"

        # Convert header block to bytes and combine with body
        result = header_block.encode("utf-8")
        if self.body:
            result += self.body

        return result


@dataclass
class HttpResponse:
    """Data class to store HTTP response details"""

    version: str
    status_code: int
    reason: str
    headers: List[str]
    body: Optional[bytes] = None

    def reconstruct_headers(self) -> bytes:
        """
        Reconstruct the HTTP header of the response.
        """
        headers = "\r\n".join(self.headers)
        status_line = f"{self.version} {self.status_code} {self.reason}\r\n"
        header_block = f"{status_line}{headers}\r\n\r\n"
        return header_block.encode("utf-8")

    def reconstruct_body(self) -> bytes:
        """
        Reconstruct the body of the response.
        """
        result = b""
        if self.body:
            result += self.body
            result += b"\n\n"
        return result

    def reconstruct(self) -> bytes:
        """Reconstruct HTTP response from stored details"""
        return self.reconstruct_headers() + self.reconstruct_body()


def extract_path(full_path: str) -> str:
    """Extract clean path from full URL or path string"""
    logger.debug(f"Extracting path from {full_path}")
    if full_path.startswith(("http://", "https://")):
        parsed = urlparse(full_path)
        path = parsed.path
        if parsed.query:
            path = f"{path}?{parsed.query}"
        return path.lstrip("/")
    return full_path.lstrip("/")


def http_request_from_bytes(data: bytes) -> Optional[HttpRequest]:
    """
    Parse HTTP request details from raw bytes data.
    TODO: Make safer by checking for valid HTTP request format, check
    if there is a method if there are headers, etc.
    """
    if b"\r\n\r\n" not in data:
        return None

    headers_end = data.index(b"\r\n\r\n")
    headers = data[:headers_end].split(b"\r\n")

    request = headers[0].decode("utf-8")
    method, full_path, version = request.split(" ")

    body_start = data.index(b"\r\n\r\n") + 4
    body = data[body_start:]

    return HttpRequest(
        method=method,
        path=extract_path(full_path),
        version=version,
        headers=[header.decode("utf-8") for header in headers[1:]],
        original_path=full_path,
        target=full_path if method == "CONNECT" else None,
        body=body,
    )


class CopilotProvider(asyncio.Protocol):
    """Protocol implementation for the Copilot proxy server"""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        logger.debug("Initializing CopilotProvider")
        self.loop = loop
        self.transport: Optional[asyncio.Transport] = None
        self.target_transport: Optional[asyncio.Transport] = None
        self.peername: Optional[Tuple[str, int]] = None
        self.buffer = bytearray()
        self.target_host: Optional[str] = None
        self.target_port: Optional[int] = None
        self.handshake_done = False
        self.is_connect = False
        self.headers_parsed = False
        self.request: Optional[HttpRequest] = None
        self.ssl_context: Optional[ssl.SSLContext] = None
        self.proxy_ep: Optional[str] = None
        self.ca = CertificateAuthority.get_instance()
        self.cert_manager = TLSCertDomainManager(self.ca)
        self._closing = False
        self.pipeline_factory = PipelineFactory(SecretsManager())
        self.input_pipeline: Optional[CopilotPipeline] = None
        self.fim_pipeline: Optional[CopilotPipeline] = None
        # the context as provided by the pipeline
        self.context_tracking: Optional[PipelineContext] = None

    def _ensure_pipelines(self):
        if not self.input_pipeline or not self.fim_pipeline:
            self.input_pipeline = CopilotChatPipeline(self.pipeline_factory)
            self.fim_pipeline = CopilotFimPipeline(self.pipeline_factory)

    def _select_pipeline(self, method: str, path: str) -> Optional[CopilotPipeline]:
        if method != "POST":
            logger.debug("Not a POST request, no pipeline selected")
            return None

        for route in PIPELINE_ROUTES:
            if path == route.path:
                if route.pipeline_type == PipelineType.FIM:
                    logger.debug("Selected FIM pipeline")
                    return self.fim_pipeline
                elif route.pipeline_type == PipelineType.CHAT:
                    logger.debug("Selected CHAT pipeline")
                    return self.input_pipeline

        logger.debug("No pipeline selected")
        return None

    async def _body_through_pipeline(
        self,
        method: str,
        path: str,
        headers: list[str],
        body: bytes,
    ) -> Tuple[bytes, PipelineContext]:
        strategy = self._select_pipeline(method, path)
        if len(body) == 0 or strategy is None:
            # if we didn't select any strategy that would change the request
            # let's just pass through the body as-is
            return body, None
        return await strategy.process_body(headers, body)

    async def _request_to_target(self, headers: list[str], body: bytes):
        request_line = (
            f"{self.request.method} /{self.request.path} {self.request.version}\r\n"
        ).encode()
        logger.debug(f"Request Line: {request_line}")

        body, context = await self._body_through_pipeline(
            self.request.method,
            self.request.path,
            headers,
            body,
        )

        if context:
            self.context_tracking = context

        for header in headers:
            if header.lower().startswith("content-length:"):
                headers.remove(header)
                break
        headers.append(f"Content-Length: {len(body)}")

        header_block = "\r\n".join(headers).encode()
        headers_request_block = request_line + header_block + b"\r\n\r\n"
        logger.debug("=" * 40)
        self.target_transport.write(headers_request_block)
        logger.debug("=" * 40)

        for i in range(0, len(body), CHUNK_SIZE):
            chunk = body[i : i + CHUNK_SIZE]
            self.target_transport.write(chunk)

    def connection_made(self, transport: asyncio.Transport) -> None:
        """Handle new client connection"""
        self.transport = transport
        self.peername = transport.get_extra_info("peername")
        logger.debug(f"Client connected from {self.peername}")

    def get_headers_dict(self, complete_request) -> Dict[str, str]:
        """Convert raw headers to dictionary format"""
        headers_dict = {}
        try:
            if b"\r\n\r\n" not in complete_request:
                return {}

            headers_end = complete_request.index(b"\r\n\r\n")
            headers = complete_request[:headers_end].split(b"\r\n")[1:]

            for header in headers:
                try:
                    name, value = header.decode("utf-8").split(":", 1)
                    headers_dict[name.strip().lower()] = value.strip()
                    if name == "user-agent":
                        logger.debug(f"User-Agent header received: {value} from {self.peername}")
                except ValueError:
                    continue

            return headers_dict
        except Exception as e:
            logger.error(f"Error parsing headers: {e}")
            return {}

    def parse_headers(self) -> bool:
        """Parse HTTP headers from buffer"""
        try:
            if b"\r\n\r\n" not in self.buffer:
                return False

            headers_end = self.buffer.index(b"\r\n\r\n")
            headers = self.buffer[:headers_end].split(b"\r\n")

            request = headers[0].decode("utf-8")
            method, full_path, version = request.split(" ")

            self.request = HttpRequest(
                method=method,
                path=extract_path(full_path),
                version=version,
                headers=[header.decode("utf-8") for header in headers[1:]],
                original_path=full_path,
                target=full_path if method == "CONNECT" else None,
            )

            logger.debug(f"Request: {method} {full_path} {version}")
            return True

        except Exception as e:
            logger.error(f"Error parsing headers: {e}")
            return False

    def _check_buffer_size(self, new_data: bytes) -> bool:
        """Check if adding new data would exceed buffer size limit"""
        return len(self.buffer) + len(new_data) <= MAX_BUFFER_SIZE

    async def _forward_data_through_pipeline(self, data: bytes) -> Union[HttpRequest, HttpResponse]:
        http_request = http_request_from_bytes(data)
        if not http_request:
            # we couldn't parse this into an HTTP request, so we just pass through
            return data

        result = await self._body_through_pipeline(
            http_request.method,
            http_request.path,
            http_request.headers,
            http_request.body,
        )
        if not result:
            return data
        body, context = result
        # TODO: it's weird that we're overwriting the context.
        # Should we set the context once? Maybe when
        # creating the pipeline instance?
        self.context_tracking = context

        if context and context.shortcut_response:
            # Send shortcut response
            data_prefix = b"data:"
            http_response = HttpResponse(
                http_request.version,
                200,
                "OK",
                [
                    "server: uvicorn",
                    "cache-control: no-cache",
                    "connection: keep-alive",
                    "Content-Type: application/json",
                    "Transfer-Encoding: chunked",
                ],
                data_prefix + body,
            )
            return http_response

        else:
            # Forward request to target
            http_request.body = body

            for header in http_request.headers:
                if header.lower().startswith("content-length:"):
                    http_request.headers.remove(header)
                    break
            http_request.headers.append(f"Content-Length: {len(http_request.body)}")

            return http_request

    async def _forward_data_to_target(self, data: bytes) -> None:
        """
        Forward data to target if connection is established. In case of shortcut
        response, send a response to the client
        """
        pipeline_output = await self._forward_data_through_pipeline(data)

        if isinstance(pipeline_output, HttpResponse):
            # We need to send shortcut response
            if self.transport and not self.transport.is_closing():
                # First, close target_transport since we don't need to send any
                # request to the target
                self.target_transport.close()

                # Send the HTTP headers
                # Note that this representation ends with \r\n\r\n
                self.transport.write(pipeline_output.reconstruct_headers())

                # Encode the body, send its length, then the data
                chunk = pipeline_output.reconstruct_body()
                chunk_size = hex(len(chunk))[2:] + "\r\n"
                self.transport.write(chunk_size.encode())
                self.transport.write(chunk)
                self.transport.write(b"\r\n")

                # Send data done chunk
                chunk = b"data: [DONE]\n\n"
                # Add chunk size for DONE message
                chunk_size = hex(len(chunk))[2:] + "\r\n"
                self.transport.write(chunk_size.encode())
                self.transport.write(chunk)
                self.transport.write(b"\r\n")
                # Now send the final chunk with 0
                self.transport.write(b"0\r\n\r\n")
        else:
            if self.target_transport and not self.target_transport.is_closing():
                if isinstance(pipeline_output, HttpRequest):
                    pipeline_output = pipeline_output.reconstruct()
                self.target_transport.write(pipeline_output)

    def _has_complete_body(self) -> bool:
        """
        Check if we have received the complete request body based on Content-Length header.

        We check the headers from the buffer instead of using self.request.headers on purpose
        because with CONNECT requests, the whole request arrives in the data and is stored in
        the buffer.
        """
        try:
            # For the initial CONNECT request
            if not self.headers_parsed and self.request and self.request.method == "CONNECT":
                return True

            # For subsequent requests or non-CONNECT requests, parse the method from the buffer
            try:
                first_line = self.buffer[: self.buffer.index(b"\r\n")].decode("utf-8")
                method = first_line.split()[0]
            except (ValueError, IndexError):
                # Haven't received the complete request line yet
                return False

            if method != "POST":  # do we need to check for other methods? PUT?
                return True

            # Parse headers from the buffer instead of using self.request.headers
            headers_dict = {}
            try:
                headers_end = self.buffer.index(b"\r\n\r\n")
                if headers_end <= 0:  # Ensure we have a valid headers section
                    return False

                headers = self.buffer[:headers_end].split(b"\r\n")
                if len(headers) <= 1:  # Ensure we have headers after the request line
                    return False

                for header in headers[1:]:  # Skip the request line
                    if not header:  # Skip empty lines
                        continue
                    try:
                        name, value = header.decode("utf-8").split(":", 1)
                        headers_dict[name.strip().lower()] = value.strip()
                    except ValueError:
                        # Skip malformed headers
                        continue
            except ValueError:
                # Haven't received the complete headers yet
                return False

            # TODO: Add proper support for chunked transfer encoding
            # For now, just pass through and let the pipeline handle it
            if "transfer-encoding" in headers_dict:
                return True

            try:
                content_length = int(headers_dict.get("content-length"))
            except (ValueError, TypeError):
                # Content-Length header is required for POST requests without chunked encoding
                logger.error("Missing or invalid Content-Length header in POST request")
                return False

            body_start = headers_end + 4  # Add safety check for buffer length
            if body_start >= len(self.buffer):
                return False

            current_body_length = len(self.buffer) - body_start
            return current_body_length >= content_length
        except Exception as e:
            logger.error(f"Error checking body completion: {e}")
            return False

    def data_received(self, data: bytes) -> None:
        """
        Handle received data from client. Since we need to process the complete body
        through our pipeline before forwarding, we accumulate the entire request first.
        """
        # logger.debug(f"Received data from {self.peername}: {data}")
        try:
            if not self._check_buffer_size(data):
                self.send_error_response(413, b"Request body too large")
                return

            self.buffer.extend(data)

            while self.buffer:  # Process as many complete requests as we have
                if not self.headers_parsed:
                    self.headers_parsed = self.parse_headers()
                    if self.headers_parsed:
                        self._ensure_pipelines()
                        if self.request.method == "CONNECT":
                            if self._has_complete_body():
                                self.handle_connect()
                                self.buffer.clear()  # CONNECT requests are handled differently
                            break  # CONNECT handling complete
                        elif self._has_complete_body():
                            # Find where this request ends
                            headers_end = self.buffer.index(b"\r\n\r\n")
                            headers = self.buffer[:headers_end].split(b"\r\n")[1:]
                            content_length = 0
                            for header in headers:
                                if header.lower().startswith(b"content-length:"):
                                    content_length = int(header.split(b":", 1)[1])
                                    break

                            request_end = headers_end + 4 + content_length
                            complete_request = self.buffer[:request_end]

                            self.buffer = self.buffer[request_end:]  # Keep remaining data

                            self.headers_parsed = False  # Reset for next request

                            asyncio.create_task(self.handle_http_request(complete_request))
                    break  # Either processing request or need more data
                else:
                    if self._has_complete_body():
                        complete_request = bytes(self.buffer)
                        self.buffer.clear()  # Clear buffer for next request
                        asyncio.create_task(self._forward_data_to_target(complete_request))
                    break  # Either processing request or need more data

        except Exception as e:
            logger.error(f"Error processing received data: {e}")
            self.send_error_response(502, str(e).encode())

    async def handle_http_request(self, complete_request: bytes) -> None:
        """Handle standard HTTP request"""
        try:
            target_url = await self._get_target_url(complete_request)
        except Exception as e:
            logger.error(f"Error getting target URL: {e}")
            self.send_error_response(404, b"Not Found")
            return

        try:
            parsed_url = urlparse(target_url)
            self.target_host = parsed_url.hostname
            self.target_port = parsed_url.port or (443 if parsed_url.scheme == "https" else 80)
        except Exception as e:
            logger.error(f"Error parsing target URL: {e}")
            self.send_error_response(502, b"Bad Gateway")
            return

        try:
            target_protocol = CopilotProxyTargetProtocol(self)
            logger.debug(f"Connecting to {self.target_host}:{self.target_port}")
            await self.loop.create_connection(
                lambda: target_protocol,
                self.target_host,
                self.target_port,
                ssl=parsed_url.scheme == "https",
            )
        except Exception as e:
            logger.error(f"Error connecting to target: {e}")
            self.send_error_response(502, b"Failed to establish target connection")
            return

        try:
            has_host = False
            new_headers = []

            for header in self.request.headers:
                if header.lower().startswith("host:"):
                    has_host = True
                    new_headers.append(f"Host: {self.target_host}")
                else:
                    new_headers.append(header)

            if not has_host:
                new_headers.append(f"Host: {self.target_host}")

            if self.target_transport:
                if complete_request:
                    body_start = complete_request.index(b"\r\n\r\n") + 4
                    body = complete_request[body_start:]
                    await self._request_to_target(new_headers, body)
                else:
                    # just skip it
                    logger.info("No buffer content arrived, skipping")
            else:
                logger.error("Target transport not available")
                self.send_error_response(502, b"Failed to establish target connection")
        except Exception as e:
            logger.error(f"Error preparing or sending request to target: {e}")
            self.send_error_response(502, b"Bad Gateway")

    async def _get_target_url(self, complete_request) -> Optional[str]:
        """Determine target URL based on request path and headers"""
        headers_dict = self.get_headers_dict(complete_request)
        auth_header = headers_dict.get("authorization", "")

        if auth_header:
            match = re.search(r"proxy-ep=([^;]+)", auth_header)
            if match:
                self.proxy_ep = match.group(1)
                if not urlparse(self.proxy_ep).scheme:
                    self.proxy_ep = f"https://{self.proxy_ep}"
                return f"{self.proxy_ep}/{self.request.path}"

        return await self.get_target_url(self.request.path)

    async def _establish_target_connection(self, use_ssl: bool) -> None:
        """Establish connection to target server"""
        target_protocol = CopilotProxyTargetProtocol(self)
        await self.loop.create_connection(
            lambda: target_protocol, self.target_host, self.target_port, ssl=use_ssl
        )

    def _send_request_to_target(self) -> None:
        """Send modified request to target server"""
        if not self.target_transport:
            logger.error("Target transport not available")
            self.send_error_response(502, b"Failed to establish target connection")
            return

        headers = self._prepare_request_headers()
        self.target_transport.write(headers)

    def _prepare_request_headers(self) -> bytes:
        """Prepare modified request headers"""
        new_headers = []
        has_host = False

        for header in self.request.headers:
            if header.lower().startswith("host:"):
                has_host = True
                new_headers.append(f"Host: {self.target_host}")
            else:
                new_headers.append(header)

        if not has_host:
            new_headers.append(f"Host: {self.target_host}")

        request_line = f"{self.request.method} /{self.request.path} {self.request.version}\r\n"
        header_block = "\r\n".join(new_headers)
        return f"{request_line}{header_block}\r\n\r\n".encode()

    def handle_connect(self) -> None:
        """Handle CONNECT request for SSL/TLS tunneling"""
        try:
            path = unquote(self.request.target)
            if ":" not in path:
                raise ValueError(f"Invalid CONNECT path: {path}")

            self.target_host, port = path.split(":")
            self.target_port = int(port)

            # Get SSL context through the TLS handler
            self.ssl_context = self.cert_manager.get_domain_context(self.target_host)

            self.is_connect = True
            asyncio.create_task(self.connect_to_target())
            self.handshake_done = True

        except Exception as e:
            logger.error(f"Error handling CONNECT: {e}")
            self.send_error_response(502, str(e).encode())

    async def connect_to_target(self) -> None:
        """Establish connection to target for CONNECT requests"""
        try:
            if not self.target_host or not self.target_port:
                raise ValueError("Target host and port not set")
        except ValueError as e:
            logger.error(f"Error with target host/port: {e}")
            self.send_error_response(502, str(e).encode())
            return

        try:
            target_ssl_context = ssl.create_default_context()
            # Ensure that the target SSL certificate is verified
            target_ssl_context.check_hostname = True
            target_ssl_context.verify_mode = ssl.CERT_REQUIRED
        except Exception as e:
            logger.error(f"Error creating SSL context: {e}")
            self.send_error_response(502, b"SSL context creation failed")
            return

        try:
            # Connect to target
            logger.debug(f"Connecting to {self.target_host}:{self.target_port}")
            target_protocol = CopilotProxyTargetProtocol(self)
            transport, _ = await self.loop.create_connection(
                lambda: target_protocol,
                self.target_host,
                self.target_port,
                ssl=target_ssl_context,
                server_hostname=self.target_host,
            )
        except Exception as e:
            logger.error(f"Failed to connect to target {self.target_host}:{self.target_port}: {e}")
            self.send_error_response(502, str(e).encode())
            return

        try:
            if self.transport and not self.transport.is_closing():
                self.transport.write(
                    b"HTTP/1.1 200 Connection Established\r\n"
                    b"Proxy-Agent: Proxy\r\n"
                    b"Connection: keep-alive\r\n\r\n"
                )
                self.transport = await self.loop.start_tls(
                    self.transport, self, self.ssl_context, server_side=True
                )
        except Exception as e:
            logger.error(f"Error during TLS handshake: {e}")
            self.send_error_response(502, b"TLS handshake failed")

    def send_error_response(self, status: int, message: bytes) -> None:
        """Send error response to client"""
        if self._closing:
            return

        response = (
            f"HTTP/1.1 {status} {HTTP_STATUS_MESSAGES.get(status, 'Error')}\r\n"
            f"Content-Length: {len(message)}\r\n"
            f"Content-Type: text/plain\r\n"
            f"\r\n"
        ).encode() + message

        if self.transport and not self.transport.is_closing():
            self.transport.write(response)
            self.transport.close()

    def connection_lost(self, exc: Optional[Exception]) -> None:
        """Handle connection loss"""
        if self._closing:
            return

        self._closing = True
        logger.debug(f"Connection lost from {self.peername}")

        # Close target transport if it exists and isn't already closing
        if self.target_transport and not self.target_transport.is_closing():
            try:
                self.target_transport.close()
            except Exception as e:
                logger.error(f"Error closing target transport: {e}")

        # Clear references to help with cleanup
        self.transport = None
        self.target_transport = None
        self.buffer.clear()
        self.ssl_context = None

    @classmethod
    async def create_proxy_server(
        cls, host: str, port: int, ssl_context: Optional[ssl.SSLContext] = None
    ) -> asyncio.AbstractServer:
        """Create and start proxy server"""
        loop = asyncio.get_event_loop()
        server = await loop.create_server(
            lambda: cls(loop), host, port, ssl=ssl_context, reuse_port=True, start_serving=False
        )
        logger.debug(f"Proxy server running on https://{host}:{port}")
        return server

    @classmethod
    async def run_proxy_server(cls) -> None:
        """Run the proxy server"""
        try:
            ca = CertificateAuthority.get_instance()
            ssl_context = ca.create_server_ssl_context()
            config = Config.get_config()
            server = await cls.create_proxy_server(config.host, config.proxy_port, ssl_context)

            async with server:
                await server.serve_forever()
        except Exception as e:
            logger.error(f"Proxy server error: {e}")
            raise

    @staticmethod
    async def get_target_url(path: str) -> Optional[str]:
        """Get target URL for the given path"""
        # Check for exact path match
        for route in VALIDATED_ROUTES:
            if path == route.path:
                return str(route.target)

        # Check for prefix match
        for route in VALIDATED_ROUTES:
            # For prefix matches, keep the rest of the path
            remaining_path = path[len(route.path) :]
            logger.debug(f"Remaining path: {remaining_path}")
            # Make sure we don't end up with double slashes
            if remaining_path and remaining_path.startswith("/"):
                remaining_path = remaining_path[1:]
            target = urljoin(str(route.target), remaining_path)
            return target

        logger.warning(f"No route found for path: {path}")
        return None


class CopilotProxyTargetProtocol(asyncio.Protocol):
    """Protocol implementation for proxy target connections"""

    def __init__(self, proxy: CopilotProvider):
        self.proxy = proxy
        self.transport: Optional[asyncio.Transport] = None

        self.headers_sent = False
        self.sse_processor: Optional[SSEProcessor] = None
        self.output_pipeline_instance: Optional[OutputPipelineInstance] = None
        self.stream_queue: Optional[asyncio.Queue] = None
        self.processing_task: Optional[asyncio.Task] = None

        self.finish_stream = False

        # For debugging only
        # self.data_sent = []

    def connection_made(self, transport: asyncio.Transport) -> None:
        """Handle successful connection to target"""
        self.transport = transport
        logger.debug(f"Connection established to target: {transport.get_extra_info('peername')}")
        self.proxy.target_transport = transport

    def _ensure_output_processor(self) -> None:
        if self.proxy.context_tracking is None:
            logger.debug("No context tracking, no need to process pipeline")
            # No context tracking, no need to process pipeline
            return

        if self.sse_processor is not None:
            logger.debug("Already initialized, no need to reinitialize")
            # Already initialized, no need to reinitialize
            return

        logger.debug("Tracking context for pipeline processing")
        self.sse_processor = SSEProcessor()
        is_fim = self.proxy.context_tracking.metadata.get("is_fim", False)
        if is_fim:
            out_pipeline_processor = self.proxy.pipeline_factory.create_fim_output_pipeline()
        else:
            out_pipeline_processor = self.proxy.pipeline_factory.create_output_pipeline()

        self.output_pipeline_instance = OutputPipelineInstance(
            pipeline_steps=out_pipeline_processor.pipeline_steps,
            input_context=self.proxy.context_tracking,
        )

    async def _process_stream(self):  # noqa: C901
        try:

            async def stream_iterator():
                while not self.stream_queue.empty():
                    incoming_record = await self.stream_queue.get()

                    record_content = incoming_record.get("content", {})

                    streaming_choices = []
                    for choice in record_content.get("choices", []):
                        is_fim = self.proxy.context_tracking.metadata.get("is_fim", False)
                        if is_fim:
                            content = choice.get("text", "")
                        else:
                            content = choice.get("delta", {}).get("content")

                        if choice.get("finish_reason", None) == "stop":
                            self.finish_stream = True

                        streaming_choices.append(
                            StreamingChoices(
                                finish_reason=choice.get("finish_reason", None),
                                index=choice.get("index", 0),
                                delta=Delta(content=content, role="assistant"),
                                logprobs=choice.get("logprobs", None),
                                p=choice.get("p", None),
                            )
                        )

                    # Convert record to ModelResponse
                    mr = ModelResponse(
                        id=record_content.get("id", ""),
                        choices=streaming_choices,
                        created=record_content.get("created", 0),
                        model=record_content.get("model", ""),
                        object="chat.completion.chunk",
                        stream=True,
                    )
                    yield mr

            # needs to be set as the flag gets reset on finish_data
            finish_stream_flag = any(
                choice.get("finish_reason") == "stop"
                for record in list(self.stream_queue._queue)
                for choice in record.get("content", {}).get("choices", [])
            )
            async for record in self.output_pipeline_instance.process_stream(
                stream_iterator(), cleanup_sensitive=False, finish_stream=finish_stream_flag,
            ):
                chunk = record.model_dump_json(exclude_none=True, exclude_unset=True)
                sse_data = f"data: {chunk}\n\n".encode("utf-8")
                chunk_size = hex(len(sse_data))[2:] + "\r\n"
                self._proxy_transport_write(chunk_size.encode())
                self._proxy_transport_write(sse_data)
                self._proxy_transport_write(b"\r\n")

            if self.finish_stream:
                self.finish_data()

        except asyncio.CancelledError:
            logger.debug("Stream processing cancelled")
            raise
        except Exception as e:
            logger.error(f"Error processing stream: {e}")
        finally:
            # Clean up
            self.stream_queue = None
            if self.processing_task and not self.processing_task.done():
                self.processing_task.cancel()

    def finish_data(self):
        logger.debug("Finishing data stream")
        sse_data = b"data: [DONE]\n\n"
        # Add chunk size for DONE message too
        chunk_size = hex(len(sse_data))[2:] + "\r\n"
        self._proxy_transport_write(chunk_size.encode())
        self._proxy_transport_write(sse_data)
        self._proxy_transport_write(b"\r\n")
        # Now send the final zero chunk
        self._proxy_transport_write(b"0\r\n\r\n")

        # For debugging only
        # print("===========START DATA SENT====================")
        # for data in self.data_sent:
        #     print(data)
        # self.data_sent = []
        # print("===========START DATA SENT====================")

        self.finish_stream = False
        self.headers_sent = False

    def _process_chunk(self, chunk: bytes):
        # For debugging only
        # print("===========START DATA RECVD====================")
        # print(chunk)
        # print("===========END DATA RECVD======================")

        records = self.sse_processor.process_chunk(chunk)

        for record in records:
            if self.stream_queue is None:
                # Initialize queue and start processing task on first record
                self.stream_queue = asyncio.Queue()
                self.processing_task = asyncio.create_task(self._process_stream())

            self.stream_queue.put_nowait(record)

    @_dump_response
    def _proxy_transport_write(self, data: bytes):
        # For debugging only
        # self.data_sent.append(data)
        if not self.proxy.transport or self.proxy.transport.is_closing():
            logger.error("Proxy transport not available")
            return
        self.proxy.transport.write(data)

    def data_received(self, data: bytes) -> None:
        """Handle data received from target"""
        self._ensure_output_processor()

        if self.proxy.transport and not self.proxy.transport.is_closing():
            if not self.sse_processor:
                # Pass through non-SSE data unchanged
                self.proxy.transport.write(data)
                return

            # Check if this is the first chunk with headers
            if not self.headers_sent:
                header_end = data.find(b"\r\n\r\n")
                if header_end != -1:
                    self.headers_sent = True
                    # Send headers first
                    headers = data[:header_end]

                    # If Transfer-Encoding is not present, add it
                    if b"Transfer-Encoding:" not in headers:
                        headers = headers + b"\r\nTransfer-Encoding: chunked"

                    headers = headers + b"\r\n\r\n"

                    self._proxy_transport_write(headers)
                    logger.debug(f"Headers sent: {headers}")

                    data = data[header_end + 4 :]

            self._process_chunk(data)

    def connection_lost(self, exc: Optional[Exception]) -> None:
        """Handle connection loss to target"""

        logger.debug("Lost connection to target")
        if (
            not self.proxy._closing
            and self.proxy.transport
            and not self.proxy.transport.is_closing()
        ):
            try:
                self.proxy.transport.close()
            except Exception as e:
                logger.error(f"Error closing proxy transport: {e}")

        # Clean up resources
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
        if self.proxy.context_tracking and self.proxy.context_tracking.sensitive:
            self.proxy.context_tracking.sensitive.secure_cleanup()
