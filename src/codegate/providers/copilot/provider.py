import asyncio
import re
import ssl
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote, urljoin, urlparse

import structlog

from codegate.ca.codegate_ca import CertificateAuthority
from codegate.config import Config
from codegate.providers.copilot.mapping import VALIDATED_ROUTES

logger = structlog.get_logger("codegate")

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
        self._closing = False

    def connection_made(self, transport: asyncio.Transport) -> None:
        """Handle new client connection"""
        self.transport = transport
        self.peername = transport.get_extra_info("peername")
        logger.debug(f"Client connected from {self.peername}")

    @staticmethod
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

    def get_headers_dict(self) -> Dict[str, str]:
        """Convert raw headers to dictionary format"""
        headers_dict = {}
        try:
            if b"\r\n\r\n" not in self.buffer:
                return {}

            headers_end = self.buffer.index(b"\r\n\r\n")
            headers = self.buffer[:headers_end].split(b"\r\n")[1:]

            for header in headers:
                try:
                    name, value = header.decode("utf-8").split(":", 1)
                    headers_dict[name.strip().lower()] = value.strip()
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
                path=self.extract_path(full_path),
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

    def _forward_data_to_target(self, data: bytes) -> None:
        """Forward data to target if connection is established"""
        if self.target_transport and not self.target_transport.is_closing():
            self.target_transport.write(data)

    def data_received(self, data: bytes) -> None:
        """Handle received data from client"""
        try:
            if not self._check_buffer_size(data):
                self.send_error_response(413, b"Request body too large")
                return

            self.buffer.extend(data)

            if not self.headers_parsed:
                self.headers_parsed = self.parse_headers()
                if self.headers_parsed:
                    if self.request.method == "CONNECT":
                        self.handle_connect()
                    else:
                        asyncio.create_task(self.handle_http_request())
            else:
                self._forward_data_to_target(data)

        except Exception as e:
            logger.error(f"Error processing received data: {e}")
            self.send_error_response(502, str(e).encode())

    async def handle_http_request(self) -> None:
        """Handle standard HTTP request"""

        try:
            target_url = await self._get_target_url()
            if not target_url:
                self.send_error_response(404, b"Not Found")
                return

            parsed_url = urlparse(target_url)
            self.target_host = parsed_url.hostname
            self.target_port = parsed_url.port or (443 if parsed_url.scheme == "https" else 80)

            target_protocol = CopilotProxyTargetProtocol(self)
            logger.debug(f"Connecting to {self.target_host}:{self.target_port}")
            await self.loop.create_connection(
                lambda: target_protocol,
                self.target_host,
                self.target_port,
                ssl=parsed_url.scheme == "https",
            )

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

            request_line = (
                f"{self.request.method} /{self.request.path} {self.request.version}\r\n"
            ).encode()
            logger.debug(f"Request Line: {request_line}")
            header_block = "\r\n".join(new_headers).encode()
            headers = request_line + header_block + b"\r\n\r\n"

            if self.target_transport:
                logger.debug("=" * 40)
                self.target_transport.write(headers)
                logger.debug("=" * 40)

                body_start = self.buffer.index(b"\r\n\r\n") + 4
                body = self.buffer[body_start:]

                for i in range(0, len(body), CHUNK_SIZE):
                    chunk = body[i : i + CHUNK_SIZE]
                    self.target_transport.write(chunk)
            else:
                logger.debug("=" * 40)
                logger.error("Target transport not available")
                logger.debug("=" * 40)
                self.send_error_response(502, b"Failed to establish target connection")

        except Exception as e:
            logger.error(f"Error handling HTTP request: {e}")
            self.send_error_response(502, str(e).encode())

    async def _get_target_url(self) -> Optional[str]:
        """Determine target URL based on request path and headers"""
        headers_dict = self.get_headers_dict()
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

            cert_path, key_path = self.ca.get_domain_certificate(self.target_host)
            self.ssl_context = self._create_ssl_context(cert_path, key_path)

            self.is_connect = True
            asyncio.create_task(self.connect_to_target())
            self.handshake_done = True

        except Exception as e:
            logger.error(f"Error handling CONNECT: {e}")
            self.send_error_response(502, str(e).encode())

    def _create_ssl_context(self, cert_path: str, key_path: str) -> ssl.SSLContext:
        """Create SSL context for CONNECT tunneling"""
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(cert_path, key_path)
        ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        return ssl_context

    async def connect_to_target(self) -> None:
        """Establish connection to target for CONNECT requests"""
        try:
            if not self.target_host or not self.target_port:
                raise ValueError("Target host and port not set")

            target_ssl_context = ssl.create_default_context()
            target_ssl_context.check_hostname = False
            target_ssl_context.verify_mode = ssl.CERT_NONE

            target_protocol = CopilotProxyTargetProtocol(self)
            transport, _ = await self.loop.create_connection(
                lambda: target_protocol, self.target_host, self.target_port, ssl=target_ssl_context
            )

            if self.transport and not self.transport.is_closing():
                self.transport.write(
                    b"HTTP/1.1 200 Connection Established\r\n"
                    b"Proxy-Agent: ProxyPilot\r\n"
                    b"Connection: keep-alive\r\n\r\n"
                )

                self.transport = await self.loop.start_tls(
                    self.transport, self, self.ssl_context, server_side=True
                )

        except Exception as e:
            logger.error(f"Failed to connect to target {self.target_host}:{self.target_port}: {e}")
            self.send_error_response(502, str(e).encode())

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
            lambda: cls(loop), host, port, ssl=ssl_context, reuse_port=True, start_serving=True
        )
        logger.debug(f"Proxy server running on {host}:{port}")
        return server

    @classmethod
    async def run_proxy_server(cls) -> None:
        """Run the proxy server"""
        try:
            ca = CertificateAuthority.get_instance()
            ssl_context = ca.create_ssl_context()
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

    def connection_made(self, transport: asyncio.Transport) -> None:
        """Handle successful connection to target"""
        self.transport = transport
        self.proxy.target_transport = transport

    def data_received(self, data: bytes) -> None:
        """Handle data received from target"""
        if self.proxy.transport and not self.proxy.transport.is_closing():
            self.proxy.transport.write(data)

    def connection_lost(self, exc: Optional[Exception]) -> None:
        """Handle connection loss to target"""
        if (
            not self.proxy._closing
            and self.proxy.transport
            and not self.proxy.transport.is_closing()
        ):
            try:
                self.proxy.transport.close()
            except Exception as e:
                logger.error(f"Error closing proxy transport: {e}")
