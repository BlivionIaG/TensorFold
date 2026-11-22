"""Minimal HTTP/1.1 on asyncio streams (stdlib only): request parsing for the proxy's listener, an upstream client that
decodes chunked / content-length / close-delimited bodies, and an SSE line scanner.

Every connection carries exactly one exchange (`Connection: close`).  Streams are forwarded close-delimited, which is valid HTTP/1.1 and
needs no re-chunking.  Closing the upstream socket is how a client disconnect is propagated (TensorFold counts it as a client disconnection
and abandons the request).
"""
import asyncio
import json
import ssl

MAX_HEADER = 64 * 1024


class HttpError(Exception):
    def __init__(self, status, message, code="invalid_request_error"):
        Exception.__init__(self, message)
        self.status, self.message, self.code = status, message, code


class Request:
    __slots__ = ("method", "target", "path", "query", "headers", "body", "version")

    def header(self, name, default=None):
        return self.headers.get(name.lower(), default)


async def _read_headers(reader, first_line_required=True):
    try:
        raw = await reader.readuntil(b"\r\n\r\n")
    except asyncio.IncompleteReadError as e:
        if not e.partial:
            return None
        raise HttpError(400, "truncated request head")
    except asyncio.LimitOverrunError:
        raise HttpError(431, "request head too large")
    if len(raw) > MAX_HEADER:
        raise HttpError(431, "request head too large")
    lines = raw[:-4].decode("latin-1").split("\r\n")
    headers = {}
    for ln in lines[1:]:
        if ":" not in ln:
            raise HttpError(400, "bad header line")
        k, v = ln.split(":", 1)
        k = k.strip().lower()
        v = v.strip()
        headers[k] = headers[k] + ", " + v if k in headers else v
    return lines[0], headers


async def _read_body(reader, headers, max_body, writer=None):
    te = headers.get("transfer-encoding", "").lower()
    if "chunked" in te:
        out = bytearray()
        while True:
            line = await reader.readline()
            size = int(line.split(b";")[0].strip() or b"0", 16)
            if size == 0:
                while (await reader.readline()).strip():       # trailers
                    pass
                return bytes(out)
            out += await reader.readexactly(size)
            await reader.readexactly(2)
            if len(out) > max_body:
                raise HttpError(413, "request body too large")
    cl = headers.get("content-length")
    if cl is None:
        return b""
    n = int(cl)
    if n > max_body:
        raise HttpError(413, "request body too large")
    return await reader.readexactly(n)


async def read_request(reader, writer, max_body):
    """-> Request, or None when the peer closed without sending anything"""
    head = await _read_headers(reader)
    if head is None:
        return None
    line, headers = head
    parts = line.split(" ")
    if len(parts) != 3:
        raise HttpError(400, "bad request line")
    r = Request()
    r.method, r.target, r.version = parts[0].upper(), parts[1], parts[2]
    r.path, _, r.query = r.target.partition("?")
    r.headers = headers
    if headers.get("expect", "").lower() == "100-continue":
        writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")
        await writer.drain()
    try:
        r.body = await _read_body(reader, headers, max_body)
    except (ValueError, asyncio.IncompleteReadError):
        raise HttpError(400, "bad request body")
    return r


_REASONS = {200: "OK", 400: "Bad Request", 401: "Unauthorized", 404: "Not Found", 405: "Method Not Allowed", 413: "Payload Too Large",
            429: "Too Many Requests", 431: "Request Header Fields Too Large", 499: "Client Closed Request", 500: "Internal Server Error",
            502: "Bad Gateway", 503: "Service Unavailable", 504: "Gateway Timeout"}


def head_bytes(status, headers):
    lines = ["HTTP/1.1 %d %s" % (status, _REASONS.get(status, "Status"))]
    for k, v in headers:
        lines.append("%s: %s" % (k, v))
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")


async def send_response(writer, status, body=b"", headers=None, content_type="application/json"):
    h = [("Content-Type", content_type), ("Content-Length", str(len(body))), ("Connection", "close")] + list(headers or [])
    writer.write(head_bytes(status, h) + body)
    await writer.drain()


def error_body(status, message, code=None, etype="invalid_request_error"):
    return json.dumps({"error": {"message": message, "type": etype, "code": code or str(status)}}).encode()


async def send_error(writer, status, message, code=None, headers=None):
    etype = "invalid_request_error" if status < 500 else "server_error"
    await send_response(writer, status, error_body(status, message, code, etype), headers)


# ------------------------------------------------------------------------------------------------------------------------------
# upstream client
# ------------------------------------------------------------------------------------------------------------------------------
class Upstream:
    """one HTTP exchange with a backend.  Use: u = await Upstream.open(...); await u.send(...); await u.read_head(); async for chunk in u.body()"""

    def __init__(self, reader, writer):
        self.reader, self.writer = reader, writer
        self.status = None
        self.headers = {}
        self._chunked = False
        self._remaining = None

    @classmethod
    async def open(cls, scheme, host, port, connect_timeout):
        ctx = ssl.create_default_context() if scheme == "https" else None
        r, w = await asyncio.wait_for(asyncio.open_connection(host, port, ssl=ctx, limit=1 << 20), connect_timeout)
        return cls(r, w)

    async def send(self, method, path, host, headers, body=b""):
        h = [("Host", host), ("Connection", "close"), ("Accept", "*/*")]
        if body or method in ("POST", "PUT"):
            h.append(("Content-Length", str(len(body))))
        h += list(headers)
        self.writer.write(("%s %s HTTP/1.1\r\n" % (method, path) + "".join("%s: %s\r\n" % kv for kv in h) + "\r\n").encode("latin-1") + body)
        await self.writer.drain()

    async def read_head(self):
        head = await _read_headers(self.reader)
        if head is None:
            raise ConnectionError("backend closed the connection before the response")
        line, self.headers = head
        parts = line.split(" ", 2)
        self.status = int(parts[1])
        self._chunked = "chunked" in self.headers.get("transfer-encoding", "").lower()
        cl = self.headers.get("content-length")
        self._remaining = int(cl) if cl is not None and not self._chunked else None
        return self.status

    async def body(self):
        """async iterator over decoded body bytes, as they arrive"""
        r = self.reader
        if self._chunked:
            while True:
                line = await r.readline()
                if not line:
                    raise ConnectionError("backend closed mid-stream")
                size = int(line.split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    while (await r.readline()).strip():
                        pass
                    return
                data = await r.readexactly(size)
                await r.readexactly(2)
                yield data
        elif self._remaining is not None:
            left = self._remaining
            while left > 0:
                data = await r.read(min(left, 65536))
                if not data:
                    raise ConnectionError("backend closed mid-body")
                left -= len(data)
                yield data
        else:
            while True:
                data = await r.read(65536)
                if not data:
                    return
                yield data

    async def read_all(self, limit=1 << 28):
        out = bytearray()
        async for c in self.body():
            out += c
            if len(out) > limit:
                raise ConnectionError("response too large")
        return bytes(out)

    def close(self):
        try:
            self.writer.close()
        except Exception:
            pass

    def abort(self):
        """hard close (RST-like): lets the backend see the disconnect at once"""
        try:
            self.writer.transport.abort()
        except Exception:
            self.close()


async def simple_request(scheme, host, port, method, path, headers, body=b"", timeout=3.0):
    """one short exchange -> (status, headers, body bytes); raises on connect/timeout/protocol errors"""
    async def go():
        u = await Upstream.open(scheme, host, port, timeout)
        try:
            await u.send(method, path, "%s:%s" % (host, port), headers, body)
            await u.read_head()
            data = await u.read_all()
            return u.status, u.headers, data
        finally:
            u.close()
    return await asyncio.wait_for(go(), timeout)


# ------------------------------------------------------------------------------------------------------------------------------
# SSE
# ------------------------------------------------------------------------------------------------------------------------------
class SseScanner:
    """feed raw stream bytes, get the `data:` payloads (bytes) of the events completed so far"""

    def __init__(self):
        self.buf = b""

    def feed(self, data):
        self.buf += data
        out = []
        while True:
            i = self.buf.find(b"\n")
            if i < 0:
                break
            line, self.buf = self.buf[:i].rstrip(b"\r"), self.buf[i + 1:]
            if line.startswith(b"data:"):
                out.append(line[5:].strip())
        return out
