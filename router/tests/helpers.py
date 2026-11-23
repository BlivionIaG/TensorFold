"""shared test helpers: in-process fake backends + router on one asyncio loop, tiny HTTP client"""
import asyncio
import json
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from tfrouter import config as cfgmod                       # noqa: E402
from tfrouter.httpio import SseScanner, Upstream            # noqa: E402
from tfrouter.server import Router                          # noqa: E402
import fakebackend                                          # noqa: E402

SPEED = 200.0


class Fake:
    def __init__(self, server, srv, port):
        self.server, self.srv, self.port, self.m = server, srv, port, server.m

    @property
    def url(self):
        return "http://127.0.0.1:%d" % self.port

    async def close(self):
        self.m.task.cancel()
        self.srv.close()
        for r in list(self.m.active):
            r.cancelled = True


async def start_fake(profile, speed=SPEED, **kw):
    a = fakebackend.parse_args(["--profile", profile, "--speed", str(speed)])
    for k, v in kw.items():
        setattr(a, k, v)
    s = fakebackend.Server(a)
    s.m.task = asyncio.ensure_future(s.m.run())
    srv = await asyncio.start_server(s.handle, "127.0.0.1", 0, limit=1 << 20)
    return Fake(s, srv, srv.sockets[0].getsockname()[1])


async def start_router(fakes, overrides=None, speed=SPEED):
    ov = {"time_scale": speed, "log": {"access": False},
          "backends": {n: {"url": f.url} for n, f in fakes.items()}}
    if overrides:
        _deep(ov, overrides)
    cfg = cfgmod.load(None, ov)
    r = Router(cfg, log=lambda m: None)
    port = await r.start(port=0)
    r.port = port
    return r


def _deep(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep(dst[k], v)
        else:
            dst[k] = v


async def http(port, method, path, body=None, headers=(), raw=False):
    """-> (status, headers, body bytes)"""
    u = await Upstream.open("http", "127.0.0.1", port, 5)
    try:
        data = b"" if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
        await u.send(method, path, "x", [("Content-Type", "application/json")] + list(headers), data)
        await u.read_head()
        return u.status, u.headers, await u.read_all()
    finally:
        u.close()


async def stream(port, body, headers=(), path="/v1/chat/completions", stop_after=None):
    """-> (status, headers, [(wall_s, payload bytes)])  ; stop_after: close the socket after that many events"""
    u = await Upstream.open("http", "127.0.0.1", port, 5)
    evs = []
    try:
        await u.send("POST", path, "x", [("Content-Type", "application/json")] + list(headers), json.dumps(body).encode())
        await u.read_head()
        sc = SseScanner()
        t0 = time.monotonic()
        async for c in u.body():
            for p in sc.feed(c):
                evs.append((time.monotonic() - t0, p))
            if stop_after is not None and len(evs) >= stop_after:
                u.abort()
                break
        return u.status, u.headers, evs
    finally:
        u.close()


def chat_body(n_tokens, cpt=3.3, uniq="a", stream=False, max_tokens=20, model="DeepSeek-V4.1-Flash", extra=None):
    """a chat request of about n_tokens fake tokens (fake tokenizer: chars/cpt + 4 per message + 2)"""
    chars = max(16, int((n_tokens - 4 - 2) * cpt))
    text = (uniq + "-") * (chars // (len(uniq) + 1) + 1)
    b = {"model": model, "messages": [{"role": "user", "content": text[:chars]}], "stream": stream, "max_tokens": max_tokens}
    if extra:
        b.update(extra)
    return b


def run(coro):
    return asyncio.run(coro)
