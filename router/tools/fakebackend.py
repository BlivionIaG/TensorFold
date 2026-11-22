#!/usr/bin/env python3
"""fakebackend.py: an OpenAI/TensorFold-shaped SSE server with the simulator's machine model, for offline router tests (stdlib only, asyncio).

It is bench/sim.py's `Machine` run in (scaled) real time behind real HTTP:
  * slots; a request waits for a free slot, then its prompt is prefilled FIFO, one prompt at a time [MEAS baselines.md s4];
  * profile S: 2048-row chunks with 250 ms of decode rounds between two chunks; profile M: a prefill stops every decoding stream [MEAS s5];
  * a decode round advances every stream by tokens_per_round(k) tokens in round_ms(k) ms; one SSE event per stream per round;
  * prefill table, round table, slots, context limit come from tfrouter.config presets (the measured numbers) and can be scaled
    (--prefill-scale 1.3 = a machine 30 % slower than its table) to test the router's calibration;
  * prefix cache: the longest common character prefix with a cached prompt+reply (LRU by tokens) is not prefilled again; usage reports cached_tokens;
  * reply: usage in the final event, a `tensorfold` block (prefill_seconds, seconds, time_to_first_token, tokens_per_second, in virtual seconds);
  * /health, /v1/models, /v1/tokenize, /v1/load, API key check, fault injection (/fake/fault), counters (/fake/stats), cancellation on client disconnect.
--speed N runs the machine N times faster than real time (event times are exact on the machine's own timeline; wall sleeps only decide when
an event becomes visible), and every reported duration is in virtual seconds.
"""
import argparse
import asyncio
import contextlib
import json
import math
import os
import sys
import time
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tfrouter.config import PRESETS                       # noqa: E402  (the measured tables are the single source of truth)
from tfrouter.httpio import HttpError, read_request, send_response   # noqa: E402
from tfrouter.model import interp                          # noqa: E402


def lcp(a, b):
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


class Req:
    def __init__(self, rid, n_in, n_out, text, t_arr):
        self.id, self.n_in, self.n_out, self.text, self.t_arr = rid, n_in, n_out, text, t_arr
        self.q = asyncio.Queue()
        self.cached = 0
        self.rows = n_in
        self.tokens = 0
        self.acc = 0.0
        self.has_slot = False
        self.cancelled = False
        self.t_pf_start = self.t_first = self.t_done = None
        self.job = None


class Job:
    def __init__(self, req, total, chunk):
        self.req, self.rem, self.chunk = req, total, chunk
        self.total = total


class Machine:
    def __init__(self, a):
        p = PRESETS[a.profile]
        self.a, self.name = a, a.name or a.profile
        self.speed = a.speed
        self.slots = a.slots or p["slots"]
        self.max_ctx = a.max_context or p["max_context"]
        self.pf_pts = p["prefill_pts"]
        self.round_pts = p["decode"]["round_ms_pts"]
        self.tpe_pts = p["decode"]["tokens_per_round_pts"]
        self.mode = p["prefill"]["mode"]
        self.chunk_rows = p["prefill"]["chunk_rows"]
        self.window = p["prefill"]["window_s"]
        self.pf_scale, self.dec_scale = a.prefill_scale, a.decode_scale
        self.floor_extra = 0.0
        self.t0 = time.monotonic()
        self.t = 0.0
        self.wake = asyncio.Event()
        self.queue = []                 # prefill jobs FIFO
        self.active = []
        self.admit_q = deque()
        self.slots_used = 0
        self.win_left = 0.0
        self.cache = []                 # [text, tokens, last_use]
        self.cache_tokens = 0
        self.cache_cap = a.cache_tokens or p["prefix_cache_tokens"]
        self.stats = dict(requests=0, completed=0, cancelled=0, rejected=0, tokens_out=0, busy_pf_v=0.0, busy_dec_v=0.0, cached_tokens=0, prompt_tokens=0,
                          max_active=0, max_queue=0)
        self.task = None

    def vnow(self):
        return (time.monotonic() - self.t0) * self.speed

    # ---- tokens -----------------------------------------------------------------------------------------------------------
    def count_tokens(self, chars, nmsgs):
        return int(math.ceil(chars / self.a.cpt)) + nmsgs * self.a.msg_overhead + 2

    def tok_text(self, i0, n):
        cpt = self.a.cpt
        out = []
        for i in range(i0, i0 + n):
            L = int(round((i + 1) * cpt)) - int(round(i * cpt))
            out.append("abcdefgh"[i % 8] * max(1, L))
        return "".join(out)

    # ---- timing model -------------------------------------------------------------------------------------------------------
    def pf_seconds(self, rows):
        tot = interp(self.pf_pts, rows) * self.pf_scale
        if rows <= self.chunk_rows:
            return tot, tot
        return tot, tot * self.chunk_rows / rows

    def step_s(self, k):
        return interp(self.round_pts, max(1, k)) / 1000.0 / self.dec_scale

    def tpe(self, k):
        return max(1.0, interp(self.tpe_pts, max(1, k)))

    # ---- prefix cache ---------------------------------------------------------------------------------------------------------
    def cache_lookup(self, text, n_in):
        best, ent = 0, None
        for e in self.cache:
            l = lcp(text, e[0])
            if l > best:
                best, ent = l, e
        if not ent:
            return 0
        ent[2] = time.monotonic()
        toks = int(best / self.a.cpt) // 64 * 64
        return max(0, min(toks, n_in - 1, ent[1]))

    def cache_put(self, text, tokens):
        self.cache = [e for e in self.cache if not (text.startswith(e[0]) or e[0] == text)]
        self.cache.append([text, tokens, time.monotonic()])
        self.cache_tokens = sum(e[1] for e in self.cache)
        while self.cache_tokens > self.cache_cap and len(self.cache) > 1:
            self.cache.sort(key=lambda e: e[2])
            self.cache_tokens -= self.cache.pop(0)[1]

    # ---- request admission --------------------------------------------------------------------------------------------------------
    def submit(self, r):
        self.stats["requests"] += 1
        if self.slots_used < self.slots:
            self._start(r)
        else:
            self.admit_q.append(r)
            self.stats["max_queue"] = max(self.stats["max_queue"], len(self.admit_q))

    def _start(self, r):
        self.slots_used += 1
        r.has_slot = True
        r.cached = self.cache_lookup(r.text, r.n_in)
        r.rows = max(1, r.n_in - r.cached)
        tot, ch = self.pf_seconds(r.rows)
        r.job = Job(r, tot, ch)
        self.queue.append(r.job)
        self.wake.set()

    def _release(self, r):
        if r.has_slot:
            r.has_slot = False
            self.slots_used -= 1
            while self.admit_q:
                nxt = self.admit_q.popleft()
                if not nxt.cancelled:
                    self._start(nxt)
                    break

    def cancel(self, r):
        if r.cancelled or r.t_done is not None:
            return
        r.cancelled = True
        self.stats["cancelled"] += 1
        if r in self.admit_q:
            self.admit_q.remove(r)
        if r.job in self.queue:
            self.queue.remove(r.job)
        if r in self.active:
            self.active.remove(r)
        self._release(r)

    # ---- the machine ---------------------------------------------------------------------------------------------------------------
    async def advance(self, d):
        self.t += d
        delay = (self.t - self.vnow()) / self.speed
        await asyncio.sleep(delay if delay > 0.0004 else 0)

    async def run(self):
        while True:
            job = self.queue[0] if self.queue else None
            k = len(self.active)
            self.stats["max_active"] = max(self.stats["max_active"], k)
            if job is None and k == 0:
                self.wake.clear()
                await self.wake.wait()
                self.t = max(self.t, self.vnow())
                continue
            if self.mode == "interleave":
                go_pf = job is not None and (k == 0 or self.win_left <= 1e-12)
            else:
                go_pf = job is not None
            if go_pf:
                r = job.req
                if r.t_pf_start is None:
                    r.t_pf_start = self.t
                d = min(job.chunk, job.rem)
                await self.advance(d)
                self.stats["busy_pf_v"] += d
                self.win_left = self.window
                if job not in self.queue:           # cancelled during the chunk
                    continue
                job.rem -= d
                if job.rem <= 1e-9:
                    self.queue.remove(job)
                    self._first_token(r)
            elif k > 0:
                d = self.step_s(k)
                await self.advance(d)
                self.stats["busy_dec_v"] += d
                self.win_left -= d
                tp = self.tpe(len(self.active))
                for r in list(self.active):
                    if r.cancelled:
                        continue
                    r.acc += tp
                    whole = int(r.acc + 1e-9)
                    r.acc -= whole
                    nt = min(whole, r.n_out - r.tokens)
                    if nt > 0:
                        r.q.put_nowait(("tok", r.tokens, nt, self.t))
                        r.tokens += nt
                        self.stats["tokens_out"] += nt
                    if r.tokens >= r.n_out:
                        self._finish(r)

    def _first_token(self, r):
        r.t_first = self.t
        r.tokens = 1
        self.stats["tokens_out"] += 1
        self.cache_put(r.text, r.n_in)
        r.q.put_nowait(("tok", 0, 1, self.t))
        if r.tokens >= r.n_out:
            self._finish(r)
        else:
            self.active.append(r)

    def _finish(self, r):
        if r in self.active:
            self.active.remove(r)
        r.t_done = self.t
        self.stats["completed"] += 1
        self.stats["prompt_tokens"] += r.n_in
        self.stats["cached_tokens"] += r.cached
        self.cache_put(r.text + self.tok_text(0, r.n_out), r.n_in + r.n_out)
        r.q.put_nowait(("done", r.tokens, 0, self.t))
        self._release(r)


class Server:
    def __init__(self, a):
        self.a = a
        self.m = Machine(a)
        self.fault = {"mode": "ok"}
        self.keys = []
        if a.api_key_file:
            with open(a.api_key_file) as f:
                self.keys = [ln.strip().split(": ")[-1] for ln in f if ln.strip() and not ln.startswith("#")]
        self.seq = 0
        self.inflight_http = 0

    def authed(self, req):
        if not self.keys:
            return True
        tok = req.header("authorization", "")
        return tok.startswith("Bearer ") and tok[7:].strip() in self.keys

    async def handle(self, reader, writer):
        try:
            req = await read_request(reader, writer, 1 << 28)
            if req is None:
                return
            if self.fault["mode"] == "down" and not req.path.startswith("/fake/"):
                writer.transport.abort()                 # a dead machine: connections reset (the fake's own control routes still answer)
                return
            await self.route(req, reader, writer)
        except HttpError as e:
            with contextlib.suppress(Exception):
                await send_response(writer, e.status, json.dumps({"error": {"message": e.message}}).encode())
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            with contextlib.suppress(Exception):
                writer.close()

    async def route(self, req, reader, writer):
        p = req.path
        m = self.m
        if p in ("/health", "/"):
            if self.fault["mode"] in ("err503",):
                return await send_response(writer, 503, b'{"status":"unavailable"}')
            if self.keys:
                return await send_response(writer, 200, b'{"status":"ok"}')
            live = {"connections": len(m.active) + len(m.queue) + len(m.admit_q), "waiting": len(m.queue) + len(m.admit_q), "decode_tokens_per_second": 0.0,
                    "prefill_tokens_per_second": 0.0}
            return await send_response(writer, 200, json.dumps({"status": "ok", "model": self.a.model, "model_ids": [self.a.model], "max_batch_size": m.slots, "live": live}).encode())
        if p.startswith("/fake/"):
            if p == "/fake/stats":
                s = dict(m.stats, vnow=m.vnow(), machine_t=m.t, active=len(m.active), queued=len(m.queue) + len(m.admit_q), slots_used=m.slots_used, fault=self.fault,
                         cache_entries=len(m.cache), cache_tokens=m.cache_tokens)
                return await send_response(writer, 200, json.dumps(s).encode())
            if p == "/fake/fault":
                self.fault = json.loads(req.body or b"{}")
                return await send_response(writer, 200, json.dumps(self.fault).encode())
            raise HttpError(404, "no such fake route")
        if not self.authed(req):
            return await send_response(writer, 401, b'{"error":{"message":"invalid API key"}}', [("WWW-Authenticate", "Bearer")])
        if req.method == "GET" and p in ("/v1/models", "/models"):
            return await send_response(writer, 200, json.dumps({"object": "list", "data": [{"id": self.a.model, "object": "model", "context_length": m.max_ctx, "max_model_len": m.max_ctx}]}).encode())
        if req.method == "GET" and p == "/v1/load":
            return await send_response(writer, 200, json.dumps({"instance": "fake", "model": self.a.model, "lanes": m.slots, "http_active": {"chat_completions": self.inflight_http, "completions": 0},
                                                                "requests_running": len(m.active), "requests_waiting": len(m.queue) + len(m.admit_q)}).encode())
        if req.method == "POST" and p in ("/v1/tokenize", "/tokenize"):
            b = json.loads(req.body or b"{}")
            n = self.count_prompt(b)[0]
            return await send_response(writer, 200, json.dumps({"count": n, "max_model_len": m.max_ctx, "tokens": []}).encode())
        if req.method == "POST" and p in ("/v1/chat/completions", "/v1/completions"):
            if self.fault["mode"] == "err503":
                return await send_response(writer, 503, b'{"error":{"message":"engine unavailable"}}')
            if self.fault["mode"] == "err429":
                return await send_response(writer, 429, b'{"error":{"message":"busy"}}', [("Retry-After", "1")])
            self.inflight_http += 1
            try:
                return await self.completion(req, reader, writer, p.endswith("chat/completions"))
            finally:
                self.inflight_http -= 1
        raise HttpError(404, "no such route")

    def count_prompt(self, b):
        """-> (tokens, text)"""
        if "messages" in b and b["messages"] is not None:
            parts, nm = [], 0
            for msg in b["messages"]:
                c = msg.get("content") or ""
                if isinstance(c, list):
                    c = "".join(x.get("text", "") for x in c)
                parts.append(msg.get("role", "") + ":" + c + "\n")
                nm += 1
            text = "".join(parts)
            return self.m.count_tokens(sum(len(msg.get("content") or "") if isinstance(msg.get("content"), str) else 0 for msg in b["messages"]), nm), text
        pr = b.get("prompt") or ""
        if isinstance(pr, list):
            pr = "".join(x for x in pr if isinstance(x, str))
        return self.m.count_tokens(len(pr), 0), pr

    async def completion(self, req, reader, writer, chat):
        m, a = self.m, self.a
        b = json.loads(req.body or b"{}")
        if b.get("model") not in (None, a.model):
            raise HttpError(404, "model %r not found (this server is %s)" % (b.get("model"), a.model))
        n_in, text = self.count_prompt(b)
        n_out = int(req.header("x-fake-output-tokens") or b.get("max_tokens") or b.get("max_completion_tokens") or 128)
        if n_in + n_out > m.max_ctx:
            m.stats["rejected"] += 1
            raise HttpError(400, "context_length_exceeded: %d + %d > %d" % (n_in, n_out, m.max_ctx), "context_length_exceeded")
        if self.fault["mode"] == "hang":
            await asyncio.sleep(3600)
        self.seq += 1
        r = Req(self.seq, n_in, n_out, text, m.vnow())
        stream = bool(b.get("stream"))
        incl = bool((b.get("stream_options") or {}).get("include_usage"))
        gone = asyncio.get_running_loop().create_future()

        async def watch():
            with contextlib.suppress(Exception):
                while await reader.read(4096):
                    pass
            if not gone.done():
                gone.set_result(True)
        wt = asyncio.ensure_future(watch())
        m.submit(r)
        cid = "chatcmpl-fake%d" % r.id
        created = int(time.time())

        def chunk(delta_text, finish=None, role=False, extra=None, empty=False):
            if chat:
                d = {}
                if role:
                    d["role"] = "assistant"
                if delta_text:
                    d["content"] = delta_text
                ch = [{"index": 0, "delta": d, "finish_reason": finish}]
            else:
                ch = [{"index": 0, "text": delta_text or "", "finish_reason": finish}]
            o = {"id": cid, "object": "chat.completion.chunk" if chat else "text_completion", "created": created, "model": a.model, "choices": [] if empty else ch}
            if extra:
                o.update(extra)
            return ("data: " + json.dumps(o, separators=(",", ":")) + "\n\n").encode()

        def final_extra():
            vt_first = (r.t_first - r.t_arr) if r.t_first is not None else 0.0
            dec = (r.t_done - r.t_first) if r.t_first is not None and r.t_done is not None else 0.0
            usage = {"prompt_tokens": r.n_in, "completion_tokens": r.tokens, "total_tokens": r.n_in + r.tokens,
                     "prompt_tokens_details": {"cached_tokens": r.cached}, "completion_tokens_details": {"reasoning_tokens": 0}}
            ex = {"usage": usage}
            if not a.no_timings:
                ex["tensorfold"] = {"engine": "fake", "time_to_first_token": round(vt_first, 4), "prefill_seconds": round((r.t_first or 0) - (r.t_pf_start or 0), 4),
                                    "seconds": round(dec, 4), "tokens_per_second": round((r.tokens - 1) / dec, 2) if dec > 0 else 0.0}
                ex["tensorfold"].update(getattr(r, "tf_extra", {}))      # a subclass (tools/fakepd.py) adds e.g. pulled_prefix
            return ex

        try:
            if stream:
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n")
                await writer.drain()
            nevents = 0
            first = True
            body_text = []
            while True:
                getter = asyncio.ensure_future(r.q.get())
                done, _ = await asyncio.wait({getter, gone}, return_when=asyncio.FIRST_COMPLETED)
                if getter not in done:
                    getter.cancel()
                    m.cancel(r)
                    return
                kind, i0, n, vt = getter.result()
                if kind == "tok":
                    txt = m.tok_text(i0, n)
                    body_text.append(txt)
                    if stream:
                        if self.fault["mode"] == "drop_mid" and nevents >= int(self.fault.get("after", 2)):
                            writer.transport.abort()
                            m.cancel(r)
                            return
                        data = chunk(txt, role=first)
                        writer.write(("%x\r\n" % len(data)).encode() + data + b"\r\n")
                        await writer.drain()
                        nevents += 1
                        first = False
                    continue
                # done
                if stream:
                    out = chunk("", finish="length", extra=None if incl else final_extra())
                    if incl:
                        out += chunk("", empty=True, extra=final_extra())
                    out += b"data: [DONE]\n\n"
                    writer.write(("%x\r\n" % len(out)).encode() + out + b"\r\n0\r\n\r\n")
                    await writer.drain()
                else:
                    msg = "".join(body_text)
                    if chat:
                        o = {"id": cid, "object": "chat.completion", "created": created, "model": a.model,
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": msg}, "finish_reason": "length"}]}
                    else:
                        o = {"id": cid, "object": "text_completion", "created": created, "model": a.model, "choices": [{"index": 0, "text": msg, "finish_reason": "length"}]}
                    o.update(final_extra())
                    await send_response(writer, 200, json.dumps(o).encode())
                return
        except (ConnectionError, OSError):
            m.cancel(r)
        finally:
            wt.cancel()
            if r.t_done is None:
                m.cancel(r)


async def amain(a):
    if os.environ.get("TFROUTER_SLOW_CALLBACK_S"):
        asyncio.get_running_loop().set_debug(True)
        asyncio.get_running_loop().slow_callback_duration = float(os.environ["TFROUTER_SLOW_CALLBACK_S"])
    s = Server(a)
    s.m.task = asyncio.ensure_future(s.m.run())
    srv = await asyncio.start_server(s.handle, a.host, a.port, limit=1 << 20, backlog=256)
    port = srv.sockets[0].getsockname()[1]
    print("listening on %d" % port, flush=True)
    async with srv:
        await srv.serve_forever()


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=("S", "M"), required=True)
    ap.add_argument("--name")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--speed", type=float, default=10.0, help="virtual seconds per wall second")
    ap.add_argument("--slots", type=int)
    ap.add_argument("--max-context", type=int)
    ap.add_argument("--model")
    ap.add_argument("--api-key-file")
    ap.add_argument("--cpt", type=float, default=3.3, help="characters per token of this fake tokenizer")
    ap.add_argument("--msg-overhead", type=int, default=4)
    ap.add_argument("--prefill-scale", type=float, default=1.0, help=">1: slower prefill than the measured table")
    ap.add_argument("--decode-scale", type=float, default=1.0, help=">1: faster decode than the measured table")
    ap.add_argument("--cache-tokens", type=int)
    ap.add_argument("--no-timings", action="store_true", help="omit the tensorfold runtime block")
    a = ap.parse_args(argv)
    a.model = a.model or PRESETS[a.profile]["model"]
    return a


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(amain(parse_args()))
