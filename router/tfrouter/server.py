"""The OpenAI-compatible proxy: listener, request classification, decision, forwarding (SSE passthrough, cancellation, failover),
health/load probing, explain log, admin routes."""
import asyncio
import contextlib
import hashlib
import hmac
import json
import os
import sys
import time
from collections import deque

from . import admission, pd, policy
from .backend import Backend, Inflight, KeyFile
from .httpio import (HttpError, SseScanner, Upstream, error_body, head_bytes, read_request, send_error, send_response, simple_request)
from .prefix import PrefixIndex
from .tokens import TokenEstimator, request_size

CHAT_PATHS = {"/v1/chat/completions", "/chat/completions"}
COMP_PATHS = {"/v1/completions", "/completions"}
TOKENIZE_PATHS = {"/v1/tokenize", "/tokenize", "/v1/detokenize", "/detokenize"}


class ClientGone(Exception):
    pass


class Clock:
    """router seconds = wall seconds * scale (scale > 1 only in offline tests against fast-forwarded fake backends)"""

    def __init__(self, scale=1.0):
        self.scale = scale
        self.t0 = time.monotonic()

    def now(self):
        return (time.monotonic() - self.t0) * self.scale

    async def sleep(self, s):
        await asyncio.sleep(max(0.0, s) / self.scale)

    def wall(self, s):
        """router seconds -> wall seconds (for asyncio timeouts)"""
        return s / self.scale


class ClientConn:
    """the downstream socket: watches for the client hanging up while a response is being produced"""

    def __init__(self, reader, writer):
        self.reader, self.writer = reader, writer
        self.gone = asyncio.get_running_loop().create_future()
        self.head_sent = False
        self.body_bytes = 0
        self._watch = asyncio.ensure_future(self._watcher())

    async def _watcher(self):
        try:
            while True:
                if not await self.reader.read(4096):
                    break
        except Exception:
            pass
        if not self.gone.done():
            self.gone.set_result(True)

    def mark_gone(self):
        if not self.gone.done():
            self.gone.set_result(True)

    async def write(self, data):
        if self.gone.done():
            raise ClientGone()
        try:
            self.writer.write(data)
            await self.writer.drain()
        except (ConnectionError, OSError):
            self.mark_gone()
            raise ClientGone()

    def stop_watch(self):
        """stop watching for a hang-up but keep the socket: the caller still has an answer to send"""
        self._watch.cancel()

    def close(self):
        self._watch.cancel()
        with contextlib.suppress(Exception):
            self.writer.close()


def parse_event(payload):
    """one SSE `data:` payload -> dict of what the router cares about (first-token flag, reply text, usage, runtime block, finish)"""
    if payload == b"[DONE]":
        return {"done": True}
    try:
        o = json.loads(payload)
    except ValueError:
        return {}
    out = {}
    if not isinstance(o, dict):
        return out
    if o.get("error"):
        out["error"] = o["error"]
    ch = o.get("choices") or []
    if ch and isinstance(ch[0], dict):
        c0 = ch[0]
        delta = c0.get("delta") or {}
        text = delta.get("content") if delta else None
        if text is None:
            text = c0.get("text")
        rc = delta.get("reasoning_content") if delta else None
        if text:
            out["text"] = text
        if text or rc or (delta and delta.get("tool_calls")):
            out["token"] = True
        if c0.get("finish_reason"):
            out["finish"] = c0["finish_reason"]
    u = o.get("usage")
    if isinstance(u, dict):
        out["usage"] = u
    tf = o.get("tensorfold")
    if isinstance(tf, dict):
        out["tf"] = tf
    return out


def usage_info(usage, tf):
    info = {}
    if usage:
        info["prompt_tokens"] = usage.get("prompt_tokens")
        info["completion_tokens"] = usage.get("completion_tokens")
        d = usage.get("prompt_tokens_details") or {}
        info["cached_tokens"] = d.get("cached_tokens") or 0
    if tf:
        if tf.get("prefill_seconds") is not None:
            info["prefill_seconds"] = float(tf["prefill_seconds"])
        if tf.get("seconds") is not None:
            info["decode_seconds"] = float(tf["seconds"])
        for k in ("pulled_prefix", "remote_prefix"):      # PD: a prefix the server pulled from another machine (reply's tensorfold block)
            if tf.get(k) is not None:
                info[k] = int(tf[k])
    return info


def parse_load(obj):
    """/v1/load JSON -> number of model requests the backend has in progress (all clients), or None.
    [INFERENCE] docs/api.md names `http_active` (by route group) and the request counts but not their JSON nesting: look the names up anywhere."""
    if not isinstance(obj, dict):
        return None
    ha = obj.get("http_active")
    if isinstance(ha, dict):
        return sum(v for v in ha.values() if isinstance(v, (int, float)))
    if isinstance(ha, (int, float)):
        return int(ha)
    return None


def parse_load_tokens(obj):
    """/v1/load JSON -> prompt tokens waiting or filling on the server (all clients), or None: `waiting_prompt_tokens` + `filling_prompt_tokens_left`"""
    if not isinstance(obj, dict) or obj.get("waiting_prompt_tokens") is None:
        return None
    try:
        return int(obj["waiting_prompt_tokens"]) + int(obj.get("filling_prompt_tokens_left") or 0)
    except (TypeError, ValueError):
        return None


class Router:
    def __init__(self, cfg, log=None):
        self.cfg = cfg
        self.clock = Clock(cfg["time_scale"])
        self.backends = {n: Backend(c, cfg) for n, c in cfg["backends"].items() if c["enabled"]}
        self.order = list(self.backends.values())
        self.tokens = TokenEstimator(cfg)
        self.prefix = PrefixIndex(cfg)
        self.prefix.budget = {n: int(c.get("prefix_cache_tokens") or 0) for n, c in cfg["backends"].items()}
        self.ring = deque(maxlen=int(cfg["log"]["ring"]))
        self.seq = 0
        self.server = None
        self.tasks = []
        self.stats = {"requests": 0, "dry_run": 0, "rejected": 0, "no_backend": 0, "tokenize_calls": 0, "tokenize_fail": 0, "client_cancels": 0, "failovers": 0, "errors": 0,
                      "held": 0, "queue_full": 0, "queue_timeout": 0}
        self.client_keys = KeyListFile(cfg.get("client_api_key_file"))
        self._explain_fp = None
        if cfg["log"]["explain_file"]:
            self._explain_fp = open(cfg["log"]["explain_file"], "a", buffering=1)
        self.log = log or (lambda m: sys.stderr.write(m + "\n"))
        self.dry_run = cfg["mode"]["dry_run"]
        self.shadow = cfg["mode"]["shadow"]
        self.pd = pd.State(cfg["pd"])
        self.gated = bool(cfg["admission"]["enabled"]) and not self.shadow       # shadow mode is a transparent proxy to one backend
        self.gate = admission.WaitQueue()
        for b in self.order:
            b.on_free = self.gate.kick

    # ------------------------------------------------------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------------------------------------------------------
    async def start(self, host=None, port=None):
        l = self.cfg["listen"]
        self.server = await asyncio.start_server(self.handle, host or l["host"], l["port"] if port is None else port, limit=1 << 20, backlog=256)
        for b in self.order:
            self.tasks.append(asyncio.ensure_future(self.probe_loop(b)))
        self.tasks.append(asyncio.ensure_future(self.housekeeping()))
        return self.server.sockets[0].getsockname()[1]

    async def stop(self):
        for t in self.tasks:
            t.cancel()
        for t in self.tasks:
            with contextlib.suppress(BaseException):
                await t
        if self.server:
            self.server.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self.server.wait_closed(), 2.0)
        if self._explain_fp:
            self._explain_fp.close()

    async def housekeeping(self):
        while True:
            await self.clock.sleep(30.0)
            self.prefix.prune(self.clock.now())

    # ------------------------------------------------------------------------------------------------------------------
    # connection handler
    # ------------------------------------------------------------------------------------------------------------------
    async def handle(self, reader, writer):
        try:
            req = await read_request(reader, writer, int(self.cfg["max_body_mb"]) * 1024 * 1024)
            if req is None:
                return
            if req.path not in ("/health", "/") and not self.client_keys.check(req):
                await send_error(writer, 401, "missing or invalid API key", "invalid_api_key", [("WWW-Authenticate", "Bearer")])
                return
            await self.route(req, reader, writer)
        except HttpError as e:
            with contextlib.suppress(Exception):
                await send_error(writer, e.status, e.message, e.code)
        except (ConnectionError, asyncio.IncompleteReadError, asyncio.CancelledError):
            pass
        except Exception as e:
            self.stats["errors"] += 1
            self.log("tfrouter: internal error: %r" % (e,))
            with contextlib.suppress(Exception):
                await send_error(writer, 500, "internal router error", "router_error")
        finally:
            with contextlib.suppress(Exception):
                writer.close()

    async def route(self, req, reader, writer):
        p, m = req.path, req.method
        if m == "GET" and p in ("/health", "/"):
            return await self.r_health(writer)
        if m == "GET" and p in ("/v1/models", "/models"):
            return await self.r_models(writer)
        if p.startswith("/router/"):
            return await self.r_admin(req, writer)
        if m == "POST" and (p in CHAT_PATHS or p in COMP_PATHS):
            return await self.proxy(req, reader, writer, "chat" if p in CHAT_PATHS else "completions")
        if m == "POST" and p in TOKENIZE_PATHS:
            return await self.passthrough(req, writer)
        raise HttpError(404, "route not served by tfrouter: %s %s" % (m, p), "not_found")

    # ------------------------------------------------------------------------------------------------------------------
    # simple routes
    # ------------------------------------------------------------------------------------------------------------------
    async def r_health(self, writer):
        now = self.clock.now()
        st = {b.name: {"available": b.available(now), "breaker": b.breaker.state, "health_ok": b.health_ok} for b in self.order}
        ok = any(v["available"] for v in st.values())
        await send_response(writer, 200 if ok else 503, json.dumps({"status": "ok" if ok else "unavailable", "backends": st}).encode())

    async def r_models(self, writer):
        mc = max(b.profile.max_context for b in self.order)
        pub = self.cfg["public_model"]
        ids = [pub] + [pub + s for s in self.cfg["class_suffixes"]]
        data = [{"id": i, "object": "model", "created": int(time.time()), "owned_by": "tfrouter", "context_length": mc, "max_model_len": mc} for i in ids]
        await send_response(writer, 200, json.dumps({"object": "list", "data": data}).encode())

    async def r_admin(self, req, writer):
        now = self.clock.now()
        if req.path == "/router/state":
            o = {"now": round(now, 2), "backends": [b.state(now) for b in self.order], "stats": self.stats,
                 "tokens": {"chars_per_token": round(self.tokens.cpt_of(), 3), "samples": self.tokens.samples, "output_ewma": {k: round(v, 1) for k, v in self.tokens.out.items()},
                            "exact_calls": self.tokens.exact_calls, "exact_fail": self.tokens.exact_fail},
                 "affinity": {"entries": len(self.prefix.table), "sessions": len(self.prefix.sessions), "lookups": self.prefix.lookups, "hits": self.prefix.hits, "invalidations": self.prefix.invalidations},
                 "mode": {"dry_run": self.dry_run, "shadow": self.shadow, "policy": self.cfg["policy"]["mode"]}}
            o["admission"] = {"enabled": self.cfg["admission"]["enabled"], "waiting": len(self.gate),
                              "per_backend": {b.name: {"in_use": b.in_use(), "max_inflight": b.max_inflight, "waiting": self.gate.depth(b.name), "queue_max": b.queue_max} for b in self.order}}
            if pd.enabled(self.cfg):
                o["pd"] = self.pd.as_dict(now)
        elif req.path == "/router/explain":
            o = list(self.ring)
        elif req.path == "/router/config":
            o = {k: v for k, v in self.cfg.items() if k != "backends"}
            o["backends"] = {n: {k: v for k, v in b.cfg.items() if k not in ("api_key_file", "api_key_env")} for n, b in self.backends.items()}
        else:
            raise HttpError(404, "no such admin route", "not_found")
        await send_response(writer, 200, json.dumps(o).encode())

    async def passthrough(self, req, writer):
        """tokenizer routes: stateless, the same checkpoint on both machines -> any available backend, primary first"""
        now = self.clock.now()
        for b in sorted(self.order, key=lambda x: x.role != "primary"):
            if not b.available(now):
                continue
            try:
                st, h, data = await simple_request(b.scheme, b.host, b.port, "POST", b.base + req.path, self._auth(b) + [("Content-Type", "application/json")], req.body, timeout=10.0)
            except Exception:
                continue
            return await send_response(writer, st, data, [("X-Router-Backend", b.name)], h.get("content-type", "application/json"))
        raise HttpError(503, "no backend available", "no_backend")

    # ------------------------------------------------------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------------------------------------------------------
    def _auth(self, b):
        k = b.key.get()
        return [("Authorization", "Bearer " + k)] if k else []

    def resolve_class(self, req, model):
        cfg = self.cfg
        h = req.header(cfg["class_header"])
        if h:
            h = h.strip().lower()
            if h in cfg["policy"]["classes"]:
                return h, model
            raise HttpError(400, "unknown request class %r (use one of %s)" % (h, ", ".join(cfg["policy"]["classes"])), "invalid_class")
        if isinstance(model, str):
            for suf, c in cfg["class_suffixes"].items():
                if model.endswith(suf):
                    return c, model[:-len(suf)]
        return cfg["default_class"], model

    def _boundaries(self):
        lim = []
        for b in self.order:
            lim.append(b.profile.max_context)
            if b.guard_rows:
                lim.append(b.guard_rows)
        return lim

    async def exact_count(self, endpoint, body, anch):
        """ask a backend's tokenizer endpoint; None if no backend answers"""
        key = (endpoint, anch.points[-1] if anch and anch.points else None, anch.total_chars if anch else None)
        if anch and anch.points:
            c = self.tokens.cache_get(key)
            if c is not None:
                return c
        now = self.clock.now()
        for b in sorted(self.order, key=lambda x: x.role != "primary"):
            if not b.available(now) or not b.cfg.get("tokenizer"):
                continue
            if endpoint == "chat":
                payload = {"model": b.model, "messages": body.get("messages"), "add_generation_prompt": True}
                for k in ("tools", "reasoning_effort", "chat_template_kwargs"):
                    if body.get(k) is not None:
                        payload[k] = body[k]
            else:
                payload = {"model": b.model, "prompt": body.get("prompt"), "add_special_tokens": False}
            self.tokens.exact_calls += 1
            try:
                st, h, data = await simple_request(b.scheme, b.host, b.port, "POST", b.base + b.cfg["tokenize_path"],
                                                   self._auth(b) + [("Content-Type", "application/json")], json.dumps(payload).encode(), timeout=self.tokens.exact_timeout)
                if st != 200:
                    raise ValueError("status %d" % st)
                n = int(json.loads(data)["count"])
            except Exception:
                self.tokens.exact_fail += 1
                continue
            if anch and anch.points:
                self.tokens.cache_put(key, n)
            return n
        return None

    # ------------------------------------------------------------------------------------------------------------------
    # the proxy
    # ------------------------------------------------------------------------------------------------------------------
    async def proxy(self, req, reader, writer, endpoint):
        cfg, now = self.cfg, self.clock.now()
        try:
            body = json.loads(req.body or b"{}")
        except ValueError:
            raise HttpError(400, "request body is not valid JSON", "invalid_json")
        if not isinstance(body, dict):
            raise HttpError(400, "request body must be a JSON object")
        if endpoint == "chat" and not isinstance(body.get("messages"), list):
            raise HttpError(400, "'messages' must be a list")
        if endpoint == "completions" and body.get("prompt") is None:
            raise HttpError(400, "'prompt' is required")
        self.seq += 1
        rid = "r%d" % self.seq
        self.stats["requests"] += 1
        stream = bool(body.get("stream"))
        cls, public = self.resolve_class(req, body.get("model"))
        known = {cfg["public_model"]} | {b.model for b in self.order}
        if not cfg["accept_any_model"] and public not in known and public is not None:
            raise HttpError(404, "model %r not found" % public, "model_not_found")

        # ---- prompt size ------------------------------------------------------------------------------------------------
        t = self.tokens
        chars, nmsgs = request_size(endpoint, body)
        if endpoint == "chat":
            seed = ""
            if body.get("tools") or body.get("chat_template_kwargs") or body.get("response_format"):
                # what changes the rendered prefix besides the messages: tools, template kwargs (SessionTier: they re-key the backend's cache)
                seed = json.dumps([body.get("tools"), body.get("chat_template_kwargs"), body.get("response_format")], sort_keys=True, default=str)
            anch = self.prefix.anchors_chat(body["messages"], seed)
        elif isinstance(body["prompt"], str):
            anch = self.prefix.anchors_text(body["prompt"])
        else:
            anch = None
        n_in, n_exact, n_src = t.estimate(chars, nmsgs), chars < 0, "ids" if chars < 0 else "estimate"
        if not n_exact and t.want_exact(n_in, self._boundaries()):
            n = await self.exact_count(endpoint, body, anch)
            self.stats["tokenize_calls"] += 1
            if n is not None:
                t.observe(chars, nmsgs, n)
                n_in, n_exact, n_src = n, True, "tokenize"
            else:
                self.stats["tokenize_fail"] += 1
        mt = body.get("max_tokens") or body.get("max_completion_tokens")
        hint = req.header(cfg["expected_output_header"].lower())
        out_est = t.out_est(cls, mt, int(hint) if hint and hint.isdigit() else None)
        ctx_out = min(int(mt), int(cfg["tokens"]["ctx_output_cap"])) if mt else out_est
        now = self.clock.now()
        sid = req.header(cfg["session_header"].lower())
        cached = {}
        for b in self.order:
            c1, _ = self.prefix.lookup(anch, b.name, now)
            c2 = self.prefix.lookup_session(sid, b.name, now, n_in)
            g = int(cfg["backends"][b.name].get("resume_grid") or 1)      # S resumes on a 2048-token grid (TF_DS_REPLAY_FLOOR): cached = floor(h/g)*g
            cached[b.name] = max(c1, c2) // g * g
        slo_h = req.header(cfg["slo_ttft_header"].lower())
        try:
            slo = float(slo_h) if slo_h else None
        except ValueError:
            raise HttpError(400, "bad %s header" % cfg["slo_ttft_header"])
        force = req.header(cfg["route_header"].lower())
        if force and force not in self.backends and not (pd.enabled(cfg) and force == pd.SM):
            raise HttpError(400, "unknown backend %r in %s" % (force, cfg["route_header"]))
        s_only = pd.features(cfg, body, req.header(cfg["pd"]["exact_header"].lower()))
        ctx = policy.ReqCtx(rid, cls, n_in, n_exact, out_est, ctx_out, cached, slo, force, sid, s_only, self.pd.off_until > now)
        if self.gated:
            ctx.ahead = self.gate.ahead(admission.prio_of(cfg, cls), admission.INF)
        dec = policy.decide(cfg, self.order, ctx, now)
        holder = {}

        def get_client():
            if "c" not in holder:
                holder["c"] = ClientConn(reader, writer)
            return holder["c"]
        waited = 0.0
        if not self.dry_run and not self.shadow:
            try:
                dec, waited = await admission.bind(self, ctx, dec, get_client, lambda t: policy.decide(cfg, self.order, ctx, t))
            except ClientGone:
                self.stats["client_cancels"] += 1
                get_client().close()
                return
            except BaseException:
                if "c" in holder:
                    holder["c"].close()
                raise
        used = dec
        if self.shadow and not force:
            ctx2 = policy.ReqCtx(rid, cls, n_in, n_exact, out_est, ctx_out, cached, slo, self.shadow, sid)
            used = policy.decide(cfg, self.order, ctx2, now)
        record = {"ts": round(now, 3), "id": rid, "endpoint": req.path, "class": cls, "stream": stream, "n_in": n_in, "n_src": n_src, "out_est": out_est,
                  "slo_ttft": None if not cfg["policy"]["classes"][cls].get("slo") else round(policy.slo_ttft_for(cfg, ctx), 2), "session": sid,
                  "cached": cached, "held_s": round(waited, 3), **({"s_only": s_only} if s_only else {}), **dec.as_dict()}
        if self.shadow and not force:
            record["shadow"] = used.choice
        self.ring.append(record)
        if cfg["log"]["explain"]:
            line = json.dumps(record, separators=(",", ":"))
            if self._explain_fp:
                self._explain_fp.write(line + "\n")
            else:
                self.log("explain " + line)
        decision_hdr = [("X-Router-Decision", json.dumps(record, separators=(",", ":")))] if req.header(cfg["explain_header"].lower()) in ("1", "true") else []

        if self.dry_run:
            self.stats["dry_run"] += 1
            await send_response(writer, 200, json.dumps({"router_decision": record}).encode(), decision_hdr)
            return
        if used.status != 200:
            self.stats["no_backend" if used.status == 503 and used.retry_after is None else "rejected"] += 1
            hdrs = [("Retry-After", str(used.retry_after or 5))] if used.status == 503 else []
            if "c" in holder:
                holder["c"].stop_watch()
            await send_error(writer, used.status, used.error, "router_" + (used.rule or "error"), hdrs + decision_hdr)
            return

        client = get_client()
        try:
            await self.forward_loop(client, req, body, endpoint, stream, rid, cls, used, anch, chars, nmsgs, n_in, ctx, decision_hdr, record)
        except ClientGone:
            pass
        finally:
            client.close()

    # ------------------------------------------------------------------------------------------------------------------
    async def race(self, aw, client, timeout):
        """await aw unless the client hangs up (ClientGone) or `timeout` wall seconds pass (TimeoutError)"""
        task = asyncio.ensure_future(aw)
        try:
            done, _ = await asyncio.wait({task, client.gone}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            task.cancel()
            raise
        if task in done:
            return task.result()
        task.cancel()
        with contextlib.suppress(BaseException):
            await task
        if client.gone.done():
            raise ClientGone()
        raise asyncio.TimeoutError()

    async def forward_loop(self, client, req, body, endpoint, stream, rid, cls, dec, anch, chars, nmsgs, n_in, ctx, decision_hdr, record):
        fo = self.cfg["failover"]
        tried = []
        last_err = "no backend tried"
        for name in dec.ranking:
            if len(tried) >= int(fo["max_attempts"]):
                break
            now = self.clock.now()
            if not pd.available(self, name, dec, now) or name not in dec.preds:
                continue
            b = self.backends[dec.paths[name].backend] if name in dec.paths else self.backends[name]
            if self.gated and not b.lane_free():
                # a failover target without a free lane waits for one (the first candidate of an admitted request always has one)
                if not tried or not await admission.wait_lane(self, ctx, b, client):
                    last_err = "backend %s has no free slot" % b.name
                    continue
                now = self.clock.now()
            pred = dec.preds[name]
            rec = Inflight(rid, cls, stream, n_in, max(1, n_in - ctx.cached.get(b.name, 0)), ctx.cached.get(b.name, 0), ctx.out_est)
            rec.meta = record
            pd.admit(self, rec, name, dec, now, pred)
            b.counts["chosen"] += 1
            b.breaker.reserve(now)
            self.prefix.record(anch, b.name, n_in, now, uncached=rec.rows)
            if ctx.session:
                self.prefix.record_session(ctx.session, b.name, n_in, now)
            tried.append(name)
            res, err = await self.forward_one(client, b, rec, req, body, endpoint, stream, pred, anch, chars, nmsgs, ctx, decision_hdr, tried, dec.paths.get(name))
            if res == "done":
                return
            last_err = err
            self.stats["failovers"] += 1
            b.counts["failovers_from"] += 1
        # nothing worked
        self.stats["no_backend"] += 1
        msg = "all attempted backends failed (%s): %s" % (",".join(tried) or "none available", last_err)
        if not client.head_sent:
            await send_error(client.writer, 503, msg, "backends_failed", [("Retry-After", "5")] + decision_hdr)
            client.head_sent = True
        else:
            await self._sse_error(client, msg)

    async def _sse_error(self, client, msg):
        ev = b"data: " + json.dumps({"error": {"message": msg, "type": "server_error", "code": "backend_failed"}}).encode() + b"\n\ndata: [DONE]\n\n"
        with contextlib.suppress(ClientGone):
            await client.write(ev)

    def _fail(self, b, rec, now, cancelled=False):
        b.abort(rec, now, cancelled=cancelled, failed=not cancelled)

    async def forward_one(self, client, b, rec, req, body, endpoint, stream, pred, anch, chars, nmsgs, ctx, decision_hdr, tried, route=None):
        """-> ('done', None) the exchange is over (success, passthrough error, client gone) ; ('retry', reason) try another backend"""
        cfg, fo, clock = self.cfg, self.cfg["failover"], self.clock
        up = None
        path = "/v1/chat/completions" if endpoint == "chat" else "/v1/completions"
        b2 = dict(body)
        b2["model"] = b.model
        if route is not None and route.directive and pd.enabled(cfg):       # PD: tell M's server where this prompt is prefilled (SM: force, MM: never)
            b2["prefill"] = route.directive
        payload = json.dumps(b2, separators=(",", ":")).encode()
        headers = self._auth(b) + [("Content-Type", "application/json")]
        if stream:
            headers.append(("Accept", "text/event-stream"))
        fb = clock.wall(fo["first_byte_timeout_s"] + fo["first_byte_per_ptok_s"] * (pred.ttft if stream else pred.e2e))
        idle = clock.wall(fo["idle_timeout_s"])
        try:
            try:
                up = await self.race(Upstream.open(b.scheme, b.host, b.port, fo["connect_timeout_s"]), client, fo["connect_timeout_s"] + 1)
                await self.race(up.send("POST", b.base + path, "%s:%s" % (b.host, b.port), headers, payload), client, fb)
                status = await self.race(up.read_head(), client, fb)
            except ClientGone:
                raise
            except (OSError, ConnectionError, asyncio.TimeoutError, asyncio.IncompleteReadError, HttpError, ValueError) as e:
                now = clock.now()
                b.breaker.failure(now)
                self._fail(b, rec, now)
                return "retry", "%s: %s" % (type(e).__name__, e)

            now = clock.now()
            if status in fo["retry_statuses"] or status >= 500:
                data = b""
                with contextlib.suppress(Exception):
                    data = await asyncio.wait_for(up.read_all(1 << 16), 2.0)
                if status != 429:
                    b.breaker.failure(now)
                else:
                    b.breaker.release_trial()
                self._fail(b, rec, now)
                return "retry", "HTTP %d %s" % (status, data[:200].decode("utf-8", "replace"))
            if status != 200:
                # client error (bad request, bad key...): the same answer from any backend, pass it through
                data = await self.race(up.read_all(1 << 20), client, fb)
                b.breaker.release_trial()
                b.abort(rec, now)
                if status in (401, 403):
                    self.log("tfrouter: backend %s rejected our API key (HTTP %d)" % (b.name, status))
                await send_response(client.writer, status, data, [("X-Router-Backend", b.name)] + decision_hdr, up.headers.get("content-type", "application/json"))
                client.head_sent = True
                return "done", None

            base_hdrs = [("X-Router-Backend", b.name), ("X-Router-Request-Id", rec.id)] + decision_hdr
            if not stream:
                data = await self.race(up.read_all(), client, fb)
                now = clock.now()
                try:
                    o = json.loads(data)
                except ValueError:
                    o = {}
                u = o.get("usage") if isinstance(o, dict) else None
                tf = o.get("tensorfold") if isinstance(o, dict) else None
                reply = None
                if isinstance(o, dict) and o.get("choices"):
                    c0 = o["choices"][0]
                    reply = (c0.get("message") or {}).get("content") if endpoint == "chat" else c0.get("text")
                if not client.head_sent:
                    await send_response(client.writer, 200, data, base_hdrs, up.headers.get("content-type", "application/json"))
                    client.head_sent = True
                self._complete(b, rec, now, usage_info(u, tf), anch, chars, nmsgs, reply)
                return "done", None

            # ---- streaming ------------------------------------------------------------------------------------------------
            if not client.head_sent:
                await client.write(head_bytes(200, [("Content-Type", up.headers.get("content-type", "text/event-stream")), ("Cache-Control", "no-cache"),
                                                    ("Connection", "close")] + base_hdrs))
                client.head_sent = True
            sc = SseScanner()
            usage = tf = None
            reply = []
            finished = False
            it = up.body().__aiter__()
            while True:
                try:
                    chunk = await self.race(_next(it), client, idle)
                except ClientGone:
                    raise
                except (OSError, ConnectionError, asyncio.TimeoutError, asyncio.IncompleteReadError, ValueError) as e:
                    now = clock.now()
                    b.breaker.failure(now)
                    self._fail(b, rec, now)
                    if client.body_bytes == 0 and len(tried) < int(fo["max_attempts"]):
                        return "retry", "%s mid-stream before the first byte: %s" % (type(e).__name__, e)
                    await self._sse_error(client, "backend %s failed mid-stream (%s)" % (b.name, type(e).__name__))
                    return "done", None
                if chunk is None:
                    break
                await client.write(chunk)
                client.body_bytes += len(chunk)
                for payload_ in sc.feed(chunk):
                    ev = parse_event(payload_)
                    if not ev:
                        continue
                    if ev.get("token"):
                        b.first_token(rec, clock.now())
                    if "text" in ev:
                        reply.append(ev["text"])
                    if "usage" in ev:
                        usage = ev["usage"]
                    if "tf" in ev:
                        tf = ev["tf"]
                    if ev.get("finish") or ev.get("done"):
                        finished = True
                    if ev.get("error"):
                        finished = False
            now = clock.now()
            if not finished and usage is None:
                b.breaker.failure(now)
                self._fail(b, rec, now)
                await self._sse_error(client, "backend %s ended the stream early" % b.name)
                return "done", None
            self._complete(b, rec, now, usage_info(usage, tf), anch, chars, nmsgs, "".join(reply) if reply else None)
            return "done", None
        except ClientGone:
            self.stats["client_cancels"] += 1
            if up:
                up.abort()
            b.breaker.release_trial()
            self._fail(b, rec, clock.now(), cancelled=True)
            return "done", None
        finally:
            if up:
                up.close()

    def _complete(self, b, rec, now, info, anch, chars, nmsgs, reply):
        b.breaker.success(now)
        b.finish(rec, now, info)
        pd.complete(self, b, rec, now, info)
        pt, ct = info.get("prompt_tokens"), info.get("completion_tokens")
        if pt:
            self.tokens.observe(chars, nmsgs, pt)
            cached_seen = info.get("cached_tokens")
            if cached_seen is not None and rec.cached_est >= 4096 and cached_seen < 0.5 * rec.cached_est:
                self.prefix.invalidate(anch, b.name)          # we expected a hit and the backend did not have it: evicted (or restarted)
                self.stats["affinity_misses"] = self.stats.get("affinity_misses", 0) + 1
            self.prefix.record(anch, b.name, pt, now)
            if self.prefix.record_reply and reply and anch is not None and ct:
                ext = self.prefix.extend_with_reply(anch, reply)
                self.prefix.record_extension(ext, anch.total_chars, b.name, pt, ct, now)
        if ct:
            self.tokens.observe_out(rec.cls, ct)
        out = {"backend": b.name, "ok": True, "total_s": round(now - rec.t_admit, 3), "prompt_tokens": pt, "completion_tokens": ct,
               "cached_tokens": info.get("cached_tokens", 0), "pred_ttft": round(rec.est_first - rec.t_admit, 3) if rec.est_first is not None else None}
        if rec.t_first is not None and rec.stream:
            out["ttft_s"] = round(rec.t_first - rec.t_admit, 3)
        if rec.meta is not None:
            rec.meta["outcome"] = out
        if self.cfg["log"]["access"]:
            self.log("access " + json.dumps({"id": rec.id, "class": rec.cls, **out}, separators=(",", ":")))

    # ------------------------------------------------------------------------------------------------------------------
    # health + load probing
    # ------------------------------------------------------------------------------------------------------------------
    async def probe_loop(self, b):
        h = self.cfg["health"]
        next_load = 0.0
        load_off_until = 0.0
        while True:
            now = self.clock.now()
            ok = await self.probe_health(b)
            if ok and h["external_load"] and now >= next_load and now >= load_off_until:
                next_load = now + h["load_poll_s"]
                r = await self.probe_load(b)
                if r == "off":
                    load_off_until = now + 10 * h["interval_s"]
                    b.load_off = True
                elif r == "ok":
                    b.load_off = False
            await self.clock.sleep(h["interval_s"] if ok else max(0.5, h["interval_s"] / 3))

    async def probe_health(self, b):
        h = self.cfg["health"]
        now = self.clock.now()
        try:
            st, hd, data = await simple_request(b.scheme, b.host, b.port, "GET", b.base + b.cfg["health_path"], self._auth(b), timeout=self.clock.wall(h["timeout_s"]))
            ok = st == 200
            live = slots = None
            if ok:
                with contextlib.suppress(Exception):
                    o = json.loads(data)
                    live = o.get("live")
                    slots = o.get("max_batch_size")
        except Exception:
            ok, live, slots = False, None, None
        b.set_health(ok, now, live, slots)
        if not ok:
            b.breaker.failure(now)
        elif b.load_off and isinstance(live, dict) and isinstance(live.get("connections"), (int, float)):
            b.set_external(live["connections"])      # no /v1/load (the production build answers 404): /health live.connections = running + waiting requests
        return ok

    async def probe_load(self, b):
        h = self.cfg["health"]
        try:
            st, hd, data = await simple_request(b.scheme, b.host, b.port, "GET", b.base + b.cfg["load_path"], self._auth(b), timeout=self.clock.wall(h["load_timeout_s"]))
        except Exception:
            return "err"
        if st in (401, 403, 404):
            return "off"
        if st != 200:
            return "err"
        try:
            obj = json.loads(data)
            b.set_external(parse_load(obj))
            if pd.enabled(self.cfg) and b.name == self.cfg["pd"]["prefill_backend"]:
                b.set_prefill_load(parse_load_tokens(obj))
        except ValueError:
            return "err"
        return "ok"


async def _next(it):
    try:
        return await it.__anext__()
    except StopAsyncIteration:
        return None


class KeyListFile:
    """the router's own client API keys (same file format as TensorFold's); no file = open"""

    def __init__(self, path):
        self.path, self.keys, self.mt = path, [], None
        self._reload()

    def _reload(self):
        if not self.path:
            return
        try:
            mt = os.stat(self.path).st_mtime_ns
        except OSError:
            self.keys, self.mt = [], None
            return
        if mt == self.mt:
            return
        self.mt, keys = mt, []
        import re
        pat = re.compile(r"^[A-Za-z0-9._\-]+(\[[^\]]*\])?\s*:\s*(\S+)$")
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    m = pat.match(line)
                    keys.append((m.group(2) if m else line).encode())
        self.keys = keys

    def check(self, req):
        if not self.path:
            return True
        self._reload()
        tok = req.header("authorization", "")
        if tok.lower().startswith("bearer "):
            tok = tok[7:].strip()
        else:
            tok = req.header("x-api-key", "")
        tb = tok.encode()
        ok = False
        for k in self.keys:                      # compare against every key, no early exit
            ok |= hmac.compare_digest(hashlib.sha256(k).digest(), hashlib.sha256(tb).digest())
        return ok
