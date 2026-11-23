#!/usr/bin/env python3
"""fakepd.py: the M server of a PD+ fleet for offline router tests: tools/fakebackend.py's M machine plus `--prefill-from` behaviour.

A request whose body says `"prefill": "force"` (and whose prompt has at least `--min-rows` rows the server does not hold) is
  1. exported by the S fake: a 1-token completion of the same prompt is POSTed to `--prefill-from` (S prefills it FIFO, like any prompt),
  2. landed: the hand-off time H(n) (tfrouter.pd.handoff_s) passes on M's timeline,
  3. decoded here: the prompt counts as cached, the first token costs one tail step, the reply carries tensorfold.pulled_prefix / remote_prefix.
`"prefill": "never"` and an absent field prefill on M itself (a bad value is a 400, as on the real server).
`POST /fake/pair {"down": true}` makes forced requests fall back to local prefill (the real server does the same when the pair or the link is
down); `GET /fake/pdstats` counts the directives seen, the pulls and the fallbacks.
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import fakebackend                                         # noqa: E402
from tfrouter import pd as pdmod                            # noqa: E402
from tfrouter.config import DEFAULTS                        # noqa: E402
from tfrouter.httpio import send_response, simple_request   # noqa: E402

TAIL_S = DEFAULTS["pd"]["handoff"]["tail_s"]


class PdMachine(fakebackend.Machine):
    def __init__(self, a):
        super().__init__(a)
        self.next_pull = None

    def submit(self, r):
        p, self.next_pull = self.next_pull, None          # set synchronously just before Server.completion creates the Req and submits it
        if p:
            r.pulled = True
            r.tf_extra = p
        super().submit(r)

    def _start(self, r):
        super()._start(r)
        if getattr(r, "pulled", False):
            r.cached = max(0, r.n_in - 1)
            r.rows = 1
            r.job.total = r.job.rem = r.job.chunk = TAIL_S


class PdServer(fakebackend.Server):
    def __init__(self, a, source, min_rows=4096):
        super().__init__(a)
        self.m = PdMachine(a)
        self.source = source                       # (host, port) of the S fake, or None
        self.min_rows = min_rows
        self.pair_down = False
        self.pdstats = {"force": 0, "never": 0, "auto": 0, "pulled": 0, "fallback": 0}

    async def route(self, req, reader, writer):
        if req.path == "/fake/pair":
            self.pair_down = bool(json.loads(req.body or b"{}").get("down"))
            return await send_response(writer, 200, json.dumps({"down": self.pair_down}).encode())
        if req.path == "/fake/pdstats":
            return await send_response(writer, 200, json.dumps(self.pdstats).encode())
        return await super().route(req, reader, writer)

    async def completion(self, req, reader, writer, chat):
        b = json.loads(req.body or b"{}")
        directive = b.get("prefill", "auto")
        if directive not in ("auto", "force", "never"):
            return await send_response(writer, 400, b'{"error":{"message":"prefill: expected auto, force or never"}}')
        self.pdstats[directive] += 1
        n_in, text = self.count_prompt(b)
        held = self.m.cache_lookup(text, n_in)
        if directive == "force" and self.source and n_in - held >= self.min_rows:
            if self.pair_down:
                self.pdstats["fallback"] += 1
            else:
                exp = {k: v for k, v in b.items() if k not in ("prefill", "model")}
                exp.update(max_tokens=1, stream=False)
                host, port = self.source
                st, _, _ = await simple_request("http", host, port, "POST", "/v1/chat/completions" if "messages" in b else "/v1/completions",
                                                [("Content-Type", "application/json")], json.dumps(exp).encode(), timeout=600)
                if st == 200:
                    await asyncio.sleep(pdmod.handoff_s(DEFAULTS["pd"], n_in) / self.m.speed)
                    self.m.next_pull = {"pulled_prefix": n_in - 1, "remote_prefix": n_in - 1}
                    self.pdstats["pulled"] += 1
                else:
                    self.pdstats["fallback"] += 1
        return await super().completion(req, reader, writer, chat)


async def amain(a, source_url):
    src = None
    if source_url:
        hp = source_url.split("//")[-1].split(":")
        src = (hp[0], int(hp[1]))
    s = PdServer(a, src, a.min_rows)
    s.m.task = asyncio.ensure_future(s.m.run())
    srv = await asyncio.start_server(s.handle, a.host, a.port, limit=1 << 20, backlog=256)
    print("listening on %d" % srv.sockets[0].getsockname()[1], flush=True)
    async with srv:
        await srv.serve_forever()


def main(argv=None):
    import argparse
    import contextlib
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill-from", help="URL of the S fake (http://host:port)")
    ap.add_argument("--min-rows", type=int, default=4096)
    a, rest = ap.parse_known_args(argv)
    fa = fakebackend.parse_args(["--profile", "M"] + rest)
    fa.min_rows = a.min_rows
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(amain(fa, a.prefill_from))


if __name__ == "__main__":
    main()
