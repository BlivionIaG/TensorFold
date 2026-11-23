#!/usr/bin/env python3
"""replay.py: replay the simulator's workload mixes (chat / RAG / agent / long-doc / mixed / batch) through the REAL router process
against two fake backends (S, M) running the simulator's machine model in fast-forwarded real time, and report the same metrics as
bench/sim.py on the same traces.

    python3 tools/replay.py run --wl chat --rate 0.12 --seed 1 --horizon 300 --speed 10 --setups A,B,router-ll,router-cost [--sim]
    python3 tools/replay.py list

Setups: A = client -> fake S directly; B = client -> fake M directly; router-cost / router-ll = client -> tfrouter (policy.mode cost / ll) -> S, M;
        router-cost-exact = same with exact tokenizer counts (tokens.mode exact); router-cost-noaff = affinity off;
        router-e2 = the policy of the E2 run (no admission control, plain output EWMA, interference over the whole request, 0.3 s primary bias only).
Everything is in VIRTUAL seconds (wall x speed).  The traces are the simulator's generators with the same seed and horizon, so `--sim`
(run bench/sim.py's DES on the identical trace for the matching setup) is a like-for-like comparison.
"""
import argparse
import asyncio
import json
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
from tfrouter.config import PRESETS                          # noqa: E402
from tfrouter.httpio import SseScanner, Upstream             # noqa: E402

try:
    import sim                                               # noqa: E402  (tools/sim.py)
except ImportError:                                          # pragma: no cover
    sim = None

SIM_NAME = {"A": "A  S alone", "B": "B  M alone", "router-ll": "A+B least-loaded", "router-cost": "A+B cost-aware (no remote)", "router-e2": "A+B cost-aware (no remote)"}
SETUPS = ["A", "B", "router-ll", "router-cost", "router-cost-exact", "router-cost-noaff", "router-e2"]
E2_POLICY = {"policy": {"interf_overlap": False, "primary_bias_frac": 0.0}, "tokens": {"output_warmup": False}, "admission": {"enabled": False}}


# --------------------------------------------------------------------------------------------------------------------------------
# processes
# --------------------------------------------------------------------------------------------------------------------------------
class Proc:
    def __init__(self, args, env=None, log=None):
        self.args = args
        self.p = None
        self.log = log                                   # stderr goes here (a path), else it is dropped

    async def start(self, want_port=True):
        self.p = await asyncio.create_subprocess_exec(*self.args, stdout=asyncio.subprocess.PIPE, stderr=(None if os.environ.get('TFROUTER_SLOW_CALLBACK_S') else (open(self.log, 'ab') if self.log else asyncio.subprocess.DEVNULL)), cwd=ROOT)
        if want_port:
            line = await asyncio.wait_for(self.p.stdout.readline(), 20)
            self.port = int(line.split()[-1])
        return self

    def stop(self):
        if self.p and self.p.returncode is None:
            self.p.terminate()


async def get_json(port, path):
    u = await Upstream.open("http", "127.0.0.1", port, 3)
    try:
        await u.send("GET", path, "x", [])
        await u.read_head()
        return json.loads(await u.read_all())
    finally:
        u.close()


async def post_json(port, path, obj):
    u = await Upstream.open("http", "127.0.0.1", port, 3)
    try:
        await u.send("POST", path, "x", [("Content-Type", "application/json")], json.dumps(obj).encode())
        await u.read_head()
        return json.loads(await u.read_all())
    finally:
        u.close()


# --------------------------------------------------------------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------------------------------------------------------------
class Prompter:
    """builds chat messages whose fake-backend token count is n_target (same formula as the fake tokenizer)"""
    SYSTEM = "You are a helpful assistant for the benchmark."

    def __init__(self, cpt, msg_overhead=4):
        self.cpt, self.mo = cpt, msg_overhead

    def chars_for(self, n_target, nmsgs):
        return max(64, int((n_target - self.mo * nmsgs - 2) * self.cpt))

    def first(self, n_target, rng):
        total = self.chars_for(n_target, 2)
        c = max(64, total - len(self.SYSTEM))
        return [{"role": "system", "content": self.SYSTEM}, {"role": "user", "content": rng.randbytes(c // 2 + 1).hex()[:c]}]

    def next_turn(self, hist, reply, n_target, rng):
        msgs = hist + [{"role": "assistant", "content": reply}]
        so_far = sum(len(m["content"]) for m in msgs)
        total = self.chars_for(n_target, len(msgs) + 1)
        c = max(64, total - so_far)
        msgs.append({"role": "user", "content": rng.randbytes(c // 2 + 1).hex()[:c]})
        return msgs


# --------------------------------------------------------------------------------------------------------------------------------
# client
# --------------------------------------------------------------------------------------------------------------------------------
class Result:
    __slots__ = ("rid", "t_arr", "t_first", "t_done", "gaps", "n_in", "n_out", "cls", "backend", "status", "err", "tokens", "cached", "turn")

    def __init__(self, rid, n_in, n_out, cls, turn=0):
        self.rid, self.n_in, self.n_out, self.cls, self.turn = rid, n_in, n_out, cls, turn
        self.t_arr = self.t_first = self.t_done = None
        self.gaps, self.backend, self.status, self.err, self.tokens, self.cached = [], None, None, None, 0, 0


class Client:
    def __init__(self, port, speed, model, class_of, host="127.0.0.1", key=None, temperature=None, extra=None):
        self.port, self.speed, self.model, self.class_of, self.host, self.key, self.temperature, self.extra = port, speed, model, class_of, host, key, temperature, extra
        self.t0 = time.monotonic()

    def vt(self):
        return (time.monotonic() - self.t0) * self.speed

    async def request(self, msgs, res, extra_headers=()):
        """one streaming chat completion; fills res; returns the reply text (or None)"""
        res.t_arr = self.vt()
        hdrs = [("Content-Type", "application/json")] + list(extra_headers) + ([("Authorization", "Bearer " + self.key)] if self.key else [])
        if self.class_of(res.cls) != "interactive":
            hdrs.append(("X-Request-Class", self.class_of(res.cls)))
        body = {"model": self.model, "messages": msgs, "stream": True, "max_tokens": res.n_out}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.extra:
            body.update(self.extra)
        reply = []
        u = None
        try:
            u = await Upstream.open("http", self.host, self.port, 5)
            await u.send("POST", "/v1/chat/completions", "%s:%d" % (self.host, self.port), hdrs, json.dumps(body).encode())
            res.status = await u.read_head()
            res.backend = u.headers.get("x-router-backend")
            sc, last = SseScanner(), None
            async for chunk in u.body():
                t = self.vt()
                for p in sc.feed(chunk):
                    if p == b"[DONE]":
                        continue
                    o = json.loads(p)
                    if o.get("error"):
                        res.err = o["error"].get("message", "error")
                    ch = o.get("choices") or []
                    txt = (ch[0].get("delta") or {}).get("content") if ch else None
                    if txt:
                        reply.append(txt)
                        if res.t_first is None:
                            res.t_first = t
                        if last is not None:
                            res.gaps.append(t - last)
                        last = t
                    us = o.get("usage")
                    if us:
                        res.tokens = us.get("completion_tokens", res.tokens)
                        res.cached = (us.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            if res.status == 200 and not res.err and res.t_first is not None:
                res.t_done = self.vt()
        except Exception as e:                                   # noqa: BLE001
            res.err = "%s: %s" % (type(e).__name__, e)
        finally:
            if u:
                u.close()
        return "".join(reply)


async def run_workload(client, wl, rate, seed, horizon, drain, prompter, S_port=None, M_port=None):
    gen = sim.WORKLOADS[wl][0]
    rng = random.Random(seed)
    reqs, convs = gen(rng, rate, horizon)                      # identical to sim.run_one's trace
    results = []
    tasks = []

    async def one(r):
        await asyncio.sleep(max(0.0, (r.t_arr - client.vt()) / client.speed))
        res = Result(r.id, r.n_in, r.n_out, r.cls)
        results.append(res)
        await client.request(prompter.first(r.n_in, random.Random(seed * 1000003 + int(r.id))), res)

    async def conv(c):
        r0 = c.first
        await asyncio.sleep(max(0.0, (r0.t_arr - client.vt()) / client.speed))
        crng = random.Random(seed * 7919 + c.id)
        msgs, reply = None, ""
        for ti, (n_in, n_out) in enumerate(c.turns):
            res = Result((c.id, ti), n_in, n_out, "agent", ti)
            results.append(res)
            msgs = prompter.first(n_in, crng) if ti == 0 else prompter.next_turn(msgs, reply, n_in, crng)
            reply = await client.request(msgs, res)
            if res.t_done is None:
                break
            if ti < len(c.turns) - 1:
                await asyncio.sleep(c.thinks[ti] / client.speed)

    for r in reqs:
        tasks.append(asyncio.ensure_future(one(r)))
    for c in convs:
        tasks.append(asyncio.ensure_future(conv(c)))
    end_v = horizon + drain
    try:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), max(1.0, (end_v - client.vt()) / client.speed))
    except asyncio.TimeoutError:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return results


# --------------------------------------------------------------------------------------------------------------------------------
# one setup
# --------------------------------------------------------------------------------------------------------------------------------
def class_of(wl):
    return lambda cls: "background" if cls == "batch" else "interactive"


async def run_setup(setup, wl, rate, seed, horizon, speed, drain, fake_cpt, outdir=None, router_cfg=None, tag=None):
    fakes = {}
    procs = []
    try:
        for name in ("S", "M"):
            f = await Proc([sys.executable, os.path.join(HERE, "fakebackend.py"), "--profile", name, "--speed", str(speed), "--cpt", str(fake_cpt)]).start()
            fakes[name] = f
            procs.append(f)
        router = None
        if setup.startswith("router"):
            cfg = {"time_scale": speed, "log": {"access": False}, "listen": {"host": "127.0.0.1", "port": 0},
                   "backends": {"S": {"url": "http://127.0.0.1:%d" % fakes["S"].port}, "M": {"url": "http://127.0.0.1:%d" % fakes["M"].port}},
                   "policy": {"mode": "ll" if setup == "router-ll" else "cost"}}
            if setup == "router-cost-exact":
                cfg["tokens"] = {"mode": "exact"}
            if setup == "router-cost-noaff":
                cfg["affinity"] = {"enabled": False}
            if setup == "router-e2":
                _deep(cfg, E2_POLICY)
            if router_cfg:
                _deep(cfg, router_cfg)
            tmp = os.path.join(outdir or "/tmp", "rcfg-%s-%s-%d-%s.json" % (setup, wl, os.getpid(), tag or "x"))
            json.dump(cfg, open(tmp, "w"))
            dec = os.path.join(outdir or "/tmp", "decisions-%s-%s-%s-%s.jsonl" % (tag or "run", setup, wl, rate))
            if os.path.exists(dec):
                os.unlink(dec)
            # port 0 is not printable by the router: pick a free port
            import socket
            s = socket.socket()
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
            s.close()
            cfg["listen"]["port"] = port
            json.dump(cfg, open(tmp, "w"))
            router = await Proc([sys.executable, "-m", "tfrouter", "--config", tmp, "--explain-file", dec], None, log=dec.replace("decisions-", "router-stderr-").replace(".jsonl", ".log")).start(want_port=False)
            procs.append(router)
            await asyncio.sleep(0.7)
            target, model = port, "DeepSeek-V4.1-Flash"
        elif setup == "A":
            target, model = fakes["S"].port, PRESETS["S"]["model"]
        elif setup == "B":
            target, model = fakes["M"].port, PRESETS["M"]["model"]
        else:
            raise SystemExit("unknown setup " + setup)
        client = Client(target, speed, model, class_of(wl))
        prompter = Prompter(fake_cpt)
        results = await run_workload(client, wl, rate, seed, horizon, drain, prompter)
        stats = {n: await get_json(f.port, "/fake/stats") for n, f in fakes.items()}
        rstate = await get_json(target, "/router/state") if router else None
        return results, stats, rstate
    finally:
        for p in procs:
            p.stop()
        await asyncio.sleep(0.1)


def _deep(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep(dst[k], v)
        else:
            dst[k] = v


# --------------------------------------------------------------------------------------------------------------------------------
# metrics (the simulator's own summarize on the measured records)
# --------------------------------------------------------------------------------------------------------------------------------
class _M:
    def __init__(self, busy):
        self.busy_time = busy


class _Sim:
    def __init__(self, reqs, stats):
        self.reqs = reqs
        self.mach = {"S": _M(stats["S"]["busy_pf_v"] + stats["S"]["busy_dec_v"]), "M": _M(stats["M"]["busy_pf_v"] + stats["M"]["busy_dec_v"])}


def to_sim_reqs(results):
    out = []
    for r in results:
        q = sim.Req(r.rid, r.t_arr or 0.0, r.n_in, r.n_out, r.cls)
        q.t_arr = r.t_arr
        if r.t_done is not None and r.t_first is not None:
            q.t_first, q.t_done, q.gaps = r.t_first, r.t_done, list(r.gaps)
        out.append(q)
    return out


def summarize(wl, results, stats, horizon, slo=None):
    return sim.summarize([_Sim(to_sim_reqs(results), stats)], wl, horizon=horizon, slo=slo)


def backend_split(results, horizon):
    w0 = 0.15 * horizon
    rows = [r for r in results if r.t_arr is not None and w0 <= r.t_arr <= horizon and r.backend]
    n = len(rows)
    toM = [r for r in rows if r.backend == "M"]
    big = [r for r in rows if r.n_in > 8000]
    return {"n": n, "M_share": len(toM) / n if n else float("nan"), "big_to_M": (sum(1 for r in big if r.backend == "M") / len(big)) if big else float("nan"),
            "errors": sum(1 for r in results if r.err), "unfinished": sum(1 for r in results if r.t_done is None and not r.err)}


def sim_summary(wl, rate, setup, seed, horizon):
    sm = sim.run_one(wl, rate, sim.SETUPS[SIM_NAME[setup]], sim.PARAM, seed, horizon)
    r = sim.summarize([sm], wl, horizon=horizon)
    toM = [q for q in sm.reqs if q.t_arr >= 0.15 * horizon and q.t_arr <= horizon and q.D == "M"]
    allr = [q for q in sm.reqs if q.t_arr >= 0.15 * horizon and q.t_arr <= horizon and q.D]
    r["M_share"] = len(toM) / len(allr) if allr else float("nan")
    return r, sm


COLS = [("TTFT p50", "ttft50", "{:.1f}"), ("TTFT p95", "ttft95", "{:.1f}"), ("ITL p50 ms", "itl50", "{:.0f}"), ("ITL p95 ms", "itl95", "{:.0f}"), ("E2E p50", "e2e50", "{:.1f}"),
        ("E2E p95", "e2e95", "{:.1f}"), ("out tok/s", "out_tps", "{:.1f}"), ("good req/min", "good", "{:.1f}"), ("fin %", "fin", "{:.0%}"), ("util S", "utilS", "{:.0%}"),
        ("util M", "utilM", "{:.0%}")]


def fmt(v, f):
    return "nan" if v != v else f.format(v)


def print_table(title, rows, out=print):
    out("\n**%s**\n" % title)
    out("| setup | " + " | ".join(c[0] for c in COLS) + " | M share |")
    out("|---|" + "---|" * (len(COLS) + 1))
    for name, r in rows:
        out("| %s | " % name + " | ".join(fmt(r.get(k, float("nan")), f) for _, k, f in COLS) + " | %s |" % fmt(r.get("M_share", float("nan")), "{:.0%}"))


# --------------------------------------------------------------------------------------------------------------------------------
async def amain(a):
    wl, rate, horizon = a.wl, a.rate, a.horizon
    all_rows, sim_rows = [], []
    summaries = {}
    for setup in a.setups.split(","):
        t = time.time()
        results, stats, rstate = await run_setup(setup, wl, rate, a.seed, horizon, a.speed, a.drain, a.cpt, a.outdir, tag=a.tag)
        r = summarize(wl, results, stats, horizon)
        sp = backend_split(results, horizon)
        r["M_share"] = sp["M_share"]
        all_rows.append(("%s (real-time, speed %g)" % (setup, a.speed), r))
        summaries[setup] = dict(summary=r, split=sp, router=rstate, stats=stats)
        print("# %s done in %.0f s wall; %s" % (setup, time.time() - t, sp), file=sys.stderr, flush=True)
        if a.sim and setup in SIM_NAME:
            rs, _ = sim_summary(wl, rate, setup, a.seed, horizon)
            sim_rows.append(("%s (sim.py DES, same trace)" % SIM_NAME[setup], rs))
            summaries[setup]["sim"] = rs
    print_table("%s @ %g/s seed %d horizon %g s: real router + fake backends vs simulator" % (wl, rate, a.seed, horizon), all_rows + sim_rows)
    if a.json:
        json.dump(summaries, open(a.json, "w"), default=str)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    r = sub.add_parser("run")
    r.add_argument("--wl", default="chat")
    r.add_argument("--rate", type=float, default=0.12)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--horizon", type=float, default=300.0)
    r.add_argument("--drain", type=float, default=300.0)
    r.add_argument("--speed", type=float, default=10.0)
    r.add_argument("--setups", default="A,router-cost")
    r.add_argument("--cpt", type=float, default=3.3, help="the fake backends' chars per token")
    r.add_argument("--sim", action="store_true", help="also run bench/sim.py's DES on the same trace for the matching setups")
    r.add_argument("--outdir", default="/tmp")
    r.add_argument("--tag")
    r.add_argument("--json")
    a = ap.parse_args()
    if a.cmd == "list":
        print("workloads:", ", ".join(sim.WORKLOADS), "\nsetups:", ", ".join(SETUPS))
        return
    asyncio.run(amain(a))


if __name__ == "__main__":
    main()
