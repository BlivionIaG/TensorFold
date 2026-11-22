#!/usr/bin/env python3
"""llmbench.py: streaming latency / throughput bench for an OpenAI-compatible chat server (stdlib only, python >= 3.9).

Every request streams (SSE). Chunk arrival times are taken with time.perf_counter() on the client (run it ON the server's
host, through loopback, to keep the network out of the numbers). Prompts are slices of the --corpus file behind a unique nonce at the
very start, so no prefix/prompt cache can hit; the token count of a request is the
server's own usage.prompt_tokens, never an estimate. temperature 0, thinking off (chat_template_kwargs enable_thinking=false).

  llmbench.py --url http://127.0.0.1:8888 ttft   --sizes 1024x3,4096x3,32768x2,65536x1 [--warm 1024]
  llmbench.py --url ...                 decode  --tokens 256 --reps 3 [--ctx 8192]
  llmbench.py --url ...                 conc    --n 1,2,4 --prompt 0 --tokens 256      (decode throughput, short prompts)
  llmbench.py --url ...                 conc    --n 1,2,4 --prompt 4096 --tokens 1     (aggregate prefill throughput)
  llmbench.py --url ...                 hol     --long 8192,16384 --tokens 1500 --after 20   (B is sent once A has streamed 20 chunks)
  llmbench.py --url ...                 calib

Options common to all modes: --out FILE (append one JSON line per request), --abort-file FILE (if the file appears, every
open stream is cut at once: used as a safety brake while loading a production server), --tag TEXT (nonce prefix).

ITL: gaps between consecutive stream events. Chunks that arrive within 2 ms of each other (a speculative round that
delivers several tokens in one burst) are merged into ONE event, so an event gap is what a user sees between two screen
updates; tok/event is reported so a per-token figure can be derived. tok/s = (completion_tokens - 1) / (last - first chunk).
"""
import argparse
import http.client
import json
import os
import random
import socket
import sys
import threading
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
MERGE_EPS = 0.002  # seconds: chunks closer than this are one event (one decode round)
CHARS_PER_TOKEN = None  # calibrated chars/token (pdbench sets it): last-resort token estimate when the server gives no count

_open_socks = set()
_open_lock = threading.Lock()
_abort_flag = threading.Event()


def start_abort_monitor(path):
    def run():
        while True:
            if os.path.exists(path):
                _abort_flag.set()
                with _open_lock:
                    socks = list(_open_socks)
                for s in socks:
                    try:
                        s.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
            time.sleep(0.1)
    threading.Thread(target=run, daemon=True).start()


def pct(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def events_of(chunks, toks=None):
    """merge chunk times (seconds from request start) closer than MERGE_EPS: returns (event times, chunks per event).
    With toks (tokens per chunk) the second list is TOKENS per event instead."""
    ev = []
    cnt = []
    for i, t in enumerate(chunks):
        w = toks[i] if toks else 1
        if ev and t - ev[-1] < MERGE_EPS:
            cnt[-1] += w
        else:
            ev.append(t)
            cnt.append(w)
    return ev, cnt


def chunk_tokens(r):
    """tokens carried by every content chunk (an SSE event): returns (list of ints, method). Methods, best first:
    'ids' the chunk lists its token ids (choice/delta token_ids or logprobs.content), 'usage' the chunk carries a cumulative
    usage.completion_tokens (delta to the previous one), 'chars' the request's completion_tokens split over the chunks in
    proportion to their characters (the reply's own chars/token), 'cpt' characters / the calibrated chars-per-token when the
    server reports no completion_tokens, 'chunks' one token per chunk (nothing else known)."""
    n, nc = r["chunk_n"], len(r["chunks"])
    if nc and all(x is not None for x in n):
        return [max(int(x), 1) for x in n], r["chunk_src"]
    ch = r["chunk_chars"]
    tot = float(sum(ch))
    if nc and tot > 0 and r["ct"]:
        want = float(r["ct"])
    elif nc and tot > 0 and CHARS_PER_TOKEN:
        want = tot / CHARS_PER_TOKEN
    else:
        return [1] * nc, "chunks"
    out, cum, prev = [], 0, 0
    for c in ch:
        cum += c
        out.append(max(int(round(want * cum / tot)) - prev, 1))
        prev += out[-1]
    return out, "chars" if r["ct"] else "cpt"


class Server(object):
    def __init__(self, url, model="x", temperature=0.0, extra=None, api_key=None):
        u = urllib.parse.urlparse(url)
        self.host, self.port = u.hostname, u.port or 80
        self.model, self.temperature, self.extra = model, temperature, extra or {}
        self.api_key = api_key

    def chat(self, prompt, max_tokens, hook=None, messages=None, **kw):
        """one streaming request (messages, if given, replaces the single user prompt). Returns a dict (see keys below).
        Never raises on abort: ok=False, err=text. hook(n_chunks_so_far) is called after every content chunk."""
        body = {"model": self.model, "messages": messages or [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                "temperature": self.temperature, "stream": True, "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False}}
        body.update(self.extra)
        body.update(kw)
        raw = json.dumps(body).encode()
        r = {"ok": False, "err": None, "chunks": [], "ct": None, "pt": None, "cached": None, "finish": None, "tf": None,
             "ttft": None, "t_hdr": None, "text": [], "exact_mode": None, "speculative": None,
             "chunk_n": [], "chunk_chars": [], "chunk_src": None}
        last_ct = 0
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        conn = http.client.HTTPConnection(self.host, self.port, timeout=float(os.environ.get("LB_TIMEOUT") or 3600))
        t0 = time.perf_counter()
        r["t0_abs"] = time.time()
        try:
            conn.connect()
            with _open_lock:
                _open_socks.add(conn.sock)
            conn.request("POST", "/v1/chat/completions", raw, headers)
            resp = conn.getresponse()
            r["t_hdr"] = time.perf_counter() - t0
            if resp.status != 200:
                r["err"] = "HTTP %d %s" % (resp.status, resp.read(300))
                return r
            while True:
                line = resp.readline()
                if not line:
                    break
                line = line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    break
                now = time.perf_counter() - t0
                try:
                    j = json.loads(d)
                except ValueError:
                    continue
                if j.get("usage"):
                    u = j["usage"]
                    r["pt"], r["ct"] = u.get("prompt_tokens"), u.get("completion_tokens")
                    ct_now = u.get("completion_tokens")
                    r["cached"] = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
                if j.get("tensorfold"):
                    r["tf"] = j["tensorfold"]
                if j.get("exact_mode"):
                    r["exact_mode"] = j["exact_mode"]
                if j.get("speculative"):
                    r["speculative"] = j["speculative"]
                if j.get("error") and not r["err"]:
                    r["err"] = "stream error: %s" % json.dumps(j["error"])[:300]
                for c in j.get("choices", []):
                    dl = c.get("delta") or {}
                    t = dl.get("content") or dl.get("reasoning_content")
                    if t:
                        r["chunks"].append(now)
                        r["text"].append(t)
                        r["chunk_chars"].append(len(t))
                        ids = c.get("token_ids") or dl.get("token_ids") or (c.get("logprobs") or {}).get("content")
                        if isinstance(ids, list) and ids:
                            r["chunk_n"].append(len(ids))
                            r["chunk_src"] = "ids"
                        elif j.get("usage") and ct_now and ct_now > last_ct:
                            r["chunk_n"].append(ct_now - last_ct)
                            if r["chunk_src"] != "ids":
                                r["chunk_src"] = "usage"
                        else:
                            r["chunk_n"].append(None)
                        if j.get("usage") and ct_now:
                            last_ct = ct_now
                        if hook:
                            hook(len(r["chunks"]))
                    if c.get("finish_reason"):
                        r["finish"] = c["finish_reason"]
            r["ok"] = True
        except Exception as e:  # noqa: BLE001 (abort shows up as a reset/EOF: report it, do not die)
            r["err"] = "%s: %s" % (type(e).__name__, e)
        finally:
            r["wall"] = time.perf_counter() - t0
            with _open_lock:
                if conn.sock is not None:
                    _open_socks.discard(conn.sock)
            try:
                conn.close()
            except OSError:
                pass
        if r["chunks"]:
            r["ttft"] = r["chunks"][0]
        r["tok_per_chunk"], r["tok_src"] = chunk_tokens(r)
        r["text"] = "".join(r["text"])[:200]
        return r


class Corpus(object):
    """Prompt text: random paragraphs of the corpus (split on blank lines, trimmed to 1500 chars) joined until the wanted
    length, so every prompt mixes docs and code in the corpus' own proportions and chars/token stays within a few percent
    from prompt to prompt. cpt (chars per token) is calibrated once and refined after every answered request (learn())."""

    def __init__(self, path):
        text = open(path, "r", encoding="utf-8", errors="ignore").read()
        self.paras = [p.strip()[:1500] for p in text.split("\n\n") if len(p.strip()) > 40]
        self.cpt = None  # chars per token
        self.n = 0

    def body(self, nchars, rng):
        out, tot = [], 0
        while tot < nchars:
            p = self.paras[rng.randrange(len(self.paras))]
            out.append(p)
            tot += len(p) + 2
        return "\n\n".join(out)[:nchars]

    def prompt(self, ntok, tag, rng):
        self.n += 1
        nonce = "%s-%d-%06d" % (tag, self.n, rng.randrange(10 ** 6))
        head = "[doc %s] Read the document below, then answer the question after it.\n\n" % nonce
        tail = "\n\nQuestion: reply with the single word OK."
        nchars = max(100, int(ntok * self.cpt) - 60)
        return head + self.body(nchars, rng) + tail

    def learn(self, prompt, pt):
        """blend the observed chars/token of a finished request into cpt."""
        if pt and pt > 200:
            self.cpt = 0.6 * self.cpt + 0.4 * (len(prompt) / float(pt))


def calibrate(srv, corp, rng, log=print):
    """chars per prompt token from two probes (max_tokens 1): slope removes the fixed chat-template overhead."""
    corp.cpt = 4.0
    pts = []
    for nchars in (6000, 24000):
        p = "[doc calib-%06d]\n%s" % (rng.randrange(10 ** 6), corp.body(nchars, rng))
        r = srv.chat(p, 1)
        if not r["ok"] or not r["pt"]:
            raise SystemExit("calibration request failed: %s" % r["err"])
        pts.append((len(p), r["pt"]))
    (c0, t0), (c1, t1) = pts
    corp.cpt = (c1 - c0) / float(t1 - t0)
    log("calib: %.3f chars/token (probes %s)" % (corp.cpt, pts))
    return corp.cpt


STORIES = ["a lighthouse keeper who finds a message in a bottle", "a clockmaker in a city where time runs backwards",
           "two rival chess masters snowed in at a mountain hotel", "a botanist who discovers a plant that sings",
           "a courier crossing a desert with a package she must not open", "a retired detective and a missing violin",
           "an astronomer who receives a signal that repeats her own name", "a ferry captain on a river that never ends"]


def story_prompt(i):
    return ("Write a long, detailed short story (at least 1500 words, no preamble, do not stop early) about %s."
            % STORIES[i % len(STORIES)])


def summarize(r):
    ch = r["chunks"]
    out = {"t0_abs": r.get("t0_abs"), "ok": r["ok"], "err": r["err"], "pt": r["pt"], "ct": r["ct"], "cached": r["cached"], "finish": r["finish"],
           "ttft": r["ttft"], "wall": r.get("wall")}
    if ch:
        ev, cnt = events_of(ch)
        gaps = [b - a for a, b in zip(ev, ev[1:])]
        _, tpe = events_of(ch, r["tok_per_chunk"])
        tg = [g / float(n) for g, n in zip(gaps, tpe[1:]) for _ in range(n)]
        dec = ch[-1] - ch[0]
        ct = r["ct"] or len(ch)
        out.update({"chunks": len(ch), "events": len(ev), "decode_s": dec, "tok_s": (ct - 1) / dec if dec > 0 else None,
                    "tok_per_event": ct / float(len(ev)), "itl_p50": pct(gaps, 50), "itl_p95": pct(gaps, 95),
                    "itl_max": max(gaps) if gaps else None, "itl_mean": sum(gaps) / len(gaps) if gaps else None,
                    "stall": max(gaps) if gaps else None, "tok_itl_p50": pct(tg, 50), "tok_itl_p95": pct(tg, 95),
                    "tok_itl_p99": pct(tg, 99), "tok_src": r["tok_src"],
                    "ev_t": [round(t, 4) for t in ev], "ev_n": list(tpe)})
    if r["tf"]:
        out["tf"] = r["tf"]
    if r.get("exact_mode"):
        out["exact_mode"] = r["exact_mode"]
    if r.get("speculative"):
        out["speculative"] = r["speculative"]
    return out


def fmt(x, nd=3):
    return "nan" if x is None or x != x else ("%.*f" % (nd, x))


def line_of(s):
    return ("pt=%s ct=%s ttft=%ss decode=%ss tok/s=%s tok/event=%s ITLevent p50=%sms p95=%sms max=%sms finish=%s%s" % (
        s.get("pt"), s.get("ct"), fmt(s.get("ttft")), fmt(s.get("decode_s"), 2), fmt(s.get("tok_s"), 2),
        fmt(s.get("tok_per_event"), 2), fmt((s.get("itl_p50") or float("nan")) * 1000, 1),
        fmt((s.get("itl_p95") or float("nan")) * 1000, 1), fmt((s.get("itl_max") or float("nan")) * 1000, 1),
        s.get("finish"), "" if s["ok"] else " ERR=%s" % s["err"]))


class Out(object):
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()

    def put(self, rec):
        if not self.path:
            return
        with self.lock:
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")


# ------------------------------------------------------------------ modes
def mode_ttft(a, srv, corp, rng, out):
    specs = []
    for s in a.sizes.split(","):
        n, _, reps = s.partition("x")
        specs.append((int(n), int(reps or 3)))
    if a.warm:
        wp = corp.prompt(a.warm, a.tag + "w", rng)
        r = srv.chat(wp, 1)
        corp.learn(wp, r["pt"])
        print("warm-up %d: %s" % (a.warm, line_of(summarize(r))), flush=True)
    for n, reps in specs:
        res = []
        if a.warm_each and n <= 32768:
            wp = corp.prompt(n, a.tag + "w", rng)
            r = srv.chat(wp, 1)
            corp.learn(wp, r["pt"])
            print("warm-each %d: pt=%s ttft=%s" % (n, r["pt"], fmt(r["ttft"])), flush=True)
        for k in range(reps):
            tp = corp.prompt(n, a.tag, rng)
            r = srv.chat(tp, 1)
            corp.learn(tp, r["pt"])
            s = summarize(r)
            s.update({"mode": "ttft", "want": n, "rep": k})
            out.put(s)
            if not r["ok"]:
                print("want=%d rep=%d FAILED %s" % (n, k, r["err"]), flush=True)
                return
            ts = s["pt"] / s["ttft"]
            res.append((s["ttft"], s["pt"], ts))
            print("ttft want=%d rep=%d pt=%d ttft=%.3fs prefill=%.0f tok/s cached=%s tf=%s" % (
                n, k, s["pt"], s["ttft"], ts, s["cached"], json.dumps(s.get("tf")) if s.get("tf") else "-"), flush=True)
        ts = sorted(x[2] for x in res)
        tt = sorted(x[0] for x in res)
        print("SUMMARY want=%d pt~%d reps=%d ttft best=%.3f median=%.3f prefill tok/s best=%.0f median=%.0f" % (
            n, res[0][1], len(res), tt[0], tt[len(tt) // 2], ts[-1], ts[len(ts) // 2]), flush=True)


def mode_decode(a, srv, corp, rng, out):
    for k in range(a.reps):
        p = story_prompt(k)
        if a.ctx:
            p = corp.prompt(a.ctx, a.tag, rng).rsplit("\n\nQuestion:", 1)[0] + "\n\n" + p
        r = srv.chat(p, a.tokens)
        s = summarize(r)
        s.update({"mode": "decode", "rep": k, "ctx": a.ctx})
        out.put(s)
        print("decode ctx=%d rep=%d %s" % (a.ctx, k, line_of(s)), flush=True)
        if not r["ok"]:
            return


def mode_conc(a, srv, corp, rng, out):
    for n in [int(x) for x in a.n.split(",")]:
        for rep in range(a.reps):
            results = [None] * n
            prompts = []
            for i in range(n):
                p = story_prompt(i + 3 * rep)
                if a.prompt and a.tokens > 1:
                    p = corp.prompt(a.prompt, a.tag, rng).rsplit("\n\nQuestion:", 1)[0] + "\n\n" + p
                elif a.prompt:
                    p = corp.prompt(a.prompt, a.tag, rng)
                prompts.append(p)
            gate = threading.Barrier(n + 1)

            def work(i):
                gate.wait()
                results[i] = srv.chat(prompts[i], a.tokens)
            th = [threading.Thread(target=work, args=(i,)) for i in range(n)]
            for t in th:
                t.start()
            gate.wait()
            tstart = time.perf_counter()
            for t in th:
                t.join()
            wall = time.perf_counter() - tstart
            ss = [summarize(r) for r in results]
            for i, s in enumerate(ss):
                s.update({"mode": "conc", "n": n, "i": i, "rep": rep, "prompt": a.prompt})
                out.put(s)
                print("conc n=%d rep=%d req=%d %s" % (n, rep, i, line_of(s)), flush=True)
            ok = [(r, s) for r, s in zip(results, ss) if r["ok"] and r["chunks"]]
            if not ok:
                continue
            tot_ct = sum(s["ct"] for r, s in ok)
            tot_pt = sum(s["pt"] for r, s in ok)
            first = min(r["chunks"][0] for r, s in ok)
            last = max(r["chunks"][-1] for r, s in ok)
            # steady window: every request is decoding (latest first token .. earliest last token)
            w0 = max(r["chunks"][0] for r, s in ok)
            w1 = min(r["chunks"][-1] for r, s in ok)
            steady = None
            if w1 > w0 and a.tokens > 1:
                tok = 0.0
                for r, s in ok:
                    ev, cnt = events_of(r["chunks"])
                    tpe = s["tok_per_event"]
                    tok += tpe * sum(1 for t in ev if w0 <= t <= w1)
                steady = tok / (w1 - w0)
            agg = {"mode": "conc_agg", "n": n, "rep": rep, "prompt": a.prompt, "wall": wall, "total_ct": tot_ct,
                   "total_pt": tot_pt, "agg_decode_tok_s": (tot_ct - len(ok)) / (last - first) if last > first else None,
                   "steady_tok_s": steady, "max_ttft": max(s["ttft"] for r, s in ok),
                   "agg_prefill_tok_s": tot_pt / max(s["ttft"] for r, s in ok),
                   "per_req_tok_s": [s.get("tok_s") for r, s in ok]}
            out.put(agg)
            print("AGG n=%d rep=%d wall=%.2fs total_ct=%d agg_decode=%s tok/s steady(all decoding)=%s tok/s | prompt_tokens=%d "
                  "max_ttft=%.2fs agg_prefill=%.0f tok/s per-req tok/s=%s" % (
                      n, rep, wall, tot_ct, fmt(agg["agg_decode_tok_s"], 1), fmt(steady, 1), tot_pt, agg["max_ttft"],
                      agg["agg_prefill_tok_s"], ",".join(fmt(x, 1) for x in agg["per_req_tok_s"])), flush=True)


def phase_stats(ev, lo, hi):
    gaps = [b - a for a, b in zip(ev, ev[1:]) if lo <= b <= hi]
    if not gaps:
        return None
    return {"n": len(gaps), "p50": pct(gaps, 50), "p95": pct(gaps, 95), "max": max(gaps), "mean": sum(gaps) / len(gaps)}


def mode_hol(a, srv, corp, rng, out):
    for L in [int(x) for x in a.long.split(",")]:
        for rep in range(a.reps):
            res = {}
            ready = threading.Event()

            def hook(n):
                if n >= a.after:
                    ready.set()

            def run_a():
                res["a"] = srv.chat(story_prompt(rep + 1), a.tokens, hook=hook)
                ready.set()

            ta = threading.Thread(target=run_a)
            tA0 = time.perf_counter()
            ta.start()
            ready.wait()
            tB0 = time.perf_counter()
            bp = corp.prompt(L, a.tag, rng)
            res["b"] = srv.chat(bp, 1)
            corp.learn(bp, res["b"]["pt"])
            tB1 = time.perf_counter()
            ta.join()
            ra, rb = res["a"], res["b"]
            sa, sb = summarize(ra), summarize(rb)
            if not ra["chunks"]:
                print("hol L=%d: stream A produced nothing: %s" % (L, ra["err"]), flush=True)
                continue
            ev, cnt = events_of(ra["chunks"])
            b0, b1 = tB0 - tA0, tB0 - tA0 + (rb["ttft"] or (tB1 - tB0))
            ev_end = ra["chunks"][-1]
            before = phase_stats(ev, ev[0], b0)
            during = phase_stats(ev, b0, b1)
            after = phase_stats(ev, b1, ev_end)
            stall = max([(y - x, x) for x, y in zip(ev, ev[1:]) if y >= b0 and x <= b1] or [(0, 0)])
            rec = {"mode": "hol", "long": L, "rep": rep, "a": sa, "b": sb, "b_start": b0, "b_first_token": b1,
                   "a_events": len(ev), "before": before, "during": during, "after": after, "max_stall": stall[0],
                   "a_tokens_during": None}
            # tokens A emitted while B was prefilling
            tpe = sa.get("tok_per_event") or 1.0
            rec["a_tokens_during"] = tpe * sum(1 for t in ev if b0 <= t <= b1)
            out.put(rec)
            print("HOL long=%d rep=%d  B: pt=%s ttft=%.2fs (alone-ish) | A: ct=%s finish=%s decode %.1fs tok/s=%s" % (
                L, rep, sb["pt"], rb["ttft"] or -1, sa["ct"], sa["finish"], sa.get("decode_s") or 0, fmt(sa.get("tok_s"), 1)), flush=True)
            for name, ph in (("before", before), ("during", during), ("after", after)):
                if ph:
                    print("    A ITL %-6s events=%3d p50=%.1fms p95=%.1fms max=%.1fms mean=%.1fms" % (
                        name, ph["n"], ph["p50"] * 1000, ph["p95"] * 1000, ph["max"] * 1000, ph["mean"] * 1000), flush=True)
                else:
                    print("    A ITL %-6s (no events)" % name, flush=True)
            print("    longest A stall while B in flight: %.2fs; A tokens emitted while B prefilled: %.0f (B window %.2fs)" % (
                stall[0], rec["a_tokens_during"], b1 - b0), flush=True)
            if sa["finish"] != "length":
                print("    WARNING: A ended with finish=%s before max_tokens: raise --tokens" % sa["finish"], flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--corpus", required=True, help="text file of prompt material (paragraphs split on blank lines), e.g. the docs concatenated")
    ap.add_argument("--out", default=None)
    ap.add_argument("--abort-file", default=None)
    ap.add_argument("--tag", default="b")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--model", default="x")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--extra", default="{}", help="JSON merged into every request body")
    ap.add_argument("--key-file", default=None, help="file with an API key (Authorization: Bearer)")
    ap.add_argument("--cpt", type=float, default=None, help="chars per token (skip calibration)")
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("ttft")
    s.add_argument("--sizes", default="1024x3,4096x3,8192x3,16384x2,32768x2")
    s.add_argument("--warm", type=int, default=1024)
    s.add_argument("--warm-each", action="store_true", help="one untimed request at every size (<= 32768) before its reps")
    s = sub.add_parser("decode")
    s.add_argument("--tokens", type=int, default=256)
    s.add_argument("--reps", type=int, default=3)
    s.add_argument("--ctx", type=int, default=0)
    s = sub.add_parser("conc")
    s.add_argument("--n", default="1,2,4")
    s.add_argument("--prompt", type=int, default=0)
    s.add_argument("--tokens", type=int, default=256)
    s.add_argument("--reps", type=int, default=1)
    s = sub.add_parser("hol")
    s.add_argument("--long", default="8192,16384")
    s.add_argument("--tokens", type=int, default=1500)
    s.add_argument("--after", type=int, default=20, help="send B once A has streamed this many chunks")
    s.add_argument("--reps", type=int, default=1)
    sub.add_parser("calib")
    a = ap.parse_args()

    if a.abort_file:
        start_abort_monitor(a.abort_file)
    rng = random.Random(a.seed if a.seed is not None else int(time.time() * 1000) % (2 ** 31))
    key = open(os.path.expanduser(a.key_file)).read().strip() if a.key_file else None
    srv = Server(a.url, a.model, a.temperature, json.loads(a.extra), api_key=key)
    corp = Corpus(a.corpus)
    out = Out(a.out)
    if a.cpt:
        corp.cpt = a.cpt
    else:
        calibrate(srv, corp, rng)
    if a.mode == "calib":
        return
    {"ttft": mode_ttft, "decode": mode_decode, "conc": mode_conc, "hol": mode_hol}[a.mode](a, srv, corp, rng, out)


if __name__ == "__main__":
    main()
