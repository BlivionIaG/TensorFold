#!/usr/bin/env python3
"""Performance report for any OpenAI-compatible streaming server (tensorfold-native serve, mlx_lm.server, ...).

    serve_bench.py --url http://127.0.0.1:8080 --model NAME --tokenizer MODEL_DIR [--concurrency 1,2,4,8] [--loads 1024:512,16384:1024]
                   [--reps 2] [--json out.json] [--label "..."]    (run with: uv run --with tokenizers serve_bench.py ...)

The reporting standard: prefill (tok/s), decode (tok/s), TTFT (s), ITL (ms), at concurrency c = 1, 2, 4, 8, per request and aggregate.

For every (prompt length, c) cell, c requests start together on /v1/chat/completions (stream, greedy, thinking off). Each prompt is a salt that
is unique to the request (and the rep) in front of a filler text and a closing instruction, so no prefix cache can serve it; the filler length is
calibrated once per prompt length against the server's own `usage.prompt_tokens` (the chat template included) to exactly the asked count (the filler is
cut at a token boundary of the model's tokenizer, the salt is eight digits). The output length is fixed: `max_tokens`, `min_tokens` and `ignore_eos` are
all sent (a server that ignores the last two simply stops where it stops: the counts the server reports are used, a request whose count differs from the
asked one is flagged `short` in the table and the JSON; a prompt that the server counts off the asked length is flagged `prompt_off`).

Per request: TTFT = first content chunk - request sent; prefill tok/s = prompt_tokens / TTFT; decode tok/s = (out_tokens - 1) / (last chunk - first
chunk); ITL samples = the gap before each later chunk divided by the tokens a chunk carries (one when the server streams token by token, i.e. its
completion_tokens / chunks is below 1.05, else that ratio: a server that streams several tokens a chunk gets its gaps spread over them).
Aggregate per cell and rep: output tokens / wall (first request sent to last chunk of any request, which includes every prefill) and the steady
decode rate (sum of (out - 1) over the span between the earliest first chunk and the latest last chunk). Every figure is mean / p50 / p90 over
the requests of all reps of the cell; the aggregate columns over the reps. One warm-up round of every c (output cut to 32 tokens) is run first
and not counted. Standard library plus `tokenizers`.
"""
import argparse, http.client, json, math, os, random, statistics, sys, threading, time, urllib.parse, uuid

FILLER = ("The harbour town of Merrowick kept its ledgers in a stone tower, and every autumn the clerks climbed the stairs to count the barrels of "
          "salted fish, the bolts of grey wool and the crates of iron nails that the carts had brought down from the hills. ")
CLOSING = " Now write a very long story about a dragon who learns to code. Do not stop early; keep going with new scenes."

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--url", default="http://127.0.0.1:8080")
ap.add_argument("--model", default="default", help="the `model` field of the request")
ap.add_argument("--concurrency", default="1,2,4,8")
ap.add_argument("--loads", default="1024:512,16384:1024", help="INPUT:OUTPUT token pairs, comma separated (the user's two loads: 1k in 512 out, 16k in 1k out)")
ap.add_argument("--tokenizer", required=True, help="tokenizer.json of the model (or its directory); the script needs the `tokenizers` package: uv run --with tokenizers")
ap.add_argument("--reps", type=int, default=2, help="rounds of c requests a cell; statistics pool the requests of all rounds")
ap.add_argument("--timeout", type=float, default=3600)
ap.add_argument("--json", help="write the raw per-request records and the summary here")
ap.add_argument("--label", default="", help="a line printed above the tables (server, model, hardware, clocks)")
ap.add_argument("--api-key", default="")
args = ap.parse_args()
from tokenizers import Tokenizer  # noqa: E402 (uv run --with tokenizers)
TOK = Tokenizer.from_file(os.path.join(args.tokenizer, "tokenizer.json") if os.path.isdir(args.tokenizer) else args.tokenizer)

u = urllib.parse.urlparse(args.url)
HOST, PORT = u.hostname, u.port or 80
BASE = u.path.rstrip("/")


def body(prompt, max_tokens, stream=True):
    b = {"model": args.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "min_tokens": max_tokens,
         "ignore_eos": True, "temperature": 0, "top_p": 1, "stream": stream, "chat_template_kwargs": {"enable_thinking": False}}
    if stream:
        b["stream_options"] = {"include_usage": True}
    return b


def conn():
    return http.client.HTTPConnection(HOST, PORT, timeout=args.timeout)


def headers():
    h = {"content-type": "application/json"}
    if args.api_key:
        h["authorization"] = "Bearer " + args.api_key
    return h


def salt():
    return "".join(random.choice("0123456789") for _ in range(8))  # eight digits: eight tokens in the Qwen tokenizers, whatever they are


_ids = None



def filler_ids():
    global _ids
    if _ids is None:
        _ids = TOK.encode(FILLER * 2000, add_special_tokens=False).ids
    return _ids


def prompt_of(m, s):
    """A salt, then the filler's first m tokens decoded, then the closing instruction."""
    return f"Case {s}. " + TOK.decode(filler_ids()[:m]) + CLOSING


def count_prompt(m):
    c = conn()
    c.request("POST", BASE + "/v1/chat/completions", json.dumps(body(prompt_of(m, salt()), 1, stream=False)), headers())
    r = json.loads(c.getresponse().read())
    c.close()
    return r["usage"]["prompt_tokens"]


def calibrate(target):
    """The filler length m whose prompt the server counts as exactly `target` tokens (salted prompts: three different salts must agree)."""
    m = target
    for _ in range(12):
        ts = {count_prompt(m) for _ in range(3)}
        if ts == {target}:
            return m
        t = max(ts) if target < min(ts) else min(ts)
        m = max(0, m + target - t)
    sys.exit(f"cannot make a prompt of exactly {target} tokens (last filler {m} counted {sorted(ts)})")


def one(prompt, max_tokens, out, i, barrier):
    rec = {"ok": False}
    out[i] = rec
    try:
        c = conn()
        data = json.dumps(body(prompt, max_tokens))
        barrier.wait()
        t0 = time.perf_counter()
        for attempt in range(8):  # a server with a small listen backlog (mlx_lm.server's) resets some of c simultaneous connections: retry, TTFT counts from the first send
            try:
                c.request("POST", BASE + "/v1/chat/completions", data, headers())
                resp = c.getresponse()
                break
            except (BrokenPipeError, ConnectionResetError, http.client.RemoteDisconnected):
                rec["retries"] = attempt + 1
                c.close()
                time.sleep(0.05)
                c = conn()
        if resp.status != 200:
            rec["error"] = f"HTTP {resp.status}: {resp.read()[:200]!r}"
            return
        chunks = []  # (time, characters)
        usage = None
        while True:
            line = resp.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            ev = json.loads(payload)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                text = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                if text:
                    chunks.append((time.perf_counter(), len(text)))
        t_end = time.perf_counter()
        c.close()
        if not chunks or not usage:
            rec["error"] = "no content chunks" if not chunks else "no usage in the stream"
            return
        rec.update(ok=True, t0=t0, t_first=chunks[0][0], t_last=chunks[-1][0], t_end=t_end, chunks=len(chunks),
                   prompt_tokens=usage["prompt_tokens"], out_tokens=usage["completion_tokens"])
        per = rec["out_tokens"] / len(chunks)  # tokens a chunk: one a chunk when the server streams token by token (within 5%), else the mean
        per = 1.0 if per < 1.05 else per
        rec["itl_ms"] = [(chunks[k][0] - chunks[k - 1][0]) * 1000.0 / per for k in range(1, len(chunks))]
        rec["chunk_t_ms"] = [round((t - t0) * 1000.0, 2) for t, _ in chunks]
    except Exception as e:  # a failed request is reported, not hidden
        rec["error"] = repr(e)


def run_round(prompts, max_tokens):
    out = [None] * len(prompts)
    barrier = threading.Barrier(len(prompts))
    ts = [threading.Thread(target=one, args=(p, max_tokens, out, i, barrier)) for i, p in enumerate(prompts)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    return out


def pct(v, q):
    v = sorted(v)
    if not v:
        return float("nan")
    k = (len(v) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def stats(v):
    return (statistics.fmean(v), pct(v, 0.5), pct(v, 0.9)) if v else (float("nan"),) * 3


def f(x, d=1):
    return "-" if x != x else f"{x:.{d}f}"


cs = [int(x) for x in args.concurrency.split(",")]
loads = [tuple(int(y) for y in x.split(":")) for x in args.loads.split(",")]
results = {"label": args.label, "url": args.url, "model": args.model, "loads": args.loads, "reps": args.reps, "cells": []}
ms = {}
for L in sorted({l for l, _ in loads} | {256}):
    ms[L] = calibrate(L)
    print(f"# prompt {L}: {ms[L]} filler tokens -> exactly {L} tokens at the server (chat template included)", file=sys.stderr, flush=True)
for L, O in loads:
    r = run_round([prompt_of(ms[L], salt())], 16)
    if not r[0]["ok"]:
        sys.exit(f"warm-up failed: {r[0]['error']}")

for L, O in loads:
    for c in cs:  # warm-up: every width once on a short prompt, and the full prompt once alone; short output, counted nowhere
        r = run_round([prompt_of(ms[256], salt()) for _ in range(c)], 32)
        bad = [x["error"] for x in r if not x["ok"]]
        if bad:
            sys.exit(f"warm-up failed: {bad[0]}")
    for c in cs:
        recs, aggs = [], []
        for rep in range(args.reps):
            r = run_round([prompt_of(ms[L], salt()) for _ in range(c)], O)
            good = [x for x in r if x["ok"]]
            recs += r
            if good:
                wall = max(x["t_end"] for x in good) - min(x["t0"] for x in good)
                span = max(x["t_last"] for x in good) - min(x["t_first"] for x in good)
                aggs.append({"wall": wall, "tok_s": sum(x["out_tokens"] for x in good) / wall,
                             "decode_tok_s": sum(x["out_tokens"] - 1 for x in good) / span if span > 0 else float("nan")})
            print(f"# L={L} c={c} rep {rep}: {len(good)}/{c} ok", file=sys.stderr, flush=True)
        good = [x for x in recs if x["ok"]]
        short = sum(1 for x in good if x["out_tokens"] != O)
        cell = {"prompt_tokens_asked": L, "out_asked": O, "c": c, "requests": recs, "aggregate_rounds": aggs, "failed": len(recs) - len(good), "short": short,
                "prompt_off": sum(1 for x in good if x["prompt_tokens"] != L)}
        cell["prompt_tokens"] = statistics.fmean([x["prompt_tokens"] for x in good]) if good else float("nan")
        cell["out_tokens"] = statistics.fmean([x["out_tokens"] for x in good]) if good else float("nan")
        ttft = [x["t_first"] - x["t0"] for x in good]
        cell["ttft_s"] = stats(ttft)
        cell["prefill_tok_s"] = stats([x["prompt_tokens"] / (x["t_first"] - x["t0"]) for x in good])
        cell["decode_tok_s"] = stats([(x["out_tokens"] - 1) / (x["t_last"] - x["t_first"]) for x in good if x["t_last"] > x["t_first"]])
        cell["itl_ms"] = stats([s for x in good for s in x["itl_ms"]])
        cell["agg_tok_s"] = statistics.fmean([a["tok_s"] for a in aggs]) if aggs else float("nan")
        cell["agg_decode_tok_s"] = statistics.fmean([a["decode_tok_s"] for a in aggs]) if aggs else float("nan")
        # an aggregate prefill rate: all prompts' tokens over the time until the last first token
        cell["agg_prefill_tok_s"] = statistics.fmean([
            sum(x["prompt_tokens"] for x in recs[i * c:(i + 1) * c] if x["ok"]) / (max(x["t_first"] for x in recs[i * c:(i + 1) * c] if x["ok"]) - min(x["t0"] for x in recs[i * c:(i + 1) * c] if x["ok"]))
            for i in range(args.reps) if any(x["ok"] for x in recs[i * c:(i + 1) * c])]) if good else float("nan")
        results["cells"].append(cell)

lines = []
if args.label:
    lines += [args.label, ""]
lines.append(f"greedy, {args.reps} rounds a cell, per-request figures are mean / p50 / p90 over the requests of all rounds.")
for L, O in loads:
    lines += ["", f"### {L} input / {O} output tokens", "",
              "| c | prompt tok | prefill tok/s agg | prefill tok/s per request | decode tok/s agg | decode tok/s per request | total tok/s agg | TTFT s | ITL ms | notes |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for cell in results["cells"]:
        if cell["prompt_tokens_asked"] != L:
            continue
        t = lambda k, d=1: " / ".join(f(x, d) for x in cell[k])
        notes = []
        if cell["failed"]:
            notes.append(f"{cell['failed']} failed")
        if cell["short"]:
            notes.append(f"{cell['short']} not {cell['out_asked']} tokens (mean {cell['out_tokens']:.0f})")
        if cell["prompt_off"]:
            notes.append(f"{cell['prompt_off']} prompts not {cell['prompt_tokens_asked']} tokens")
        lines.append(f"| {cell['c']} | {cell['prompt_tokens']:.0f} | {f(cell['agg_prefill_tok_s'])} | {t('prefill_tok_s')} | {f(cell['agg_decode_tok_s'])} | {t('decode_tok_s')} | "
                     f"{f(cell['agg_tok_s'])} | {t('ttft_s', 2)} | {t('itl_ms')} | {', '.join(notes)} |")
print("\n".join(lines))
if args.json:
    json.dump(results, open(args.json, "w"), indent=1)
