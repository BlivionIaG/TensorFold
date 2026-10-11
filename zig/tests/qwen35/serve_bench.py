"""Prefill, decode tok/s, TTFT, ITL per request and total, c=1 and c=8; URL MODEL [--json F] [--no-draft]."""

import http.client
import json
import statistics
import sys
import threading
import time
import urllib.parse

URL, MODEL = sys.argv[1], sys.argv[2]
OUT = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
DRAFT = "--no-draft" not in sys.argv  # --no-draft: every request asks for "draft": false
TOKENS = 256  # reply tokens a request asks for
SHORT, MEDIUM, LONG = 256, 2048, 8192  # prompt tokens, about
VOCAB = ("the cat sat on a mat and saw the dog run in the park while we read old books by the sea "
         "under blue sky with red sun near green hill over cold water at night").split()


def prompt(words, nonce):
    """A prompt of about `words` tokens of common words, unique by `nonce` up front so no kept state serves it."""
    body = " ".join(VOCAB[(i * 7 + nonce) % len(VOCAB)] for i in range(words))
    return f"Request {nonce}. Ignore this text: {body}. Now count from one to one thousand, separated by commas."


def request(text, stamps):
    """One streamed chat completion; `stamps` gets the send time, each chunk's time and the usage."""
    url = urllib.parse.urlparse(URL)
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=1800)
    body = {"model": MODEL, "messages": [{"role": "user", "content": text}], "max_tokens": TOKENS, "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True, "draft": DRAFT,
            "chat_template_kwargs": {"enable_thinking": False}}
    stamps["sent"] = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    reply = conn.getresponse()
    stamps["chunks"] = []
    for raw in reply:
        line = raw.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"):
            continue
        doc = json.loads(line[5:])
        if doc.get("usage"):
            stamps["usage"] = doc["usage"]
            stamps["done"] = time.perf_counter()
        for choice in doc.get("choices", []):
            delta = choice.get("delta", {})
            if delta.get("content") or delta.get("reasoning_content"):
                stamps["chunks"].append(time.perf_counter())
    conn.close()


def measure(s):
    """One request's numbers from its stamps."""
    chunks, usage = s["chunks"], s.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    out = usage.get("completion_tokens", len(chunks))
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    ttft = (chunks[0] - s["sent"]) if chunks else float("nan")
    # the stream's end, not its last visible text: tokens past the end of a reply may print nothing
    end = s.get("done", chunks[-1] if chunks else s["sent"])
    span = end - chunks[0] if chunks and end > chunks[0] else float("nan")
    gaps = sorted(1000 * (b - a) for a, b in zip(chunks, chunks[1:])) or [float("nan")]
    return {
        "prompt": prompt_tokens, "out": out, "cached": cached, "ttft_ms": 1000 * ttft,
        "prefill_tps": (prompt_tokens - cached) / ttft if ttft > 0 else float("nan"),
        "decode_tps": (out - 1) / span if span > 0 else float("nan"),
        "itl_ms": 1000 * span / (out - 1) if out > 1 else float("nan"),
        "itl_p50_ms": gaps[len(gaps) // 2], "itl_p99_ms": gaps[min(len(gaps) - 1, int(0.99 * len(gaps)))],
        "itl_max_ms": gaps[-1], "sent": s["sent"], "first": chunks[0] if chunks else s["sent"],
        "last": end,
    }


def run(name, specs):
    """Requests (words, delay seconds) started together; each request's numbers and the totals."""
    stamps = [{} for _ in specs]
    threads = []
    t0 = time.perf_counter()
    for (words, delay), st, n in zip(specs, stamps, range(len(specs))):
        nonce = int(time.time() * 1000) % 100000 * 16 + n

        def go(words=words, delay=delay, st=st, nonce=nonce):
            time.sleep(delay)
            request(prompt(words, nonce), st)

        threads.append(threading.Thread(target=go))
        threads[-1].start()
    for t in threads:
        t.join()
    rows = [measure(s) for s in stamps]
    start = min(r["sent"] for r in rows)
    end = max(r["last"] for r in rows)
    prompt_end = max(r["first"] for r in rows)
    total = {
        "requests": len(rows), "wall_s": end - start,
        "out_tps": sum(r["out"] for r in rows) / (end - start),
        "prompt_tps": sum(r["prompt"] - r["cached"] for r in rows) / (prompt_end - start),
        "ttft_ms_mean": statistics.mean(r["ttft_ms"] for r in rows), "ttft_ms_max": max(r["ttft_ms"] for r in rows),
        "itl_ms_mean": statistics.mean(r["itl_ms"] for r in rows), "itl_ms_max": max(r["itl_max_ms"] for r in rows),
        "decode_tps_mean": statistics.mean(r["decode_tps"] for r in rows),
    }
    print(f"\n### {name} ({time.perf_counter() - t0:.1f} s)\n")
    print("| req | prompt | out | cached | TTFT ms | prefill tok/s | decode tok/s | ITL ms mean | p50 | p99 | max |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for i, r in enumerate(rows):
        print(f"| {i} | {r['prompt']} | {r['out']} | {r['cached']} | {r['ttft_ms']:.0f} | {r['prefill_tps']:.0f} | "
              f"{r['decode_tps']:.1f} | {r['itl_ms']:.1f} | {r['itl_p50_ms']:.1f} | {r['itl_p99_ms']:.1f} | {r['itl_max_ms']:.1f} |")
    print(f"\ntotal: {total['requests']} requests in {total['wall_s']:.2f} s; output {total['out_tps']:.1f} tok/s, "
          f"prompt {total['prompt_tps']:.0f} tok/s; TTFT mean {total['ttft_ms_mean']:.0f} ms, max {total['ttft_ms_max']:.0f} ms; "
          f"ITL mean {total['itl_ms_mean']:.1f} ms, worst gap {total['itl_ms_max']:.0f} ms; decode {total['decode_tps_mean']:.1f} tok/s a request")
    return {"name": name, "rows": rows, "total": total}


results = [run("warm-up", [(64, 0.0)])]
for words in (SHORT, MEDIUM, LONG):
    results.append(run(f"c=1, prompt {words}", [(words, 0.0)]))
results.append(run("c=8 uniform, prompt 1024, all at once", [(1024, 0.0)] * 8))
# short and long prompts alternate and arrive 0.4 s apart: long prompts land while the short ones decode
results.append(run("c=8 mixed, prompts 256 / 8192 alternating, 0.4 s apart", [(SHORT if i % 2 == 0 else LONG, 0.4 * i) for i in range(8)]))
if OUT:
    with open(OUT, "w") as f:
        json.dump(results[1:], f, indent=1)
