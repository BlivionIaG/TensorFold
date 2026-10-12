"""Streamed serving speed; usage: bench.py URL MODEL [PROMPT_TOKENS] [REPLY_TOKENS] [CONCURRENCY] [DRAFT]."""

import json
import statistics
import sys
import threading
import time
import urllib.request

URL, MODEL = sys.argv[1], sys.argv[2]
PROMPT = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
REPLY = int(sys.argv[4]) if len(sys.argv) > 4 else 256
CONC = int(sys.argv[5]) if len(sys.argv) > 5 else 1
DRAFT = (sys.argv[6] if len(sys.argv) > 6 else "on") == "on"


def text(n, seed):
    """About n tokens of distinct words (the tokenizer takes most as one or two tokens)."""
    return " ".join(f"item{(i * 7919 + seed * 104729) % 100003}" for i in range(n // 3)) + "\nSummarize the list above in one long paragraph."


def run(seed, out):
    body = {"model": MODEL, "messages": [{"role": "user", "content": text(PROMPT, seed)}], "max_tokens": REPLY, "temperature": 0.0,
            "draft": DRAFT, "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    stamps, usage = [], {}
    with urllib.request.urlopen(request, timeout=3600) as reply:
        for line in reply:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            doc = json.loads(line[6:])
            if doc.get("usage"):
                usage = doc["usage"]
            for c in doc.get("choices", []):
                if c.get("delta", {}).get("content") or c.get("delta", {}).get("reasoning_content"):
                    stamps.append(time.perf_counter())
    out.append({"t0": t0, "stamps": stamps, "usage": usage})


results = []
threads = [threading.Thread(target=run, args=(i, results)) for i in range(CONC)]
start = time.perf_counter()
for t in threads:
    t.start()
for t in threads:
    t.join()
wall = time.perf_counter() - start
ttft = [r["stamps"][0] - r["t0"] for r in results if r["stamps"]]
itl = [b - a for r in results for a, b in zip(r["stamps"], r["stamps"][1:])]
prompt_tokens = sum(r["usage"].get("prompt_tokens", 0) for r in results)
reply_tokens = sum(r["usage"].get("completion_tokens", 0) for r in results)
decode = [(r["usage"].get("completion_tokens", 1) - 1) / (r["stamps"][-1] - r["stamps"][0]) for r in results if len(r["stamps"]) > 1]
print(json.dumps({
    "concurrency": CONC, "draft": DRAFT, "prompt_tokens_each": prompt_tokens // max(CONC, 1), "reply_tokens_each": reply_tokens // max(CONC, 1),
    "ttft_ms_mean": round(1000 * statistics.mean(ttft), 1), "prefill_tok_s": round(prompt_tokens / max(sum(ttft), 1e-9) * CONC / CONC, 1),
    "decode_tok_s_per_stream": round(statistics.mean(decode), 2), "decode_tok_s_total": round(sum(decode), 2),
    "itl_ms_mean": round(1000 * statistics.mean(itl), 2) if itl else None, "itl_ms_p50": round(1000 * statistics.median(itl), 2) if itl else None,
    "wall_s": round(wall, 2)}))
