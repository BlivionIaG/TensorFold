import json, os, sys, time, urllib.request, uuid
sys.path.insert(0, os.path.expanduser("~"))
import exl3_text_abba as t

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
TOK = int(os.environ.get("TOK", "192"))
root = "~/tensorfold-twochunk-dbg"
a = t.corpus([root + "/zig/src/families/flashnext/*.zig", root + "/zig/src/core/**/*.zig"], 50000)
b = t.corpus([root + "/docs/**/*.md", root + "/*.md", root + "/zig/tests/**/*.zig"], 50000)


def one(prompt):
    body = {
        "model": os.environ.get("MODEL", "qwen3.8-flash-next-iq3_xxs"),
        "messages": [{"role": "user", "content": "Run %s.\n%s" % (uuid.uuid4().hex, prompt)}],
        "max_tokens": TOK, "temperature": 0, "ignore_eos": True, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT, json.dumps(body).encode(), {"Content-Type": "application/json"})
    start = time.perf_counter(); first = last = None; usage = {}; pieces = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if line == "data: [DONE]":
                break
            if not line.startswith("data:"):
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            for c in ev.get("choices", []):
                d = c.get("delta") or {}
                if d.get("content") or d.get("reasoning_content"):
                    now = time.perf_counter(); first = first or now; last = now; pieces += 1
    n = int(usage.get("completion_tokens", 0))
    span = (last - first) if first and last else 0
    return {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": n,
            "ttft_s": round(first - start, 2) if first else None,
            "decode_tps": round((n - 1) / span, 2) if n > 1 and span > 0 else None}


short = "Write a long, detailed story about a lighthouse keeper who finds a message in a bottle. Keep writing until told to stop."
for name, prompt in (("short", short), ("zig", a), ("docs", b)):
    for rep in range(3):
        r = one(prompt); r["prompt"] = name; r["rep"] = rep
        print(json.dumps(r), flush=True)
