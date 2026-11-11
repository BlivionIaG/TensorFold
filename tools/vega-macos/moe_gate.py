#!/usr/bin/env python3
"""Greedy gate of a running `tensorfold-native serve` (Qwen3.6-35B-A3B) against tools/vega-macos/moe_ref.py's mlx_lm references.

    tools/vega-macos/moe_gate.py --log SERVER.LOG [--port 18099] [--ref ~/.cache/tensorfold-vega/ref-moe.json] [--tag NAME] [--out DIR]

One chat request a prompt (temperature 0, thinking off, max_tokens = the reference's length); the server log's `done <id> ... sha=` is the sha of the
generated token ids, which must be the reference's. Prints per prompt the first diverging character when it differs. Also writes the
replies' text to DIR/TAG.json so two runs (cache on / TF_EXPERT_COPIES=0) can be compared with --against OTHER.json (texts and shas equal).
Exit status 0 when every sha matches (or, with --against, when the two runs are identical).
"""
import argparse, json, os, re, sys, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--log", required=True)
ap.add_argument("--port", type=int, default=18099)
ap.add_argument("--ref", default=os.path.expanduser("~/.cache/tensorfold-vega/ref-moe.json"))
ap.add_argument("--tag", default="run")
ap.add_argument("--out", default=os.path.expanduser("~/.cache/tensorfold-vega/bench"))
ap.add_argument("--against", help="a previous run's JSON: compare with it instead of the mlx references")
ap.add_argument("--rounds", type=int, default=1, help="repeat the prompt set (the cache warms across repeats; every repeat must match)")
a = ap.parse_args()

ref = json.load(open(a.ref))


def chat(prompt, n):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": n, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))


def sha_of(reply_id):
    for _ in range(20):
        for line in open(a.log).read().splitlines():
            if "done " + reply_id in line:
                m = re.search(r"sha=([0-9a-f]{12})", line)
                if m:
                    return m.group(1)
        time.sleep(0.25)
    return None


def diverge(x, y):
    for i, (p, q) in enumerate(zip(x, y)):
        if p != q:
            return i
    return min(len(x), len(y))


runs = []
ok = True
other = json.load(open(a.against)) if a.against else None
for rnd in range(a.rounds):
    for i, r in enumerate(ref):
        t0 = time.time()
        reply = chat(r["question"], len(r["tokens"]))
        text = reply["choices"][0]["message"].get("content") or ""
        sha = sha_of(reply["id"])
        want = other[len(runs)]["sha"] if other else r["sha"]
        want_text = other[len(runs)]["text"] if other else None
        same = sha == want and (want_text is None or want_text == text)
        ok &= same
        extra = "" if same else f"  DIVERGES at char {diverge(text, want_text if want_text is not None else r['text'])}"
        print(f"round {rnd} prompt {i}: {reply['usage']['completion_tokens']} tokens {time.time() - t0:.1f} s sha {sha} want {want} {'ok' if same else 'FAIL'}{extra}", flush=True)
        runs.append({"question": r["question"], "sha": sha, "text": text})
os.makedirs(a.out, exist_ok=True)
json.dump(runs, open(os.path.join(a.out, a.tag + ".json"), "w"))
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
