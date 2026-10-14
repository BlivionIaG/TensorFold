"""Two-turn chat on an 8k history: turn one cold, turn two resumes the history; TTFTs, cached tokens and replies."""
import json, sys, time, urllib.request
url, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
doc = open("/Users/kletorch/Projects/Infra/tf-lfm2/CHANGELOG.md").read()[:30000]
def chat(msgs, n):
    body = json.dumps({"model": model, "messages": msgs, "max_tokens": n, "temperature": 0, "stream": True}).encode()
    t = time.time(); first = None; text = []
    usage = {}
    req = urllib.request.Request(url + "/v1/chat/completions", body, {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            for ch in d.get("choices", []):
                delta = ch.get("delta", {})
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or delta.get("reasoning") or "")
                if piece:
                    if first is None: first = time.time() - t
                    text.append(piece)
    return dict(ttft=first, total=time.time() - t, text="".join(text), usage=usage)
m1 = [{"role": "system", "content": "You are a release-notes assistant."}, {"role": "user", "content": "Here is a changelog:\n" + doc + "\n\nSummarize the latest release in one sentence."}]
r1 = chat(m1, 48)
m2 = m1 + [{"role": "assistant", "content": r1["text"].split("</think>")[-1].strip()}, {"role": "user", "content": "And the one before it?"}]
r2 = chat(m2, 128)
r3 = chat(m2, 128)
res = dict(turn1=r1, turn2=r2, turn2_again=r3, m2=m2)
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: {a: (b if a != "text" else b[:60]) for a, b in v.items()} for k, v in res.items() if k != "m2"}))
