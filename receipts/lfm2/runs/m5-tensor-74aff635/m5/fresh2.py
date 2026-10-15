"""Send saved two-turn messages cold: m2 (turn two) and m1 (turn one) as given, record the texts."""
import json, sys, urllib.request
url, model, src, out, which = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5].split(",")
s = json.load(open(src)); m2 = s["m2"]; m1 = m2[:2]
def chat(msgs, n):
    body = json.dumps({"model": model, "messages": msgs, "max_tokens": n, "temperature": 0}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", body, {"content-type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=3600)); msg = d["choices"][0]["message"]
    return dict(text=(msg.get("reasoning_content") or msg.get("reasoning") or "") + (msg.get("content") or ""), usage=d.get("usage"))
res = {}
for w in which: res[w] = chat(m2, 128) if w.startswith("m2") else chat(m1, 48)
json.dump(res, open(out, "w"), indent=1)
