"""Cold prefill on exact token arrays: one request each, max_tokens 1, timings and cached tokens."""
import json, sys, time, urllib.request
url, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
for spec in sys.argv[4:]:
    path, reps = spec.split(":")
    ids = json.load(open(path))
    for r in range(int(reps)):
        body = json.dumps({"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0}).encode()
        t = time.time()
        res = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/completions", body, {"content-type": "application/json"}), timeout=7200))
        dt = time.time() - t
        u = res.get("usage", {})
        row = dict(tokens=len(ids), seconds=dt, tok_s=len(ids) / dt, prompt_tokens=u.get("prompt_tokens"), cached=(u.get("prompt_tokens_details") or {}).get("cached_tokens"))
        rows.append(row); print(json.dumps(row), flush=True)
json.dump(rows, open(out, "w"), indent=1)
