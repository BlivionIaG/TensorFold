"""plain.py BASE IDS_FILE TOKENS OUT: one greedy request (seed 1234) with "draft": false over a raw token array."""
import hashlib, json, os, sys
sys.path.insert(0, os.path.expanduser("$TREE/tools"))
import bench_concurrent as bench
base, pf, tokens, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
raw = open(pf).read(); ids = json.loads(raw)
r = bench.stream(base, "m", {"name": "fixture", "kind": "completion", "prompt": ids}, tokens, 0, 1234, False)
keep = ("prompt_tokens", "cached_tokens", "tokens", "cache_state", "complete", "token_sha", "token_sha256",
        "output_sha256", "ttft_s", "decode_tps", "error")
rep = {"draft": False, "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(), "requested_reply_tokens": tokens,
       "temperature": 0, "seed": 1234, "row": {k: r.get(k) for k in keep}}
json.dump(rep, open(out, "w"), indent=2)
print(json.dumps(rep["row"]))
