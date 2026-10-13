# summarize.py DIR CONFIG...: markdown tables from the receipt JSON (bench_cache, bench_openai, bench_concurrent, memwatch)
import json, os, statistics, sys
root, cfgs = sys.argv[1], sys.argv[2:]
N = [2048, 8192, 32768, 65536, 131072, 261888]
def j(c, f):
    p = os.path.join(root, c, f)
    return json.load(open(p)) if os.path.exists(p) else None
print("Cold prefill, time to first token (median of 3 cold runs, 0 cached tokens each) and prefill tok/s:\n")
print("| Prompt tokens | " + " | ".join(cfgs) + " |"); print("| --- |" + " --- |" * len(cfgs))
for n in N:
    cells = []
    for c in cfgs:
        d = j(c, f"prefill_{n}.json")
        if not d: cells.append("not run"); continue
        ok = d["qualified"] and all(r["cached_tokens"] == 0 and r["prompt_tokens"] == n for r in d["rows"])
        t = statistics.median(r["ttft_s"] for r in d["rows"])
        cells.append(f"{t:.2f} s, {n / t:,.0f} tok/s" + ("" if ok else " (not qualified)"))
    print(f"| {n:,} | " + " | ".join(cells) + " |")
print("\nDecode (`bench_openai.py --tokens 256 --reps 3 --temperatures 0`), median tok/s and time to first token:\n")
print("| Prompt | " + " | ".join(cfgs) + " |"); print("| --- |" + " --- |" * len(cfgs))
for p in ("fibonacci-raw", "gpu-chat-no-think"):
    cells = []
    for c in cfgs:
        d = j(c, "decode.json"); r = [x for x in (d or []) if x["prompt"] == p]
        cells.append(f"{r[0]['decode_tps_median']:.1f} tok/s, {r[0]['ttft_s_median'] * 1000:.0f} ms" if r else "not run")
    print(f"| {p} | " + " | ".join(cells) + " |")
print("\nConcurrency (`bench_concurrent.py --tokens 256 --reps 3 --temperatures 0 --levels 1,4 --alone --serial`), total tok/s; equal to solo / to `\"draft\": false`:\n")
print("| Prompt, streams | " + " | ".join(cfgs) + " |"); print("| --- |" + " --- |" * len(cfgs))
for p in ("code", "chat"):
    for s in (1, 4):
        cells = []
        for c in cfgs:
            d = j(c, "concurrent.json"); r = [x for x in ((d or {}).get("cells") or []) if x["prompt"] == p and x["streams"] == s]
            if not r: cells.append("not run"); continue
            x = r[0]; a, se = x["alone"], x["serial"]
            cells.append(f"{x['aggregate_tps']:.1f} tok/s; {a['equal']}/{a['equal'] + a['unequal'] + a['failed']}, {se['equal']}/{se['equal'] + se['unequal'] + se['failed']}")
        print(f"| {p}, {s} | " + " | ".join(cells) + " |")
print("\nMemory of the server processes only (every rank; sampled each second plus the kernel's high-water mark):\n")
print("| | " + " | ".join(cfgs) + " |"); print("| --- |" + " --- |" * len(cfgs))
for sess, label in (("prefill", "native window, 1 stream"), ("decode", "32K window, 4 streams")):
    v = []; r = []; h = []
    for c in cfgs:
        d = j(c, f"{sess}.mem.json")
        v.append(f"{d['peak_vram_bytes'] / 2**30:.1f} GiB" if d else "-"); r.append(f"{d['peak_rss_kib'] / 2**20:.2f} GiB" if d else "-"); h.append(f"{d['peak_hwm_kib_sum'] / 2**20:.1f} GiB" if d else "-")
    print(f"| Device memory, peak ({label}) | " + " | ".join(v) + " |")
    print(f"| Host RSS while serving, peak ({label}) | " + " | ".join(r) + " |")
    print(f"| Host RSS high-water mark incl. load ({label}) | " + " | ".join(h) + " |")
