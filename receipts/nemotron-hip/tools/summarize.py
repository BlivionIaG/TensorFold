"""summarize.py OUT: tables of every hold under OUT/<topic>/<rig>/ (markdown on stdout, tables.json beside)."""
import glob, json, os, statistics, sys
O = sys.argv[1]
PR = {"pr37": "#37", "pr38": "#38", "pr44": "#44", "pr46": "#46"}
RIGS = ["v620-tp1", "v620-tp2", "w7800-tp2", "v620-tp4"]
RIGN = {"v620-tp1": "1 × V620, tp 1", "v620-tp2": "2 × V620, tp 2", "w7800-tp2": "2 × W7800, tp 2", "v620-tp4": "4 × V620, tp 4"}
def holds(topic, rig, kind):
    out = []
    for d in sorted(glob.glob(f"{O}/{topic}/{rig}/*_{kind}")):
        if not os.path.exists(f"{d}/.done"): continue
        h, tree = os.path.basename(d).split("_")[:2]
        out.append((h, tree, d))
    return sorted(out, key=lambda x: int(x[0][1:]))
def cold(d, n):
    c = json.load(open(f"{d}/cold_{n}.json")); p = json.load(open(f"{d}/plain_{n}.json"))
    rows = c["rows"]
    return {"ttft": [r["ttft_s"] for r in rows], "cached": [r["cached_tokens"] for r in rows] + [p["row"]["cached_tokens"]],
            "prompt_tokens": [r["prompt_tokens"] for r in rows] + [p["row"]["prompt_tokens"]],
            "sha": {r["token_sha256"] for r in rows}, "plain_sha": p["row"]["token_sha256"], "qualified": c["qualified"],
            "ids_sha256": c["ids_sha256"], "fixture_ok": c["fixture_sha256"] == c["ids_sha256"]}
tables = {}
md = []
def prefill_table(topic, rig, sizes, trees, title):
    hs = [h for h in holds(topic, rig, "prefill") if h[1] in trees] + ([h for h in holds(topic, rig, "long") if h[1] in trees] if topic == "family" else [])
    if not hs: return None
    md.append(f"\n#### {title}: {RIGN[rig]}\n")
    md.append("Hold order: " + ", ".join(f"{h} {PR[t]}" for h, t, _ in hs) + ".\n")
    md.append("| prompt tokens | " + " | ".join(f"{PR[t]} median `ttft_s`" for t in trees) + (" | branch / base |" if len(trees) == 2 else " |") + " every request, in hold order | cached tokens | 16-token reply `token_sha256` | == `\"draft\": false` |")
    md.append("|" + "---|" * (len(trees) + (1 if len(trees) == 2 else 0) + 4))
    res = {}
    for n in sizes:
        per = {t: [] for t in trees}; allreq = []; cached = []; shas = set(); eq = True; ok = True
        for h, t, d in hs:
            if not os.path.exists(f"{d}/cold_{n}.json"): continue
            c = cold(d, n); per[t] += c["ttft"]; allreq.append(f"{h}: " + ", ".join(f"{x:.3f}" for x in c["ttft"]))
            cached += c["cached"]; shas |= c["sha"] | {c["plain_sha"]}; eq &= c["sha"] == {c["plain_sha"]}
            ok &= c["qualified"] and c["fixture_ok"] and all(p == n for p in c["prompt_tokens"])
        if not any(per.values()): continue
        med = {t: statistics.median(v) if v else None for t, v in per.items()}
        cells = [f"{med[t]:.3f} ({len(per[t])})" if med[t] else "–" for t in trees]
        if len(trees) == 2: cells.append(f"{med[trees[1]] / med[trees[0]]:.3f}" if all(med.values()) else "–")
        md.append(f"| {n:,} | " + " | ".join(cells) + f" | {'; '.join(allreq)} | {'all 0' if all(x == 0 for x in cached) else cached} ({len(cached)}) | "
                  + (", ".join(sorted(s[:16] for s in shas)) if shas else "–") + f" | {'equal' if eq else 'NOT EQUAL'}{'' if ok else ' (QUALIFY FAIL)'} |")
        res[n] = {"median": med, "all": per, "cached_all_zero": all(x == 0 for x in cached), "requests": len(cached), "sha": sorted(shas), "draft_false_equal": eq, "qualified": ok}
    return res
def decode_table(topic, rig, kinds=("decode", "decodeoff")):
    res = {}
    for kind in kinds:
        hs = holds(topic, rig, kind)
        if not hs: continue
        md.append(f"\n#### Decode, drafts {'on' if kind == 'decode' else 'off'}: {RIGN[rig]}\n")
        md.append("`bench_openai.py --tokens 256 --reps 3 --temperatures 0`, medians in tok/s (the 3 reps in brackets), holds in order.\n")
        md.append("| prompt | " + " | ".join(f"{h} {PR[t]}" for h, t, _ in hs) + " |")
        md.append("|---|" + "---:|" * len(hs))
        for prompt in ("fibonacci-raw", "gpu-chat-no-think"):
            cells = []
            for h, t, d in hs:
                r = [x for x in json.load(open(f"{d}/decode.json"))["results"] if x["prompt"] == prompt][0]
                cells.append(f"{r['decode_tps_median']:.2f} ({', '.join(f'{x:.2f}' for x in r['decode_tps_all'])})")
                res.setdefault(kind, {}).setdefault(prompt, []).append((h, t, r["decode_tps_median"], [e["token_sha"] for e in r["evidence"]]))
            md.append(f"| {prompt} | " + " | ".join(cells) + " |")
        for prompt in ("fibonacci-raw", "gpu-chat-no-think"):
            rows = res[kind][prompt]; trees = sorted({t for _, t, _, _ in rows})
            means = {t: statistics.mean(m for _, tt, m, _ in rows if tt == t) for t in trees}
            sh = {t: {s for _, tt, _, ss in rows if tt == t for s in ss} for t in trees}
            md.append(f"\n{prompt}: mean of medians " + ", ".join(f"{PR[t]} {means[t]:.2f}" for t in trees)
                      + ("" if "pr46" not in means or "pr38" not in means else f"; #46 / #38 {100 * (means['pr46'] / means['pr38'] - 1):+.1f} %")
                      + ("" if "pr44" not in means or "pr38" not in means else f"; #44 / #38 {100 * (means['pr44'] / means['pr38'] - 1):+.1f} %")
                      + ". `token_sha` per tree: " + "; ".join(f"{PR[t]} {', '.join(sorted(sh[t]))}" for t in trees) + ".")
    return res
def conc_table(rig):
    res = {}
    for kind in ("conc", "conc8"):
        for h, t, d in holds("family", rig, kind):
            rows = json.load(open(f"{d}/concurrent.json"))["cells"]
            md.append(f"\n#### `bench_concurrent.py --tokens 256 --reps 3 --temperatures 0 --levels {'1,4' if kind == 'conc' else '1,4,8'} --alone --serial`: {RIGN[rig]}, {PR[t]}\n")
            md.append("| prompt | streams | aggregate tok/s | per-stream tok/s | concurrent == alone (equal/unequal/failed) | alone == `\"draft\": false` |")
            md.append("|---|---:|---:|---:|---|---|")
            for r in rows:
                a = r["alone"]; s = r["serial"]
                md.append(f"| {r['prompt']} | {r['streams']} | {r['aggregate_tps']:.1f} | {r.get('per_stream_tps') or 0:.1f} | {a['equal']}/{a['unequal']}/{a['failed']} | {s['equal']}/{s['unequal']}/{s['failed']} |")
                res.setdefault(kind, []).append({k: r[k] for k in ("prompt", "streams", "aggregate_tps", "alone", "serial")})
    return res
topics = sys.argv[2:] or ["fusions", "prefill-chunk", "family"]
for topic in topics:
    md.append(f"\n## {topic}\n")
    for rig in RIGS:
        if not os.path.isdir(f"{O}/{topic}/{rig}"): continue
        md.append(f"\n### {RIGN[rig]}\n")
        t = tables.setdefault(topic, {}).setdefault(rig, {})
        if topic == "fusions":
            t["decode"] = decode_table(topic, rig)
            t["prefill"] = prefill_table(topic, rig, [2048, 8192, 32768, 65536], ["pr38", "pr46"], "Cold prefill, #38 (base) against #46 (branch)")
        elif topic == "prefill-chunk":
            t["prefill"] = prefill_table(topic, rig, [2048, 8192, 32768, 65536, 131072, 262080], ["pr37", "pr38"], "Cold prefill, #37 (base) against #38 (branch)")
        else:
            t["decode"] = decode_table(topic, rig, ("decode",)); t["conc"] = conc_table(rig)
            t["long"] = prefill_table(topic, rig, [131072, 262080], ["pr46"], "Cold prefill at #46, 131,072 and 262,080 tokens")
print("\n".join(md))
json.dump(tables, open(f"{O}/tables.json", "w"), indent=1, default=list)
