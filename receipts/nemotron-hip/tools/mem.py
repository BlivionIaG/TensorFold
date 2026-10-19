"""mem.py HOLD_DIR...: peak VmHWM / VmRSS per rank and VRAM per GPU (steady and peak) from memory_samples.jsonl and hold.json."""
import json, sys, glob, re
G = 1 << 30
for d in sys.argv[1:]:
    s = [json.loads(l) for l in open(f"{d}/memory_samples.jsonl")]
    h = json.load(open(f"{d}/hold.json"))
    ranks = {}
    for x in s:
        for i, p in enumerate(sorted(x["procs"], key=lambda p: p["pgid"])):
            r = ranks.setdefault(p["pgid"], {"vmhwm": 0, "vmrss": 0}); r["vmhwm"] = max(r["vmhwm"], p["vmhwm_kib"]); r["vmrss"] = max(r["vmrss"], p["vmrss_kib"])
    vr = {}
    for x in s:
        for c, v in (x["rocm_smi_vram"] or {}).items():
            vr.setdefault(c, []).append(int(v["VRAM Total Used Memory (B)"]))
    base = {c: int(v["VRAM Total Used Memory (B)"]) for c, v in h["vram_before_start"].items()}
    health = {c: int(v["VRAM Total Used Memory (B)"]) for c, v in h["vram_after_health"].items()}
    stop = {c: int(v["VRAM Total Used Memory (B)"]) for c, v in h["vram_after_stop"].items()}
    logs = sorted(glob.glob(f"{d}/server*.log"))
    start = [l.strip() for f in logs for l in open(f) if "Nemotron-H model" in l]
    order = sorted(ranks); order = order[-1:] + order[:-1]  # rank 0 starts last, ranks 1.. in order
    out = {"samples": len(s), "seconds": round(s[-1]["t"] - s[0]["t"]),
           "ranks_peak_vmhwm_gib": {f"rank {i}": round(ranks[g]["vmhwm"] / 2**20, 2) for i, g in enumerate(order)},
           "ranks_peak_vmrss_gib": {f"rank {i}": round(ranks[g]["vmrss"] / 2**20, 2) for i, g in enumerate(order)},
           "vram_gib": {c: {"before_start": round(base[c] / G, 2), "after_health": round(health[c] / G, 2), "peak": round(max(v) / G, 2),
                            "after_stop": round(stop[c] / G, 2), "total": 30 if "32195" in h["vram_before_start"][c]["VRAM Total Memory (B)"] else 45} for c, v in vr.items()},
           "load_to_health_s": h["load_to_health_s"], "startup": start}
    json.dump(out, open(f"{d}/memory_summary.json", "w"), indent=1)
    print(d, json.dumps(out, indent=1))
