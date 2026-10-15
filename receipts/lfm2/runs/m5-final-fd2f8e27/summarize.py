import json, os, re, sys
O = os.path.expanduser("~/tf-lfm2-test/bench-final2")
def P(f): return os.path.join(O, f)
def ex(f): return os.path.exists(P(f))
def gb(s):
    m = re.match(r"([\d.]+)([KMGT])", s); v = float(m.group(1)); return round(v * {"K": 1e-6, "M": 1e-3, "G": 1, "T": 1e3}[m.group(2)] * (1.073741824 if False else 1), 2)
BUILDS = ["before", "on", "off", "mlx"]
S = {"machine": None, "versions": open(P("versions.txt")).read(), "output_dir": O,
     "builds": {"before": "#823 head 5a6ee034 (no cheap windows, no tensor units)",
                "on": "final chain tip fd2f8e27 (#821-#873 incl. exact 4-bit decode), tensor units on",
                "off": "fd2f8e27 with tensor units switched off (local test build, never committed; M1-M4 path on the M5)",
                "mlx": "mlx-lm 0.32.0 / mlx 0.32.3 (mlx_lm.server / stream_generate)"},
     "served": {}, "inproc": {}, "two_turn": {}, "token_sha": {}, "concurrent_equal": {}}
v = S["versions"].splitlines(); S["machine"] = "Apple M5 Ultra, 64-core GPU, 256 GB, macOS " + v[1].split()[-1]
for b in BUILDS:
    S["served"][b] = {}; S["token_sha"][b] = {}; S["concurrent_equal"][b] = {}
    for r in (1, 2):
        k = f"{b}_r{r}"; d = {"files": {}}
        if not ex(f"decode_{k}.json"): continue
        dec = json.load(open(P(f"decode_{k}.json")))
        for e in dec:
            n = "code" if e["prompt"] == "fibonacci-raw" else "chat"
            d[f"decode_{n}_tps"] = round(e["decode_tps_median"], 1)
            d[f"decode_{n}_tps_all"] = e["decode_tps_all"]
            S["token_sha"][b].setdefault(n, set()).update(x["token_sha"] for x in e["evidence"])
        d["files"]["decode"] = f"decode_{k}.json"
        conc = json.load(open(P(f"conc_{k}.json")))
        eq = {}
        for c in conc["cells"]:
            if c["streams"] == 4:
                d[f"conc4_{c['prompt']}"] = {"total_tps": c["aggregate_tps"], "steady_tps": c.get("steady_tps"), "slowest_ttft_s": c["ttft_s_max"]}
            eq[f"{c['prompt']}_x{c['streams']}"] = {"alone": c["alone"], "serial": c["serial"]}
            for rep in c["reps"]:
                for x in rep["evidence"]: S["token_sha"][b].setdefault(c["prompt"], set()).add(x["token_sha"])
        S["concurrent_equal"][b][f"r{r}"] = eq; d["files"]["concurrent"] = f"conc_{k}.json"
        if ex(f"footprint_{k}.txt"):
            t = open(P(f"footprint_{k}.txt")).read(); m = re.search(r"peak\):\s+(\S+)", t)
            if m: d["peak_footprint_GB"] = gb(m.group(1)); d["peak_footprint_raw"] = m.group(1); d["files"]["footprint"] = f"footprint_{k}.txt"
        if ex(f"prefill_{k}.txt"):
            pf = {}
            for line in open(P(f"prefill_{k}.txt")):
                if line.startswith("{"):
                    j = json.loads(line); pf.setdefault(str(j["tokens"]), []).append(round(j["tok_s"], 0))
            d["prefill_tps"] = pf; d["files"]["prefill"] = f"prefill_{k}.txt"
        if ex(f"cache_{k}.json"):
            c = json.load(open(P(f"cache_{k}.json"))); d["bench_cache_ttft_s"] = round(c["rows"][0]["ttft_s"], 3)
            d["bench_cache_qualified"] = c.get("qualified"); d["files"]["bench_cache"] = f"cache_{k}.json"
            S["token_sha"][b].setdefault("cache32k", set()).update(x["token_sha"] for x in c["rows"])
        if b != "mlx" and ex(f"serve_{k}.log"):
            lg = open(P(f"serve_{k}.log")).read()
            m = re.search(r"LFM2 window ms by width: (.*)", lg)
            if m:
                parts = m.group(1).split(", "); d["window_ms"] = {p.split()[0]: float(p.split()[1]) for p in parts}
            for name, pt in (("code", 14), ("chat", 29)):
                rows = re.findall(rf"prompt={pt} cached=\d+ .*?tokens=(\d+) sha=(\w+) .*?rounds=(\d+) accepted=(\d+)/(\d+)", lg)
                seq = []
                for t_, a, x, y, z in rows:
                    key = {"tokens": int(t_), "sha": a, "rounds": int(x), "accepted": int(y), "drafted": int(z)}
                    if seq and {k: seq[-1][k] for k in key} == key: seq[-1]["count"] += 1
                    else: seq.append(dict(key, count=1))
                if seq:
                    d[f"draft_{name}_sequence"] = seq
                    first = next((q for q in seq if q["tokens"] == 256), None)
                    if first: d[f"draft_{name}"] = {k: first[k] for k in ("rounds", "accepted", "drafted", "sha")}
            d["files"]["serve_log"] = f"serve_{k}.log"
        S["served"][b][f"r{r}"] = d
    S["token_sha"][b] = {k: sorted(str(x) for x in v) for k, v in S["token_sha"][b].items()}
for b in BUILDS:
    S["inproc"][b] = {}
    for n in ("2048", "8192", "32768", "65536"):
        e = {"prompt_tps": [], "decode_tps": [], "files": []}
        for r in (1, 2):
            f = f"ip_{b}_{n}_r{r}.txt"
            if not ex(f): continue
            t = open(P(f)).read()
            if b == "mlx": m = re.search(r"prompt ([\d.]+) tok/s .*decode ([\d.]+) tok/s", t)
            else: m = re.search(r"prompt \d+ in \d+-row chunks: [\d.]+ s, (\d+) tok/s.*decode \d+: ([\d.]+) tok/s", t)
            if m: e["prompt_tps"].append(float(m.group(1))); e["decode_tps"].append(float(m.group(2))); e["files"].append(f)
        S["inproc"][b][n] = e
S["inproc"]["131k"] = {}
for b in BUILDS:
    f = f"ip131_{b}.txt"
    if not ex(f): continue
    t = open(P(f)).read(); e = {"file": f}
    if b == "mlx": m = re.search(r"prompt ([\d.]+) tok/s .*decode ([\d.]+) tok/s", t)
    else: m = re.search(r"prompt \d+ in \d+-row chunks: [\d.]+ s, (\d+) tok/s.*decode \d+: ([\d.]+) tok/s", t)
    if m: e["prompt_tps"] = float(m.group(1)); e["decode_tps"] = float(m.group(2))
    m = re.search(r"(\d+)\s+peak memory footprint", t)
    if m: e["peak_gb"] = round(int(m.group(1)) / 1e9, 2)
    S["inproc"]["131k"][b] = e
for b in ("on", "mlx"):
    f = f"two_turn_{b}.json"
    if ex(f):
        j = json.load(open(P(f))); S["two_turn"][b] = {"file": f}
        for t in ("turn1", "turn2", "turn2_again"):
            u = j[t].get("usage", {}); det = u.get("prompt_tokens_details") or {}
            S["two_turn"][b][t] = {"ttft_s": round(j[t]["ttft"], 3) if j[t]["ttft"] else None, "prompt_tokens": u.get("prompt_tokens"),
                                   "cached_tokens": det.get("cached_tokens", u.get("cached_tokens")), "text": j[t]["text"][:80]}
if "mlx" in S["two_turn"] and ex("serve_tt_mlx.log"):
    proc = [int(a) for a, b_ in re.findall(r"Prompt processing progress: (\d+)/(\d+)", open(P("serve_tt_mlx.log")).read()) if a == b_]
    S["two_turn"]["mlx"]["note"] = "mlx-lm returns no usage; prompt tokens processed per request from serve_tt_mlx.log 'Prompt processing progress n/n'"
    for t, n in zip(("turn1", "turn2", "turn2_again"), proc): S["two_turn"]["mlx"][t]["processed_tokens_log"] = n
json.dump(S, open(P("summary.json"), "w"), indent=1)

# markdown
L = ["# LFM2.5-24B-A2B 4-bit on Apple M5 Ultra: final benchmark", "", S["machine"], "", "Builds:"] + [f"- **{b}**: {S['builds'][b]}" for b in BUILDS] + [""]
def g(b, r, *ks):
    d = S["served"].get(b, {}).get(r, {})
    for k in ks:
        if not isinstance(d, dict) or k not in d: return "-"
        d = d[k]
    return d
L += ["## Served (r1 / r2)", "", "| metric | " + " | ".join(BUILDS) + " |", "|---|" + "---|" * 4]
rows = [("decode code tok/s", ("decode_code_tps",)), ("decode chat tok/s", ("decode_chat_tps",)),
        ("4-stream code total", ("conc4_code", "total_tps")), ("4-stream code steady", ("conc4_code", "steady_tps")), ("4-stream code slowest TTFT s", ("conc4_code", "slowest_ttft_s")),
        ("4-stream chat total", ("conc4_chat", "total_tps")), ("4-stream chat steady", ("conc4_chat", "steady_tps")), ("4-stream chat slowest TTFT s", ("conc4_chat", "slowest_ttft_s")),
        ("bench_cache 32k cold TTFT s", ("bench_cache_ttft_s",)), ("peak footprint GB", ("peak_footprint_GB",))]
for lab, ks in rows:
    L.append(f"| {lab} | " + " | ".join(f"{g(b,'r1',*ks)} / {g(b,'r2',*ks)}" for b in BUILDS) + " |")
for n in ("2048", "8192", "32768", "65536", "131000"):
    L.append(f"| prefill {n} tok/s | " + " | ".join(f"{g(b,'r1','prefill_tps',n)} / {g(b,'r2','prefill_tps',n)}" for b in BUILDS) + " |")
for w in ("1", "2", "4", "8", "16"):
    L.append(f"| window ms width {w} | " + " | ".join(f"{g(b,'r1','window_ms',w)} / {g(b,'r2','window_ms',w)}" for b in BUILDS) + " |")
L += ["", "## Draft rounds, code prompt (server log, requests in order, run-length)", "", "| build | round | requests | tokens | rounds | accepted | drafted | sha |", "|---|---|---|---|---|---|---|---|"]
for b in ("before", "on", "off"):
    for r in ("r1", "r2"):
        q = g(b, r, "draft_code_sequence")
        for x in (q if q != "-" else []):
            L.append(f"| {b} | {r} | {x['count']} | {x['tokens']} | {x['rounds']} | {x['accepted']} | {x['drafted']} | {x['sha']} |")
L += ["", "## In process (r1, r2)", "", "| length | metric | " + " | ".join(BUILDS) + " |", "|---|---|" + "---|" * 4]
for n in ("2048", "8192", "32768", "65536"):
    for m in ("prompt_tps", "decode_tps"):
        L.append(f"| {n} | {m} | " + " | ".join(", ".join(str(x) for x in S["inproc"][b][n][m]) for b in BUILDS) + " |")
L += ["", "## In process 131,000 tokens (/usr/bin/time -l)", "", "| build | prompt tok/s | decode tok/s | peak GB |", "|---|---|---|---|"]
for b, e in S["inproc"]["131k"].items(): L.append(f"| {b} | {e.get('prompt_tps')} | {e.get('decode_tps')} | {e.get('peak_gb')} |")
L += ["", "## Two-turn resume (cache on)", "", "| server | turn | TTFT s | prompt tokens | cached tokens | processed (mlx log) |", "|---|---|---|---|---|---|"]
for b, e in S["two_turn"].items():
    for t in ("turn1", "turn2", "turn2_again"): L.append(f"| {b} | {t} | {e[t]['ttft_s']} | {e[t]['prompt_tokens']} | {e[t]['cached_tokens']} | {e[t].get('processed_tokens_log', '-')} |")
L += ["", "## token_sha", "", "| build | " + " | ".join(["code", "chat", "cache32k"]) + " |", "|---|---|---|---|"]
for b in BUILDS: L.append(f"| {b} | " + " | ".join(",".join(S["token_sha"][b].get(k, [])) for k in ("code", "chat", "cache32k")) + " |")
L += ["", "## Concurrent == solo (alone equal/unequal, serial equal/unequal)", "", "| build | round | cell | alone | serial |", "|---|---|---|---|---|"]
for b in BUILDS:
    for r, eq in S["concurrent_equal"][b].items():
        for c, v in eq.items(): L.append(f"| {b} | {r} | {c} | {v['alone']['equal']}/{v['alone']['unequal']} | {v['serial']['equal']}/{v['serial']['unequal']} |")
open(P("summary.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
