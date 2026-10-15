"""Collects bench-m4-final2 raw files into summary.json and summary.md. Every value is read from a file in this directory."""
import json, os, re, statistics as st

O = os.path.dirname(os.path.abspath(__file__))
B = ["before", "tip", "mlx"]
LENS = [2048, 8192, 32768, 65536]


def rd(f):
    p = os.path.join(O, f)
    return open(p).read() if os.path.exists(p) else None


def js(f):
    t = rd(f)
    if t is None:
        return None
    try:
        return json.loads(t)
    except Exception:
        return None


DONE = re.compile(r"\[tensorfold\] done (\S+) prompt=(\d+) cached=(\d+) .*?tokens=(\d+) sha=(\w+) finish=(\w+) rounds=(\d+) accepted=(\d+)/(\d+)")


def done_lines(text):
    out = []
    for l in (text or "").splitlines():
        m = DONE.search(l)
        if m:
            out.append(dict(id=m[1], kind="chat" if m[1].startswith("chatcmpl") else "completion", prompt=int(m[2]), cached=int(m[3]),
                            tokens=int(m[4]), sha=m[5], finish=m[6], rounds=int(m[7]), accepted=int(m[8]), drafted=int(m[9])))
    return out


def window(text):
    m = re.search(r"LFM2 window ms by width: (.*)", text or "")
    return m[1].strip() if m else None


def footprint(f):
    t = rd(f) or ""
    m = re.search(r"Physical footprint \(peak\):\s*([\d.]+)([GMK])", t)
    if not m:
        return None
    v = float(m[1])
    return round(v if m[2] == "G" else v / 1024 if m[2] == "M" else v / 1024 / 1024, 2)


def conc_cells(d):
    res = {}
    for c in (d or {}).get("cells", []):
        res[f"{c['prompt']}_{c['streams']}"] = dict(total_tps=c.get("aggregate_tps"), steady_tps=c.get("steady_tps"), per_stream_tps=c.get("per_stream_tps"),
                                                   slowest_ttft_s=c.get("ttft_s_max"), alone=c.get("alone"), serial=c.get("serial"), failed=c.get("failed"))
    return res


served = {}
shas = {}
for b in B:
    served[b] = {}
    for r in ((1, 2, "1rerun") if b == "mlx" and os.path.exists(os.path.join(O, "decode_mlx_r1rerun.json")) else (1, 2)):
        e = {}
        dec = js(f"decode_{b}_r{r}.json")
        dtxt = rd(f"decode_{b}_r{r}.txt") or ""
        rows = [json.loads(l) for l in dtxt.splitlines() if l.startswith("{")]
        e["decode_tps"] = {("code" if "raw" in x["prompt"] else "chat"): dict(median=round(x["decode_tps_median"], 1), all=x["decode_tps_all"], ttft_s=round(x["ttft_s_median"], 3)) for x in rows}
        e["concurrent"] = conc_cells(js(f"conc_{b}_r{r}.json"))
        e["four_stream"] = {k.split("_")[0]: dict(total_tps=v["total_tps"], steady_tps=v["steady_tps"], slowest_ttft_s=v["slowest_ttft_s"]) for k, v in e["concurrent"].items() if k.endswith("_4")}
        pf = js(f"prefill_{b}_r{r}.json")
        if pf:
            e["prefill_tps"] = {}
            for x in pf:
                e["prefill_tps"].setdefault(str(x["tokens"]), []).append(round(x["tok_s"], 1))
        c = js(f"cache_{b}_r{r}.json")
        if c:
            e["bench_cache_ttft_s"] = round(c["rows"][0]["ttft_s"], 2)
            e["bench_cache_first_row"] = {k: c["rows"][0].get(k) for k in ("prompt_tokens", "cached_tokens", "cache_state", "token_sha", "decode_tps")}
        e["peak_footprint_gb"] = footprint(f"footprint_{b}_r{r}.txt")
        log = rd(f"serve_{b}_r{r}.log")
        if b != "mlx":
            e["window_ms"] = window(log)
            dl = done_lines(log)
            # bench_openai runs code (completions) then chat, 256 tokens each, before bench_concurrent; take the first such runs
            d256 = [x for x in dl if x["tokens"] == 256]
            code, chat = [], []
            for x in d256:
                if x["kind"] == "completion" and not chat:
                    code.append(x)
                elif x["kind"] == "chat":
                    chat.append(x)
                else:
                    break
            e["draft_code"] = [dict(rounds=x["rounds"], accepted=x["accepted"], drafted=x["drafted"]) for x in code]
            e["draft_chat"] = [dict(rounds=x["rounds"], accepted=x["accepted"], drafted=x["drafted"]) for x in chat]
            e["requests"] = [{k: x[k] for k in ("kind", "prompt", "cached", "tokens", "sha", "rounds", "accepted", "drafted")} for x in dl]
            for x in dl:
                shas.setdefault(f"{x['kind']} prompt={x['prompt']} tokens={x['tokens']}", {}).setdefault(f"{b}_r{r}", set()).add(x["sha"])
        else:
            for k, v in e["concurrent"].items():
                pass
            cj = js(f"conc_{b}_r{r}.json") or {}
            for c2 in cj.get("cells", []):
                for rep in c2.get("reps", []):
                    for ev in rep.get("evidence", []):
                        if ev.get("token_sha"):
                            shas.setdefault(f"{'completion' if c2['prompt'] == 'code' else 'chat'} prompt={ev['prompt_tokens']} tokens={ev['tokens']} (client)", {}).setdefault(f"{b}_r{r}", set()).add(ev["token_sha"])
        served[b][f"r{r}"] = e

token_sha = {k: {kk: sorted(vv) for kk, vv in v.items()} for k, v in sorted(shas.items())}
tf_equal = {}
for k, v in token_sha.items():
    bs = {kk: vv for kk, vv in v.items() if not kk.startswith("mlx")}
    befores = set(sum([vv for kk, vv in bs.items() if kk.startswith("before")], []))
    tips = set(sum([vv for kk, vv in bs.items() if kk.startswith("tip")], []))
    if befores and tips:
        tf_equal[k] = befores == tips and len(befores) == 1

concurrent_equal = {}
for b in B:
    for r in ((1, 2, "1rerun") if b == "mlx" and os.path.exists(os.path.join(O, "decode_mlx_r1rerun.json")) else (1, 2)):
        cells = served[b][f"r{r}"]["concurrent"]
        a = dict(equal=0, unequal=0, failed=0)
        s = dict(equal=0, unequal=0, failed=0)
        for c in cells.values():
            for k in a:
                a[k] += (c["alone"] or {}).get(k, 0)
                s[k] += (c["serial"] or {}).get(k, 0)
        concurrent_equal[f"{b}_r{r}"] = dict(alone=a, serial=s)

TFL = re.compile(r"prompt (\d+) in (\d+)-row chunks: ([\d.]+) s, (\d+) tok/s.*?decode (\d+): ([\d.]+) tok/s")
MXL = re.compile(r"^(\d+) prompt ([\d.]+) tok/s \(([\d.]+) s\) decode ([\d.]+) tok/s peak ([\d.]+) GB", re.M)


def parse_inproc(b, f):
    t = rd(f) or ""
    if b == "mlx":
        m = MXL.search(t)
        return dict(prompt_tps=float(m[2]), prompt_s=float(m[3]), decode_tps=float(m[4]), peak_gb=float(m[5])) if m else None
    m = TFL.search(t)
    return dict(prompt_tps=float(m[4]), prompt_s=float(m[3]), chunk=int(m[2]), decode_n=int(m[5]), decode_tps=float(m[6])) if m else None


inproc = {}
for b in B:
    inproc[b] = {}
    for L in LENS:
        rr = [parse_inproc(b, f"inproc_{b}_p{L}_r{r}.txt") for r in (1, 2)]
        inproc[b][str(L)] = dict(prompt_tps=[x and x["prompt_tps"] for x in rr], decode_tps=[x and x["decode_tps"] for x in rr],
                                 files=[f"inproc_{b}_p{L}_r{r}.txt" for r in (1, 2)])
        if b == "mlx":
            inproc[b][str(L)]["peak_gb"] = [x and x["peak_gb"] for x in rr]
inproc["131k"] = {}
for b in B:
    t = rd(f"inproc_{b}_p131000.txt") or ""
    x = parse_inproc(b, f"inproc_{b}_p131000.txt") or {}
    m = re.search(r"(\d+)\s+peak memory footprint", t)
    m2 = re.search(r"([\d.]+) real", t)
    x["time_peak_footprint_gb"] = round(int(m[1]) / 2**30, 2) if m else None
    x["wall_s"] = float(m2[1]) if m2 else None
    x["file"] = f"inproc_{b}_p131000.txt"
    inproc["131k"][b] = x

window_rounds = {b: [window(rd(f"window_{b}_r{r}.log")) for r in (1, 2, 3)] for b in ("before", "tip")}

two_turn = {}
for b in ("tip", "mlx"):
    d = js(f"two_turn_{b}.json")
    if d:
        two_turn[b] = {k: dict(ttft_s=d[k]["ttft"] and round(d[k]["ttft"], 3), total_s=round(d[k]["total"], 2), usage=d[k]["usage"], text_head=d[k]["text"][:120]) for k in ("turn1", "turn2", "turn2_again")}
        two_turn[b]["turn2_equals_again"] = d["turn2"]["text"] == d["turn2_again"]["text"]
    if b == "tip":
        two_turn.setdefault(b, {})["server_requests"] = [{k: x[k] for k in ("kind", "prompt", "cached", "tokens", "sha")} for x in done_lines(rd("serve_twoturn_tip.log"))]

summary = dict(
    machine=dict(chip="Apple M4 Pro", gpu_cores=20, memory_gb=48, model_id="Mac16,7", macos="26.5.1 (25F80)",
                 note="Machine was in interactive use by the user during the whole run (see load/top in progress.log); another agent also built and ran tf-lfm2-exact between steps. Served mlx r1 (12:33-12:36) OVERLAPPED one 8-bit tf-lfm2-exact run by another agent (per coordinator) and is disturbed; mlx r1rerun repeats it at the end.", overlapped=["served mlx r1"]),
    versions=dict(mlx_lm="0.32.0", mlx="0.32.3", model="LiquidAI/LFM2.5-24B-A2B-MLX-4bit @ 667408d4"),
    builds=dict(before="5a6ee034 (#823 head): m5build", tip=(rd("tip_commit.txt") or "").strip()[:8] + " (chain tip #821-#873): bin-egate", binaries=rd("binaries_sha256_start.txt")),
    served=served, inproc=inproc, window_rounds=window_rounds, two_turn=two_turn, token_sha=token_sha, token_sha_before_equals_tip=tf_equal,
    concurrent_equal=concurrent_equal)
json.dump(summary, open(os.path.join(O, "summary.json"), "w"), indent=1)

# ---- markdown ----
md = ["# LFM2.5-24B-A2B 4-bit, Apple M4 Pro 20-core GPU 48 GB, macOS 26.5.1", "",
      "Builds: before = 5a6ee034 (#823 head), tip = " + summary["builds"]["tip"] + ", mlx-lm 0.32.0 / mlx 0.32.3.", "",
      summary["machine"]["note"], ""]
md += ["## Served (port 8091, --context 131072, --prompt-cache-gib 0)", "",
       "| build | round | decode code tok/s | decode chat tok/s | 4-stream code total / steady / slowest TTFT | 4-stream chat total / steady / slowest TTFT | bench_cache 32k TTFT s | peak footprint GB |",
       "|---|---|---|---|---|---|---|---|"]
for b in B:
    for r in served[b]:
        e = served[b][r]
        fs = e.get("four_stream", {})
        f4 = lambda p: (f"{fs[p]['total_tps']} / {fs[p]['steady_tps']} / {fs[p]['slowest_ttft_s']}" if p in fs else "-")
        md.append(f"| {b} | {r} | {e['decode_tps'].get('code', {}).get('median', '-')} | {e['decode_tps'].get('chat', {}).get('median', '-')} | {f4('code')} | {f4('chat')} | {e.get('bench_cache_ttft_s', '-')} | {e.get('peak_footprint_gb', '-')} |")
md += ["", "## Served cold prefill tok/s (token arrays)", "", "| build | round | 2048 | 8192 | 32768 | 65536 | 131000 |", "|---|---|---|---|---|---|---|"]
for b in ("before", "tip"):
    for r in ("r1", "r2"):
        p = served[b][r].get("prefill_tps", {})
        md.append(f"| {b} | {r} | " + " | ".join(", ".join(str(v) for v in p.get(str(L), [])) or "-" for L in [2048, 8192, 32768, 65536, 131000]) + " |")
md += ["", "## Draft (code prompt, bench_openai 256 tokens, per request rounds / accepted / drafted)", "", "| build | round | code | chat |", "|---|---|---|---|"]
for b in ("before", "tip"):
    for r in ("r1", "r2"):
        e = served[b][r]
        f = lambda xs: "; ".join(f"{x['rounds']} / {x['accepted']} / {x['drafted']}" for x in xs)
        md.append(f"| {b} | {r} | {f(e['draft_code'])} | {f(e['draft_chat'])} |")
md += ["", "## Window ms by width (load line)", "", "| build | session | line |", "|---|---|---|"]
for b in ("before", "tip"):
    for r in ("r1", "r2"):
        md.append(f"| {b} | served {r} | {served[b][r].get('window_ms')} |")
    for i, w in enumerate(window_rounds[b]):
        md.append(f"| {b} | window round {i + 1} | {w} |")
md += ["", "## In process (tf-lfm2-bench / mlx_cold.py), r1, r2", "", "| length | build | prompt tok/s | decode tok/s |", "|---|---|---|---|"]
for L in LENS:
    for b in B:
        x = inproc[b][str(L)]
        md.append(f"| {L} | {b} | {x['prompt_tps']} | {x['decode_tps']} |")
md += ["", "## 131000 in process (/usr/bin/time -l)", "", "| build | prompt tok/s | prompt s | decode tok/s | peak footprint GB (time -l) | wall s |", "|---|---|---|---|---|---|"]
for b in B:
    x = inproc["131k"][b]
    md.append(f"| {b} | {x.get('prompt_tps')} | {x.get('prompt_s')} | {x.get('decode_tps')} | {x.get('time_peak_footprint_gb')} | {x.get('wall_s')} |")
md += ["", "## Two-turn resume (prompt cache on)", "", "| build | turn1 TTFT s | turn2 TTFT s | turn2 again TTFT s | turn2 usage | turn2 == again |", "|---|---|---|---|---|---|"]
for b, d in two_turn.items():
    if "turn1" in d:
        md.append(f"| {b} | {d['turn1']['ttft_s']} | {d['turn2']['ttft_s']} | {d['turn2_again']['ttft_s']} | {json.dumps(d['turn2']['usage'])} | {d['turn2_equals_again']} |")
if two_turn.get("tip", {}).get("server_requests"):
    md += ["", "tip server: " + "; ".join(f"prompt={x['prompt']} cached={x['cached']} tokens={x['tokens']}" for x in two_turn["tip"]["server_requests"])]
md += ["", "## token_sha (server log), before vs tip", "", "| request | per session | before == tip |", "|---|---|---|"]
for k, v in token_sha.items():
    md.append(f"| {k} | " + "; ".join(f"{kk}: {','.join(vv)}" for kk, vv in v.items()) + f" | {tf_equal.get(k, '-')} |")
md += ["", "## Concurrent == solo (bench_concurrent --alone --serial)", "", "| session | alone equal / unequal / failed | serial equal / unequal / failed |", "|---|---|---|"]
for k, v in concurrent_equal.items():
    md.append(f"| {k} | {v['alone']['equal']} / {v['alone']['unequal']} / {v['alone']['failed']} | {v['serial']['equal']} / {v['serial']['unequal']} / {v['serial']['failed']} |")
md += ["", "## Notes", "",
       "- mlx-lm chat decode (~300+ tok/s) comes with a ~1.9 s TTFT (decode_mlx_*.txt): the rate is measured after the first streamed token, so it is not comparable to TensorFold's chat decode; the code (raw completion) row is the comparable one.",
       "- mlx concurrent 'failed' counts mean the equality check could not run (mlx-lm returns no token_sha; output_sha256 only), not that outputs differed.",
       "- Gate (egate/progress.log, copied as gate_progress.log): host/build/tile exit 0, exact 4-bit 68 passed 0 failed; exact 6-bit and 8-bit crashed (exit 139) in the gate run; the coordinator released the HOLD and the run proceeded on the unchanged fd2f8e27 binaries (binaries_sha256_start/end.txt identical).",
       "- In-process mlx uses mlx_cold.py (mlx-lm default prefill step); decode lengths match TensorFold (128 at 2k/8k, 64 at 32k/64k/131k)."]
open(os.path.join(O, "summary.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
