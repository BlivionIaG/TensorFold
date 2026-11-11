"""Decode tok/s of Qwen3.6-35B-A3B (MLX 4-bit) on a Mac + external GPU with a VRAM expert cache: arithmetic over cache_sim.py's
measured miss histograms and the card/Metal numbers of the proposal. No device. Needs cache_sim_results.json (cache_sim.py all).

run: python3 cost_model.py > cost_model_results.txt

One round = one decode step of N sequences. Per layer, the card runs the dense part (mixer, router, shared expert), then the
routed experts: resident ones (hits) on the card, the layer's m missed ones by a miss path:
  (a) direct   the card's GEMV reads the missed expert from host memory over Thunderbolt, m * 1,769,472 B / 3.2 GB/s a layer,
               serial on the link (the read-through variant also stores the expert in its slot, no extra link bytes);
  (b) metal    the card flags host, Metal wakes (159 us p50 card-flag-to-completion incl. its 149 us wake), runs the m experts
               against the host copy at 144 GB/s, the card's wait releases (70 us); the card computes its hits meanwhile. Paid once
               per layer that has any miss; a miss admitted to the cache is a separate 1.69 MiB DMA copy (H2D 3.17 GB/s);
  (b*) path (b) with the admission copies made free: a bound that has the hit rate of an admitting policy and no link bytes at all
  (c)  prefetch: DMA copies issued ahead by lookahead/previous-token guesses; the residual misses use (b); the link is one
               resource for copies and direct reads, so a round is also bounded below by the link's busy time.
Card compute: the dense part reads the checkpoint's non-expert bytes (placement_model.qwen36_a3b, 1.25 GiB incl. head and the
DeltaNet state read+write) at the dense GEMV rate plus 22 launches a layer; a resident expert is read at the expert GEMV rate.
Calibrations (rates and launch): `4B-fit` dense rate = the Qwen3.5-4B layer's bytes (placement_model.qwen35_4b, mixer + MLP + 2x state)
over its measured 0.52 ms a layer minus 12.3 launches of 4.4 us (about 142 GB/s), experts at 250 GB/s (the lead's figure for the
expert GEMV); `best-250` every dense kernel also at 250 GB/s (a ceiling no card result supports); `planner` 0.37 x 410 = 152 GB/s
for everything (placement_model.PROFILES[0]); `flat-0.52` 0.52 ms a layer dense regardless of size (the 4B's whole layer cost, an
upper bound for the 35B's 22 MB layers).
Rows N > 1: dense time scales as (9.1 + 7.5 N) / (9.1 + 7.5), the Vega's measured fit on the 4B (card_batches), an assumption for
the 35B [INFERENCE]; the Mac alone is the measured 88.8 / 135.8 / 176.4 / 206.9 tok/s at 1 / 2 / 4 / 8 sequences.
"""
import json
import os

from placement_model import GiB, MiB, STATE_RW, qwen35_4b, qwen36_a3b

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "cache_sim_results.json")))
EXPERT = 3 * (524288 + 32768 + 32768)  # 1,769,472 B, MLX affine 4-bit gate/up/down, weight+scales+biases
MAC_TPS = {1: 88.8, 2: 135.8, 4: 176.4, 8: 206.9}
NL = 40

T_DIRECT = EXPERT / 3.2e9  # host read through the card's GEMV, a miss
T_DMA = EXPERT / 3.17e9  # H2D copy
CROSS0 = 159e-6 + 30e-6 + 70e-6  # card flag -> Metal done (p50, wake incl.) + two dispatches + wait release
CROSS0_P99 = 479e-6 + 30e-6 + 72e-6
CROSS0_BEST = 65e-6 + 30e-6 + 25e-6  # idle-Mac event wake, release at the card's measured 10 us hop (not the case while the card is busy)
T_METAL_EXPERT = EXPERT / 144e9  # one more expert in the Metal gather_qmm, at the Mac's measured 144 GB/s


def dense_bytes(ctx=512):
    m = qwen36_a3b()
    return sum(l.dense + STATE_RW * l.state + l.kv_per_tok * ctx for l in m.layers) + m.head, sum(l.dispatches for l in m.layers)


_L4 = qwen35_4b().layers
BW_4B = sum(l.dense + STATE_RW * l.state for l in _L4) / len(_L4) / (0.52e-3 - 12.3 * 4.4e-6)  # B/s, about 142e9
CALS = {
    "4B-fit": dict(bw_d=BW_4B, bw_e=250e9, launch=4.4e-6, flat=None),
    "best-250": dict(bw_d=250e9, bw_e=250e9, launch=4.4e-6, flat=None),
    "planner": dict(bw_d=152e9, bw_e=152e9, launch=4.4e-6, flat=None),
    "flat-0.52": dict(bw_d=None, bw_e=250e9, launch=4.4e-6, flat=0.52e-3),
}


def dense_time(cal, n, speed=1.0, ctx=512):
    b, disp = dense_bytes(ctx)
    if cal["flat"] is not None:
        t1 = cal["flat"] * NL + 3 * cal["launch"] + qwen36_a3b().head / BW_4B  # the head is not a layer: its read at the 4B-fit rate
    else:
        t1 = b / cal["bw_d"] + disp * cal["launch"]
    scale = (9.1 + 7.5 * n) / (9.1 + 7.5)
    return t1 * scale / speed


def find(results, **kw):
    for r in results:
        c = r["cfg"]
        if all(c.get(k) == v for k, v in kw.items()):
            return r
    raise KeyError(kw)


def hist_of(r):
    return {int(k): v for k, v in r["hist"].items()}


def stall(path, m, cross0=CROSS0):
    if m == 0:
        return 0.0
    a = m * T_DIRECT
    b = cross0 + m * T_METAL_EXPERT
    return {"a": a, "b": b}[path]


def round_time(r, n, cal, path, speed=1.0, cross0=None, pf_loads=0.0, admit_copies=None, ctx=512):
    """Seconds for one round of n rows given a sim result r (histogram of misses a layer-round)."""
    h = hist_of(r)
    nr = sum(h.values())
    rounds = nr / NL
    tok = r["tokens"] / rounds  # rows a round (== n except when sequences finish)
    exp_stall = 0.0
    exp_miss = 0.0
    for m, c in h.items():
        s = stall(path, m, cross0 or CROSS0)
        exp_stall += s * c / rounds
        exp_miss += m * c / rounds
    hits_a = r["accesses_per_token"] * tok - exp_miss  # expert accesses a round (union over rows) minus misses
    t_hit = hits_a * EXPERT / (cal["bw_e"] or 250e9) / speed
    chain = dense_time(cal, n, speed, ctx) + t_hit + exp_stall
    link = (exp_miss * T_DIRECT if path == "a" else 0.0) + (pf_loads * tok + (admit_copies if admit_copies is not None else 0.0) * tok) * T_DMA
    return max(chain, link), dict(dense=dense_time(cal, n, speed, ctx), hit=t_hit, stall=exp_stall, link=link, miss=exp_miss, chain=chain, rows=tok)


def tps(r, n, cal, path, **kw):
    t, parts = round_time(r, n, cal, path, **kw)
    return parts["rows"] / t, t, parts


def line(label, vals, w=8):
    return f"{label:44s}" + "".join(f"{v:>{w}}" for v in vals)


def main():
    sim1 = R["sim"]["sim_n1"]
    budgets = [2, 3, 4, 5, 5.9, 8, 10, 12, 16, 24]
    mac = MAC_TPS[1]
    b, disp = dense_bytes()
    print(f"expert = {EXPERT:,} B = {EXPERT / MiB:.4f} MiB (3 x [{524288:,} packed u32 weight + {32768:,} bf16 scales + {32768:,} bf16 biases]); "
          f"T_direct {T_DIRECT * 1e3:.3f} ms, T_dma {T_DMA * 1e3:.3f} ms, Metal crossing {CROSS0 * 1e6:.0f} us + {T_METAL_EXPERT * 1e6:.1f} us a miss")
    print(f"card dense bytes a token {b / MiB:.0f} MiB (incl. head 273 MiB, DeltaNet state r+w, KV at 512), {disp} launches; "
          f"Mac alone {mac} tok/s = {1e3 / mac:.2f} ms a token")
    for cn, cal in CALS.items():
        print(f"  dense {cn}: {dense_time(cal, 1) * 1e3:.2f} ms;  all 320 expert reads resident at the expert rate {320 * EXPERT / (cal['bw_e'] or 250e9) * 1e3:.2f} ms;  "
              f"=> a fully resident card: {1e3 / (dense_time(cal, 1) * 1e3 + 320 * EXPERT / (cal['bw_e'] or 250e9) * 1e3):.0f} tok/s")
    print("\n### cost of one layer's m misses on the critical path, microseconds")
    ms = (1, 2, 4, 8, 16, 32)
    print(line("m missed experts in the layer", [str(m) for m in ms]))
    print(line("(a) direct host read over Thunderbolt", [f"{stall('a', m) * 1e6:.0f}" for m in ms]))
    print(line("(b) Metal computes them (p50 crossing)", [f"{stall('b', m) * 1e6:.0f}" for m in ms]))
    print(line("(b) at the p99 crossing", [f"{stall('b', m, CROSS0_P99) * 1e6:.0f}" for m in ms]))
    print(line("(b) at the best-case crossing (idle Mac)", [f"{stall('b', m, CROSS0_BEST) * 1e6:.0f}" for m in ms]))
    print(line("(c) DMA prefetch: link time to move m experts", [f"{m * T_DMA * 1e6:.0f}" for m in ms]))

    print("\n### 1 sequence: tok/s by policy, budget and miss path (4B-fit calibration; Mac alone 88.8). 'card GiB' = budget + 1.9 (dense 1.29, scratch 0.4, KV+state 0.2)")
    print(line("policy / path  \\ expert budget GiB", [f"{x:g}" for x in budgets]))
    print(line("  (card VRAM GiB)", [f"{x + 1.9:.1f}" for x in budgets]))
    rows = [("LRU", dict(policy="lru", mode="uniform")), ("hybrid", dict(policy="hybrid", mode="uniform")),
            ("freq+recency+prior", dict(policy="fr", mode="uniform", prior=64)), ("static CV (no admission)", dict(policy="static", mode="uniform")),
            ("Belady OPT (bound)", dict(policy="opt", mode="uniform"))]
    cal = CALS["4B-fit"]
    for name, kw in rows:
        for path in ("a", "b", "b*"):
            vals = []
            for bd in budgets:
                r = find(sim1, budget=bd, **kw)
                # path (a) is read-through (free admission, its link time is the stall); path (b) copies each admitted miss over the
                # link; (b*) and OPT are bounds without copies; a static cache admits nothing
                adm = r["admitted_per_token"] if path == "b" and kw["policy"] not in ("static", "opt") else 0.0
                vals.append(f"{tps(r, 1, cal, 'b' if path == 'b*' else path, admit_copies=adm)[0]:.1f}")
            print(line(f"{name} / ({path})", vals))
    print("\n  path (b) rows charge one DMA copy of 1.69 MiB for every admitted miss (the link bound: a round cannot be shorter than the link's busy time).")

    print("\n### calibration sensitivity, 1 sequence, LRU, 5.9 GiB expert budget (8 GiB card), tok/s [stall ms | dense ms | hit ms | link busy ms]")
    r = find(sim1, budget=5.9, policy="lru", mode="uniform")
    for cn, cal_ in CALS.items():
        for path in ("a", "b"):
            adm = r["admitted_per_token"] if path == "b" else 0.0
            t, tt, p = tps(r, 1, cal_, path, admit_copies=adm)
            print(f"  {cn:10s} ({path}): {t:5.1f} tok/s   stall {p['stall'] * 1e3:5.1f} | dense {p['dense'] * 1e3:5.1f} | hit {p['hit'] * 1e3:4.1f} | link {p['link'] * 1e3:5.1f}  round {tt * 1e3:5.1f} ms")
    for nm, c0 in (("best-case crossing 120 us", CROSS0_BEST), ("p99 crossing", CROSS0_P99)):
        t, tt, p = tps(r, 1, CALS["4B-fit"], "b", cross0=c0, admit_copies=r["admitted_per_token"])
        print(f"  4B-fit (b) with {nm}: {t:5.1f} tok/s (stall {p['stall'] * 1e3:.1f} ms, link {p['link'] * 1e3:.1f} ms)")

    print("\n### the ceiling: a card that holds everything it needs (hit 100%, nothing crosses): tok/s vs the Mac alone")
    for cn, cal_ in CALS.items():
        tt = dense_time(cal_, 1) + 320 * EXPERT / (cal_["bw_e"] or 250e9)
        print(f"  {cn:10s}: {1 / tt:5.1f} tok/s for 1 sequence ({1 / tt / mac:.2f}x of the Mac alone)")
    print("  speed-up of the card's kernels (dense and expert, with the same launches) needed to reach the Mac alone at 100% hits, 4B-fit:")
    cal = CALS["4B-fit"]
    lo, hi = 0.5, 5.0
    for _ in range(60):
        s = (lo + hi) / 2
        t = dense_time(cal, 1, s) + 320 * EXPERT / cal["bw_e"] / s
        lo, hi = (lo, s) if 1 / t > mac else (s, hi)
    print(f"    {hi:.2f}x  (dense {dense_time(cal, 1, hi) * 1e3:.2f} ms + experts {320 * EXPERT / cal['bw_e'] / hi * 1e3:.2f} ms; the launches stay at 3.9 ms of it if they do not scale, so the bytes must stream at "
          f"{(b + 320 * EXPERT) / (1e3 / mac * 1e-3 - disp * cal['launch']) / 1e9:.0f} GB/s end to end)")

    print("\n### speed needed from the card for the cache to pay (LRU, path b with copies): smallest multiplier on all the card's compute (rates and launches) that reaches 88.8 tok/s")
    cal = CALS["4B-fit"]
    print(line("budget GiB", [f"{x:g}" for x in budgets]))
    print(line("LRU hit rate %", [f"{100 * find(sim1, budget=bd, policy='lru', mode='uniform')['hit_rate']:.1f}" for bd in budgets]))
    print(line("LRU layers with a miss %", [f"{100 * find(sim1, budget=bd, policy='lru', mode='uniform')['p_layer_miss']:.1f}" for bd in budgets]))
    vals = []
    for bd in budgets:
        r = find(sim1, budget=bd, policy="lru", mode="uniform")
        lo, hi = 0.5, 50.0
        for _ in range(80):
            s = (lo + hi) / 2
            t = tps(r, 1, cal, "b", speed=s, admit_copies=r["admitted_per_token"])[0]
            lo, hi = (lo, s) if t > mac else (s, hi)
        vals.append(f"{hi:.1f}x" if hi < 49 else "never")
    print(line("multiplier on dense+expert compute", vals))
    vals = []
    for bd in budgets:
        r = find(sim1, budget=bd, policy="lru", mode="uniform")
        lo, hi = 0.5, 50.0
        for _ in range(80):
            s = (lo + hi) / 2
            t = tps(r, 1, cal, "b", speed=s, admit_copies=0.0)[0]
            lo, hi = (lo, s) if t > mac else (s, hi)
        vals.append(f"{hi:.1f}x" if hi < 49 else "never")
    print(line("  same, admissions free (throttled to 0)", vals))

    print("\n### 2 and 4 sequences (4B-fit; the card's dense part scales with the 4B's measured rows fit)")
    for n in (2, 4):
        sim = R["sim"][f"sim_n{n}"]
        print(f"\n  N={n}: Mac alone {MAC_TPS[n]} tok/s (aggregate)")
        print(line(f"  policy / path \\ expert budget GiB", [f"{x:g}" for x in budgets]))
        for name, kw in rows[:2]:
            for path in ("a", "b"):
                vals = []
                for bd in budgets:
                    r = find(sim, budget=bd, **kw)
                    adm = r["admitted_per_token"] if path == "b" else 0.0
                    vals.append(f"{tps(r, n, cal, path, admit_copies=adm)[0]:.1f}")
                print(line(f"  {name} / ({path})", vals))
        r = find(sim, budget=24, policy="lru", mode="uniform")
        print(f"  fully resident (24 GiB card): {tps(r, n, cal, 'b')[0]:.1f} tok/s")

    print("\n### (c) prefetch: tok/s, 1 sequence, 4B-fit, (b) misses; cap 18 loads a token; copies = prefetch loads + admitted copies (LRU admits every miss)")
    pf = R["prefetch"]
    print(f"{'policy':8s}{'GiB':>5s}{'prefetch':>10s}{'cap':>5s}{'hit %':>7s}{'pf/tok':>8s}{'adm/tok':>9s}{'tok/s':>8s}{'(a)':>8s}")
    for r in pf:
        c = r["cfg"]
        if c["cap"] not in (None, 18) or c["policy"] == "fr":
            continue
        pfl, adm = r["pf_loads_per_token"], r["admitted_per_token"]
        t, _, _ = tps(r, 1, cal, "b", pf_loads=pfl, admit_copies=adm)
        ta, _, _ = tps(r, 1, cal, "a", pf_loads=pfl)
        print(f"{c['policy']:8s}{c['budget']:>5g}{str(c['prefetch']):>10s}{str(c['cap']):>5s}{100 * r['hit_rate']:>7.1f}{pfl:>8.1f}{adm:>9.1f}{t:>8.1f}{ta:>8.1f}")

    print("\n### admission throttle on the Metal path (1 sequence): tok/s with the link busy time charged")
    print(f"{'policy':8s}{'GiB':>5s}{'p':>6s}{'hit %':>7s}{'copies/tok':>11s}{'link ms':>9s}{'stall ms':>9s}{'tok/s':>8s}")
    for r in R["admit"]:
        c = r["cfg"]
        t, tt, p = tps(r, 1, cal, "b", admit_copies=r["admitted_per_token"])
        print(f"{c['policy']:8s}{c['budget']:>5g}{c['admit_p']:>6g}{100 * r['hit_rate']:>7.1f}{r['admitted_per_token']:>11.1f}{p['link'] * 1e3:>9.1f}{p['stall'] * 1e3:>9.1f}{t:>8.1f}")

    print("\n### bigger cards (1 sequence). VRAM -> expert budget = VRAM - 1.9 GiB; compute from the planner's profiles (7900 XTX-class 960 GB/s eff 0.5, 2.5 us launches;")
    print("    5090-class 1792 GB/s eff 0.55, 1.5 us) [INFERENCE: published peaks, one assumed efficiency each]; misses by Metal (b); same TB4 link and Metal numbers")
    profiles = [("7900 XTX-class", 960e9, 0.50, 2.5e-6), ("5090-class", 1792e9, 0.55, 1.5e-6)]
    vrams = [8, 10, 12, 16, 20, 24, 32]
    print(line("card \\ VRAM GiB", [str(v) for v in vrams]))
    for pname, peak, eff, launch in profiles:
        cal_ = dict(bw_d=peak * eff, bw_e=peak * eff, launch=launch, flat=None)
        vals = []
        for v in vrams:
            bd = v - 1.9
            if bd >= 16.875:
                r = find(sim1, budget=24, policy="lru", mode="uniform")
                vals.append(f"{tps(r, 1, cal_, 'b')[0]:.0f}")
                continue
            # nearest simulated budgets: interpolate tok/s linearly in budget between simulated points
            pts = sorted(set(x["cfg"]["budget"] for x in sim1 if x["cfg"]["policy"] == "lru" and x["cfg"]["mode"] == "uniform"))
            lo_b = max(p for p in pts if p <= bd)
            hi_b = min(p for p in pts if p >= bd)
            tl = tps(find(sim1, budget=lo_b, policy="lru", mode="uniform"), 1, cal_, "b", admit_copies=find(sim1, budget=lo_b, policy="lru", mode="uniform")["admitted_per_token"])[0]
            th = tps(find(sim1, budget=hi_b, policy="lru", mode="uniform"), 1, cal_, "b", admit_copies=find(sim1, budget=hi_b, policy="lru", mode="uniform")["admitted_per_token"])[0]
            vals.append(f"{tl if hi_b == lo_b else tl + (th - tl) * (bd - lo_b) / (hi_b - lo_b):.0f}")
        print(line(f"{pname} (LRU, Metal misses)", vals))
    for pname, peak, eff, launch in profiles:
        cal_ = dict(bw_d=peak * eff, bw_e=peak * eff, launch=launch, flat=None)
        r = find(sim1, budget=24, policy="lru", mode="uniform")
        print(f"  {pname} resident (hit 100%): 1 sequence {tps(r, 1, cal_, 'b')[0]:.0f} tok/s vs Mac 88.8 ({tps(r, 1, cal_, 'b')[0] / 88.8:.2f}x)")
    print("  (the proposal's own planner gives resident 158 / 299 tok/s at 8k for these two; this model drops the 8k KV read and uses ctx 512)")


if __name__ == "__main__":
    main()
