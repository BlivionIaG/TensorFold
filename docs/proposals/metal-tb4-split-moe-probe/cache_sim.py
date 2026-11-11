"""Expert-cache simulator over real router traces of Qwen3.6-35B-A3B (MLX 4-bit). No eGPU; pure replay of routing_trace.py output.

run (numpy; /tmp/mlxenv/bin/python has it): python3 cache_sim.py locality            -> temporal locality / skew / prefill union tables
     python3 cache_sim.py sim [N ...]          -> hit-rate tables for decode batches of N sequences (default 1 2 4)
     python3 cache_sim.py prefetch             -> lookahead / previous-token prefetch tables
     python3 cache_sim.py generalise           -> static-profile transfer between prompts and categories
     python3 cache_sim.py all > cache_sim_results.txt ;  results are also dumped to cache_sim_results.json for cost_model.py

Model of the cache. One slot holds one routed expert of one layer in the checkpoint's own MLX affine 4-bit layout: gate, up
and down projections, each a packed-u32 weight tile, bf16 scales and bf16 biases (group 64): 3 x (524,288 + 32,768 + 32,768)
= 1,769,472 bytes (1.6875 MiB), read from the safetensors headers. A budget of B GiB is B*2^30 / 1,769,472 slots, 40 layers
share it equally (`uniform`) or, for static profiles and `global` LRU, by popularity. A decode step is a round: for each layer
in order, the union over the rows (sequences) in the batch of the routed experts is the access set; a hit is an access set
member that is resident when the layer starts. Misses are served by the miss path (priced in cost_model.py); whether a missed
expert is also admitted to the cache is a policy parameter (read-through admission is free on the direct-read path, a copy over
the link on the Metal path).
Policies: static (profile top-frequency, never changes), lru, lfu (history counts), fr (decayed frequency, admitted only if
its score beats the weakest resident: TinyLFU-like), hybrid (half the slots pinned static + half LRU), opt (Belady with bypass,
an upper bound no online policy reaches). The profile is always trained on prompts other than the ones tested (4 folds by prompt),
except where `insample` is named.
Prefetch: before layer l runs, guessed experts for l (router lookahead from routing_trace.py's `preds`, or the previous round's
set) are loaded into the cache if absent, at most `cap` loads a round over all layers (the link allows about 20 a token); lead
time is not simulated, only the cap [INFERENCE: lookahead distance >= 3 layers is needed for an expert (0.56 ms) to arrive].
"""
import collections
import glob
import json
import math
import os
import random
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRACES = os.path.join(HERE, "traces")
NL, NE, TOPK = 40, 256, 8
EXPERT_BYTES = 3 * (524288 + 32768 + 32768)  # 1,769,472: gate, up, down each weight u32 + scales bf16 + biases bf16
GIB = 2**30
FOLDS = 4
PF_NAMES = ["A0", "A1", "A2", "B1", "B2"]


def slots(budget_gib):
    return int(budget_gib * GIB // EXPERT_BYTES)


def load(directory=TRACES):
    """name -> dict(ids (T,40,8) u8 of decode tokens, preds (T,40,5,8), pre_ids (P,40,8) prefill)."""
    out = {}
    meta = {m["name"]: m for m in json.load(open(os.path.join(directory, "meta.json")))}
    for f in sorted(glob.glob(os.path.join(directory, "*.npz"))):
        name = os.path.basename(f)[:-4]
        d = np.load(f)
        dec = d["kind"] == 1
        out[name] = dict(ids=d["ids"][dec], preds=d["preds"][dec], pre_ids=d["ids"][~dec], cat=meta[name]["cat"], w=d["w"][dec])
    return out


# ------------------------------------------------------------------ policies
class Cache:
    """One pool of slots (a layer, or all layers in `global` mode). Keys are ints."""

    def __init__(self, cap, policy, pinned=(), prior=None, decay=0.0, stamp=None):
        self.cap, self.policy = cap, policy
        self.pinned = set(pinned)
        self.od = collections.OrderedDict()  # lru order for lru/hybrid/lfu tie-break
        for k in self.pinned:
            self.od[k] = None
        self.cnt = collections.defaultdict(float)  # lfu history counts / fr decayed scores
        self.last = {}
        self.decay = decay
        self.t = 0
        if prior:
            for k, v in prior.items():
                self.cnt[k] = v
        self.nfree_lru = cap - len(self.pinned)

    # ---- score helpers (fr)
    def score(self, k):
        return self.cnt[k] * self.decay ** (self.t - self.last.get(k, self.t))

    def bump(self, k):
        if self.policy == "fr":
            self.cnt[k] = self.score(k) + 1.0
            self.last[k] = self.t
        elif self.policy == "lfu":
            self.cnt[k] += 1.0

    def contains(self, k):
        return k in self.od

    def touch(self, k):
        if k in self.od and k not in self.pinned:
            self.od.move_to_end(k)

    def admit(self, k, protect=()):
        """Insert k if the policy accepts; returns True if k is now resident."""
        if k in self.od:
            return True
        if self.policy == "static":
            return False
        if self.policy == "hybrid":
            if self.nfree_lru <= 0:
                return False
        n_res = len(self.od)
        if n_res >= self.cap:
            victim = self._victim(k, protect)
            if victim is None:
                return False
            del self.od[victim]
        self.od[k] = None
        return True

    def _victim(self, k, protect):
        pol = self.policy
        if pol in ("lru", "hybrid"):
            for v in self.od:  # oldest first
                if v not in self.pinned and v not in protect:
                    return v
            return None
        if pol == "lfu":
            best, bs = None, None
            for v in self.od:
                if v in protect:
                    continue
                s = self.cnt[v]
                if bs is None or s < bs:
                    best, bs = v, s
            return best
        if pol == "fr":
            best, bs = None, None
            for v in self.od:
                if v in protect:
                    continue
                s = self.score(v)
                if bs is None or s < bs:
                    best, bs = v, s
            if best is None or self.score(k) <= bs:  # TinyLFU-style admission: the newcomer must beat the weakest
                return None
            return best
        raise ValueError(pol)

    def hybrid_evict_ok(self):
        return True


def static_sets(counts, cap_total, mode):
    """counts (40,256) -> list of 40 sets of resident experts. mode uniform: cap_total//40 each; global: top cap_total overall."""
    sets = [set() for _ in range(NL)]
    if mode == "uniform":
        c = cap_total // NL
        for l in range(NL):
            sets[l] = set(np.argsort(-counts[l], kind="stable")[: min(c, NE)].tolist())
    else:
        flat = np.argsort(-counts.reshape(-1), kind="stable")[: min(cap_total, NL * NE)]
        for i in flat.tolist():
            sets[i // NE].add(i % NE)
    return sets


# ------------------------------------------------------------------ schedules
def make_rounds(data, names, nbatch, seed):
    """Rounds for the prompts in `names`: shuffled, grouped nbatch at a time, stepped in lockstep (a finished prompt leaves).
    A round is a dict(ids=(rows,40,8), preds=(rows,40,5,8), group=g, step=i)."""
    rng = random.Random(seed)
    names = list(names)
    rng.shuffle(names)
    rounds = []
    for g in range(0, len(names), nbatch):
        grp = names[g : g + nbatch]
        T = max(len(data[n]["ids"]) for n in grp)
        for i in range(T):
            live = [n for n in grp if i < len(data[n]["ids"])]
            rounds.append(dict(ids=np.stack([data[n]["ids"][i] for n in live]), preds=np.stack([data[n]["preds"][i] for n in live]),
                               group=g, step=i, rows=len(live)))
    return rounds


def to_sets(r):
    """per-layer union of the batch's expert sets, as tuples of ints"""
    ids = r["ids"]
    return [tuple(sorted(set(ids[:, l, :].reshape(-1).tolist()))) for l in range(NL)]


def guess_sets(r, variant):
    k = PF_NAMES.index(variant)
    p = r["preds"][:, :, k, :]  # (rows,40,8)
    out = []
    for l in range(NL):
        s = set(p[:, l, :].reshape(-1).tolist())
        s.discard(255)
        out.append(s)
    return out


def folds_of(names):
    names = sorted(names)
    return [[n for i, n in enumerate(names) if i % FOLDS == f] for f in range(FOLDS)]


def profile(data, names):
    c = np.zeros((NL, NE), dtype=np.float64)
    for n in names:
        ids = data[n]["ids"]
        for l in range(NL):
            c[l] += np.bincount(ids[:, l, :].reshape(-1), minlength=NE)
    return c


# ------------------------------------------------------------------ the simulator
def run(data, test_names, train_names, cfg, seed=0):
    """cfg keys: policy, budget (GiB), n (batch), mode uniform|global, prefetch (None|variant|'prev'|'prev+variant'), cap (loads a
    round for prefetch, None=unlimited), admit (True|False|int cap per round), decay (fr), cold (reset per group)."""
    policy, nbatch = cfg["policy"], cfg.get("n", 1)
    mode = cfg.get("mode", "uniform")
    total = min(slots(cfg["budget"]), NL * NE)
    rounds = make_rounds(data, test_names, nbatch, seed)
    prefetch = cfg.get("prefetch")
    cap_pf = cfg.get("cap")
    admit_cap = cfg.get("admit", True)
    admit_p = cfg.get("admit_p", 1.0)  # each missed expert is admitted with this probability (a throttle on admission copies)
    rnd = random.Random(seed + 17)
    decay = 0.5 ** (1.0 / (cfg.get("halflife", 128) * (NL if mode == "global" else 1)))
    counts = profile(data, train_names) if train_names else None
    if policy == "static_in":  # in-sample upper bound
        counts = profile(data, test_names)
        policy = "static"
    per = total // NL
    if policy == "opt":
        return run_opt(rounds, per, mode, total, nbatch, static_sets(counts, total, mode))

    caches, key = [], None
    if policy == "static":
        sets = static_sets(counts, total, mode)
        caches = [Cache(max(len(s), 1), "static", pinned=s) for s in sets]
        key = lambda l, e: e
        cache_of = lambda l: caches[l]
    elif mode == "global":
        pin = ()
        prior = None
        if policy == "hybrid":
            hs = static_sets(counts, total // 2, "global")
            pin = {(l << 8) | e for l in range(NL) for e in hs[l]}
        if policy in ("fr", "lfu") and counts is not None and cfg.get("prior"):
            prior = {(l << 8) | e: counts[l, e] / max(counts[l].sum(), 1) * cfg["prior"] for l in range(NL) for e in range(NE)}
        g = Cache(total, policy, pinned=pin, prior=prior, decay=decay)
        caches = [g]
        key = lambda l, e: (l << 8) | e
        cache_of = lambda l: g
    else:
        for l in range(NL):
            pin, prior = (), None
            if policy == "hybrid":
                pin = set(np.argsort(-counts[l], kind="stable")[: per // 2].tolist())
            if policy in ("fr", "lfu") and counts is not None and cfg.get("prior"):
                prior = {e: counts[l, e] / max(counts[l].sum(), 1) * cfg["prior"] for e in range(NE)}
            caches.append(Cache(per, policy, pinned=pin, prior=prior, decay=decay))
        key = lambda l, e: e
        cache_of = lambda l: caches[l]

    if policy != "static" and not cfg.get("cold") and cfg.get("prime", True) and counts is not None:
        # deployment warm start: the cache holds the offline profile's most popular experts (most popular most recent)
        if mode == "global":
            for i in reversed(np.argsort(-counts.reshape(-1), kind="stable")[:total].tolist()):
                k = ((i // NE) << 8) | (i % NE)
                if k not in caches[0].od and len(caches[0].od) < caches[0].cap:
                    caches[0].od[k] = None
        else:
            for l in range(NL):
                c = caches[l]
                for e in reversed(np.argsort(-counts[l], kind="stable")[:per].tolist()):
                    if e not in c.od and len(c.od) < c.cap:
                        c.od[e] = None

    nrounds = 0
    acc = hit = 0
    pf_loads = pf_used = admitted = 0
    hist = collections.Counter()  # misses per (layer, round)
    miss_by_layer = np.zeros(NL)
    acc_by_layer = np.zeros(NL)
    miss_by_step = collections.defaultdict(lambda: [0, 0])  # decode position (first 512) -> [misses, rows], for the cold-start curve
    rows_total = 0
    prev = [()] * NL
    last_group = None
    for r in rounds:
        if cfg.get("cold") and r["group"] != last_group:
            for c in caches:
                c.__init__(c.cap, c.policy, pinned=c.pinned, decay=decay)
            prev = [()] * NL
        last_group = r["group"]
        sets = to_sets(r)
        gs = guess_sets(r, prefetch.split("+")[-1]) if prefetch and prefetch != "prev" else None
        budget_pf = cap_pf if cap_pf is not None else 10**9
        budget_ad = (10**9 if admit_cap is True else (0 if admit_cap is False else admit_cap))
        rows_total += r["rows"]
        for l in range(NL):
            c = cache_of(l)
            c.t += 1
            S = sets[l]
            Sset = set(S)
            pre = set()
            if prefetch and policy != "static":
                cand = set()
                if prefetch.startswith("prev"):
                    cand |= set(prev[l])
                if gs is not None:
                    cand |= gs[l]
                for e in sorted(cand):
                    if budget_pf <= 0:
                        break
                    k = key(l, e)
                    if not c.contains(k):
                        if c.admit(k, protect={key(l, x) for x in cand}):
                            budget_pf -= 1
                            pf_loads += 1
                            pre.add(e)
                        else:
                            continue
                    else:
                        c.touch(k)
            misses = []
            for e in S:
                k = key(l, e)
                c.bump(k)
                if c.contains(k):
                    hit += 1
                    c.touch(k)
                    if e in pre:
                        pf_used += 1
                else:
                    misses.append(e)
            acc += len(S)
            acc_by_layer[l] += len(S)
            m = len(misses)
            miss_by_layer[l] += m
            hist[m] += 1
            if r["step"] < 512:
                miss_by_step[r["step"]][0] += m
                if l == 0:
                    miss_by_step[r["step"]][1] += r["rows"]
            if policy != "static":
                prot = {key(l, x) for x in S}
                for e in misses:
                    if budget_ad > 0 and (admit_p >= 1.0 or rnd.random() < admit_p):
                        if c.admit(key(l, e), protect=prot):
                            budget_ad -= 1
                            admitted += 1
            prev[l] = S
        nrounds += 1
    toks = rows_total
    return dict(cfg=cfg, hit_rate=hit / acc, accesses=acc, miss_per_token=(acc - hit) / toks, rounds=nrounds, tokens=toks,
                accesses_per_token=acc / toks, hist={int(k): v for k, v in sorted(hist.items())},
                p_layer_miss=1 - hist.get(0, 0) / (nrounds * NL), pf_loads_per_token=pf_loads / toks,
                pf_precision=(pf_used / pf_loads if pf_loads else None), admitted_per_token=admitted / toks,
                layer_hit=(1 - miss_by_layer / np.maximum(acc_by_layer, 1)).round(4).tolist(),
                step_miss={int(k): v for k, v in sorted(miss_by_step.items())} if cfg.get("cold") else None)


def run_opt(rounds, per, mode, total, nbatch, init):
    """Belady with bypass, per layer (uniform capacity) or one pool of `total` slots (global)."""
    sets_by_layer = [[] for _ in range(NL)]
    for r in rounds:
        s = to_sets(r)
        for l in range(NL):
            sets_by_layer[l].append(s[l])
    hit = acc = 0
    toks = sum(r["rows"] for r in rounds)
    hist = collections.Counter()
    if mode == "global":  # one pool; keys carry the layer, a round visits the layers in order
        order = [[(l << 8) | e for l in range(NL) for e in sets_by_layer[l][t]] for t in range(len(rounds))]
        groups = [(order, total, {(l << 8) | e for l in range(NL) for e in init[l]})]
        track_hist = False  # misses per (layer, round) are not tracked for the global pool
    else:
        groups = [([list(sets_by_layer[l][t]) for t in range(len(rounds))], per, init[l]) for l in range(NL)]
        track_hist = True
    for order, cap, start in groups:
        T = len(order)
        INF = 10**9
        occ = collections.defaultdict(list)
        for t in range(T):
            for k in order[t]:
                occ[k].append(t)
        ptr = collections.defaultdict(int)
        cache = {k: occ[k][0] if k in occ else INF for k in list(start)[:cap]}  # warm start with the offline profile
        for t in range(T):
            miss_here = 0
            for k in order[t]:
                ptr[k] += 1  # k is accessed at t: its next use is occ[k][ptr[k]]
            for k in order[t]:
                acc += 1
                nxt = occ[k][ptr[k]] if ptr[k] < len(occ[k]) else INF
                if k in cache:
                    hit += 1
                    cache[k] = nxt
                    continue
                miss_here += 1
                if len(cache) < cap:
                    cache[k] = nxt
                else:
                    victim = max(cache, key=cache.get)
                    if cache[victim] > nxt:
                        del cache[victim]
                        cache[k] = nxt
            if track_hist:
                hist[miss_here] += 1
    return dict(cfg=dict(policy="opt", budget=total * EXPERT_BYTES / GIB, n=nbatch, mode=mode), hit_rate=hit / acc, accesses=acc,
                miss_per_token=(acc - hit) / toks, tokens=toks, accesses_per_token=acc / toks, hist={int(k): v for k, v in sorted(hist.items())},
                p_layer_miss=(1 - hist.get(0, 0) / (len(rounds) * NL)) if track_hist else None, pf_loads_per_token=0.0, pf_precision=None,
                admitted_per_token=0.0, layer_hit=None, step_miss=None)


# ------------------------------------------------------------------ drivers
_DATA = None


def _init():
    global _DATA
    _DATA = load()


def _job(args):
    cfg, mode = args
    data = _DATA
    names = sorted(data)
    fl = folds_of(names)
    tot = None
    parts = []
    for f in range(FOLDS):
        test = fl[f]
        train = [n for g in range(FOLDS) if g != f for n in fl[g]]
        parts.append(run(data, test, train, cfg, seed=f))
    return merge(cfg, parts)


def merge(cfg, parts):
    toks = sum(p["tokens"] for p in parts)
    acc = sum(p["accesses"] for p in parts)
    hit = sum(p["hit_rate"] * p["accesses"] for p in parts)
    hist = collections.Counter()
    for p in parts:
        for k, v in p["hist"].items():
            hist[k] += v
    nr = sum(hist.values())
    lh = None
    if parts[0].get("layer_hit"):
        lh = np.average([p["layer_hit"] for p in parts], axis=0, weights=[p["accesses"] for p in parts]).round(4).tolist()
    pf = sum((p["pf_loads_per_token"] * p["tokens"]) for p in parts)
    used = sum((p["pf_precision"] or 0) * p["pf_loads_per_token"] * p["tokens"] for p in parts)
    sm = None
    if parts[0].get("step_miss"):
        sm = collections.defaultdict(lambda: [0, 0])
        for p in parts:
            for k, (m, rows) in p["step_miss"].items():
                sm[k][0] += m
                sm[k][1] += rows
    return dict(cfg=cfg, hit_rate=hit / acc, miss_per_token=(acc - hit) / toks, accesses_per_token=acc / toks, tokens=toks,
                p_layer_miss=(1 - hist.get(0, 0) / nr) if nr else None, hist={int(k): v for k, v in sorted(hist.items())},
                pf_loads_per_token=pf / toks, pf_precision=(used / pf if pf else None),
                admitted_per_token=sum(p["admitted_per_token"] * p["tokens"] for p in parts) / toks, layer_hit=lh,
                step_miss={int(k): v for k, v in sorted(sm.items())} if sm else None)


BUDGETS = [2, 3, 4, 5, 5.9, 8, 10, 12, 16, 20, 24]


def pool_map(jobs):
    with Pool(int(os.environ.get("SIM_PROCS", "4")), initializer=_init) as p:
        return p.map(_job, jobs, chunksize=1)


def fmt_row(label, vals, w=7):
    return f"{label:34s}" + "".join(f"{v:>{w}}" for v in vals)


def table(title, results, rowkeys, budgets, key="hit_rate", pct=True):
    print(f"\n{title}")
    print(fmt_row("policy \\ expert budget GiB", [f"{b:g}" for b in budgets]))
    for rk, label in rowkeys:
        vals = []
        for b in budgets:
            r = results.get((rk, b))
            vals.append("" if r is None else (f"{100 * r[key]:.1f}" if pct else f"{r[key]:.1f}"))
        print(fmt_row(label, vals))


def cmd_sim(ns):
    out = {}
    pols = [("static", "static top-freq (uniform/layer)", "uniform", {}), ("static", "static top-freq (global alloc)", "global", {}),
            ("static_in", "static, profiled on the test set", "uniform", {}),
            ("lru", "LRU (uniform/layer)", "uniform", {}), ("lru", "LRU (one global pool)", "global", {}),
            ("lfu", "LFU history (uniform/layer)", "uniform", {}),
            ("fr", "freq+recency hl128 (uniform)", "uniform", {"halflife": 128}),
            ("fr", "freq+recency hl128 + profile prior", "uniform", {"halflife": 128, "prior": 64}),
            ("hybrid", "hybrid half static + half LRU", "uniform", {}),
            ("opt", "Belady OPT (bypass, uniform/layer)", "uniform", {})]
    for n in ns:
        jobs, keys = [], []
        for pol, label, mode, extra in pols:
            for b in BUDGETS:
                jobs.append((dict(policy=pol, budget=b, n=n, mode=mode, **extra), None))
                keys.append((label, b))
        res = pool_map(jobs)
        results = dict(zip(keys, res))
        out[f"sim_n{n}"] = res
        rows = [(l, l) for _, l, _, _ in pols]
        table(f"### decode hit rate %, {n} sequence(s) a round, no prefetch, admit on miss (static: no admission)", results, rows, BUDGETS)
        table(f"### expert loads from the host a token (of {40 * 8 * n} row-expert accesses; misses a token), {n} sequence(s)", results,
              rows, BUDGETS, key="miss_per_token", pct=False)
        print("\nP(a layer-round has at least one miss), %  (the global-pool rows do not track it):")
        print(fmt_row("policy \\ expert budget GiB", [f"{b:g}" for b in BUDGETS]))
        for _, l, _, _ in pols:
            print(fmt_row(l, ["" if results[(l, b)]["p_layer_miss"] is None else f"{100 * results[(l, b)]['p_layer_miss']:.1f}" for b in BUDGETS]))
    return out


def cmd_prefetch():
    out = []
    jobs = []
    base = [("lru", "uniform", {}), ("hybrid", "uniform", {})]  # fr's admission filter rejects prefetched experts, so it is not listed
    pfs = [None, "prev", "A0", "A1", "A2", "B1", "B2", "prev+A2"]
    for b in (4, 5.9):
        for pol, mode, extra in base:
            for pf in pfs:
                for cap in ((None, 18) if pf else (None,)):
                    jobs.append((dict(policy=pol, budget=b, n=1, mode=mode, prefetch=pf, cap=cap, **extra), None))
    res = pool_map(jobs)
    print("\n### prefetch (1 sequence, decode): hit rate %, loads a token over the link by the prefetcher, how many of them were used")
    print(f"{'policy':8s}{'GiB':>5s}{'prefetch':>10s}{'cap/tok':>8s}{'hit %':>8s}{'miss/tok':>9s}{'pf loads/tok':>13s}{'pf precision':>13s}{'layers w/ miss %':>17s}")
    for r in res:
        c = r["cfg"]
        print(f"{c['policy']:8s}{c['budget']:>5g}{str(c['prefetch']):>10s}{str(c['cap']):>8s}{100 * r['hit_rate']:>8.1f}{r['miss_per_token']:>9.1f}"
              f"{r['pf_loads_per_token']:>13.1f}{('' if r['pf_precision'] is None else f'{100 * r['pf_precision']:.0f}%'):>13s}{100 * r['p_layer_miss']:>17.1f}")
        out.append(r)
    return out


def cmd_admit():
    """Admission throttle: on the Metal-miss path a missed expert reaches the card only by a DMA copy of 1.69 MiB."""
    jobs = []
    for b in (4, 5.9):
        for pol, extra in (("lru", {}), ("hybrid", {}), ("fr", {"halflife": 128, "prior": 64})):
            for p in (1.0, 0.5, 0.25, 0.1, 0.03, 0.0):
                jobs.append((dict(policy=pol, budget=b, n=1, mode="uniform", admit_p=p, **extra), None))
    res = pool_map(jobs)
    print("\n### admission throttle (1 sequence, decode, no prefetch): a missed expert is copied into the cache with probability p")
    print(f"{'policy':8s}{'GiB':>5s}{'p':>6s}{'hit %':>8s}{'miss/tok':>9s}{'copies/tok':>11s}{'layers w/ miss %':>17s}")
    for r in res:
        c = r["cfg"]
        print(f"{c['policy']:8s}{c['budget']:>5g}{c['admit_p']:>6g}{100 * r['hit_rate']:>8.1f}{r['miss_per_token']:>9.1f}"
              f"{r['admitted_per_token']:>11.1f}{100 * r['p_layer_miss']:>17.1f}")
    return res


def cmd_generalise():
    data = load()
    names = sorted(data)
    cats = sorted({data[n]["cat"] for n in names})
    out = {}
    print("\n### static top-frequency cache, 4 GiB (60 slots a layer) and 5.9 GiB (89): decode hit rate % on a held-out set")
    print(f"{'train -> test':44s}{'4 GiB':>8s}{'5.9 GiB':>9s}")

    def hit(train, test, budget, mode="uniform"):
        counts = profile(data, train)
        sets = static_sets(counts, slots(budget), mode)
        acc = h = 0
        for n in test:
            ids = data[n]["ids"]
            for l in range(NL):
                mask = np.zeros(NE, dtype=bool)
                mask[list(sets[l])] = True
                hh = mask[ids[:, l, :]]
                h += hh.sum(); acc += hh.size
        return h / acc

    def line(label, train, test):
        a, b = hit(train, test, 4), hit(train, test, 5.9)
        print(f"{label:44s}{100 * a:>8.1f}{100 * b:>9.1f}")
        out[label] = (a, b)

    line("same prompts (in-sample)", names, names)
    fl = folds_of(names)
    accs = [], []
    for f in range(FOLDS):
        train = [n for g in range(FOLDS) if g != f for n in fl[g]]
        accs[0].append((hit(train, fl[f], 4), sum(len(data[n]["ids"]) for n in fl[f])))
        accs[1].append((hit(train, fl[f], 5.9), sum(len(data[n]["ids"]) for n in fl[f])))
    w = [x[1] for x in accs[0]]
    a, b = np.average([x[0] for x in accs[0]], weights=w), np.average([x[0] for x in accs[1]], weights=w)
    print(f"{'4-fold cross-validation over prompts':44s}{100 * a:>8.1f}{100 * b:>9.1f}")
    out["cv4"] = (a, b)
    for c in cats:
        te = [n for n in names if data[n]["cat"] == c]
        tr = [n for n in names if data[n]["cat"] != c]
        line(f"leave out {c} ({len(te)} prompts)", tr, te)
    for c in cats:
        te = [n for n in names if data[n]["cat"] == c]
        line(f"train and test on {c} (in-sample)", te, te)
    rng = random.Random(1)
    for k in (1, 2, 4, 8, 16, 24):
        res = []
        for rep in range(8):
            tr = rng.sample(names, min(k, len(names) - 6))
            te = [n for n in names if n not in tr]
            res.append((hit(tr, te, 4), hit(tr, te, 5.9)))
        a, b = np.mean(res, axis=0)
        print(f"{f'train on {k} random prompts, test on the rest':44s}{100 * a:>8.1f}{100 * b:>9.1f}")
        out[f"train{k}"] = (a, b)
    # profile similarity between categories: share of one category's top-60 per layer found in another's top-60
    print("\ntop-60 overlap per layer between category profiles (share of A's top-60 in B's top-60), mean over layers:")
    prof = {c: profile(data, [n for n in names if data[n]["cat"] == c]) for c in cats}
    print(f"{'':8s}" + "".join(f"{c:>8s}" for c in cats))
    for a in cats:
        row = []
        for b in cats:
            ov = np.mean([len(set(np.argsort(-prof[a][l])[:60]) & set(np.argsort(-prof[b][l])[:60])) / 60 for l in range(NL)])
            row.append(ov)
        print(f"{a:8s}" + "".join(f"{100 * v:>7.0f}%" for v in row))
    return out


def cmd_long():
    """Long generations (traces_long, up to 1,600 decode tokens): does the hit rate hold after the first few hundred tokens?"""
    short, long_ = load(), load(os.path.join(HERE, "traces_long"))
    train = sorted(n for n in short if n not in long_)  # the long prompts' own 288-token prefixes are not in the profile
    counts = profile(short, train)
    print(f"\n### long generations: {len(long_)} prompts, {sum(len(v['ids']) for v in long_.values())} decode tokens; caches emptied at each prompt start; 5.9 GiB (89 slots a layer)")
    per = slots(5.9) // NL
    stat = [set(np.argsort(-counts[l], kind="stable")[:per].tolist()) for l in range(NL)]
    wins = [(0, 100), (100, 300), (300, 600), (600, 1000), (1000, 1600)]
    acc = {p: np.zeros((len(wins), 2)) for p in ("lru", "static (profile from the other 28 short prompts)", "hybrid")}
    for name, d in long_.items():
        ids = d["ids"]
        for pol in acc:
            for l in range(NL):
                if pol.startswith("static"):
                    c = Cache(per, "static", pinned=stat[l])
                else:
                    pin = set(np.argsort(-counts[l], kind="stable")[: per // 2].tolist()) if pol == "hybrid" else ()
                    c = Cache(per, pol, pinned=pin)
                for t in range(len(ids)):
                    S = ids[t, l].tolist()
                    miss = [e for e in S if not c.contains(e)]
                    for e in S:
                        c.touch(e)
                    for e in miss:
                        c.admit(e, protect=set(S))
                    for wi, (lo, hi) in enumerate(wins):
                        if lo <= t < hi:
                            acc[pol][wi] += (8 - len(miss), 8)
    print(f"{'policy':44s}" + "".join(f"{f'{lo}-{hi}':>10s}" for lo, hi in wins) + "   hit %")
    for pol, a in acc.items():
        print(f"{pol:44s}" + "".join(f"{100 * x / y:>10.1f}" for x, y in a))
    ov = [np.mean([len(set(d['ids'][t, l].tolist()) & set(d['ids'][t - 1, l].tolist())) / 8 for t in range(1, len(d['ids'])) for l in range(NL)]) for d in long_.values()]
    print(f"previous-token overlap on the long generations: {np.mean(ov):.3f}")


def cmd_locality():
    data = load()
    names = sorted(data)
    print(f"traces: {len(names)} prompts, {sum(len(data[n]['ids']) for n in names)} decode tokens, "
          f"{sum(len(data[n]['pre_ids']) for n in names)} prefill tokens")
    # skew: share of accesses to the top-k experts of a layer
    tot = profile(data, names)
    print("\n### skew of the router (all decode tokens): share of a layer's accesses served by its k most popular experts of 256")
    print(f"{'k':>6s}{'mean over layers':>18s}{'min layer':>11s}{'max layer':>11s}   (uniform routing would give k/256)")
    for k in (8, 16, 32, 60, 89, 128):
        sh = np.array([np.sort(tot[l])[::-1][:k].sum() / tot[l].sum() for l in range(NL)])
        print(f"{k:>6d}{100 * sh.mean():>17.1f}%{100 * sh.min():>10.1f}%{100 * sh.max():>10.1f}%   ({100 * k / 256:.1f}%)")
    ent = np.array([-(p := tot[l] / tot[l].sum())[p > 0] @ np.log2(p[p > 0]) / 8 for l in range(NL)])
    print(f"normalized entropy (1 = uniform): mean {ent.mean():.3f}, min {ent.min():.3f} (layer {ent.argmin()}), max {ent.max():.3f} (layer {ent.argmax()})")
    nz = [int((tot[l] > 0).sum()) for l in range(NL)]
    print(f"distinct experts ever used in a layer over the whole trace: min {min(nz)}, mean {np.mean(nz):.0f}, max {max(nz)} of 256")
    print("layers by hot-8 share (first 3 / last 3 of the sorted list): " + ", ".join(
        f"L{l}:{100 * np.sort(tot[l])[::-1][:8].sum() / tot[l].sum():.0f}%" for l in np.argsort([np.sort(tot[l])[::-1][:8].sum() / tot[l].sum() for l in range(NL)])[[0, 1, 2, -3, -2, -1]]))
    # temporal locality
    print("\n### temporal locality (decode, one sequence)")
    ov = []
    reuse = collections.Counter()
    first = 0
    nacc = 0
    ov_exp = []
    for n in names:
        ids = data[n]["ids"]
        T = len(ids)
        for l in range(NL):
            lastseen = {}
            prev = None
            for t in range(T):
                s = set(ids[t, l].tolist())
                if prev is not None:
                    ov.append(len(s & prev) / 8)
                for e in s:
                    nacc += 1
                    if e in lastseen:
                        d = t - lastseen[e]
                        reuse[min(d, 10**4)] += 1
                    else:
                        first += 1
                    lastseen[e] = t
                prev = s
    print(f"overlap of a layer's experts with the previous token's, mean: {np.mean(ov):.3f} (random picks of 8 from 256 would overlap {8 / 256:.3f})")
    tot_re = sum(reuse.values())
    cum = 0
    rows = []
    for lim in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        c = sum(v for d, v in reuse.items() if d <= lim)
        rows.append(f"<= {lim}: {100 * c / nacc:.1f}%")
    print(f"share of accesses whose expert was used in that layer within the last N tokens (first use of an expert in a prompt: {100 * first / nacc:.1f}%):")
    print("   " + "   ".join(rows))
    # warm-up: per-prompt-prefix profile fidelity is covered by the cold curve in `sim`
    # prefill union
    print("\n### prefill: distinct experts a layer touches for a chunk of S prompt tokens (all prompts with >= S tokens), mean over layers")
    sizes = (1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1982)
    for S in sizes:
        v = []
        for n in names:
            p = data[n]["pre_ids"]
            if len(p) >= S:
                v.append(np.mean([len(np.unique(p[:S, l, :])) for l in range(NL)]))
        if v:
            print(f"  S={S:5d}: {np.mean(v):6.1f} of 256 ({100 * np.mean(v) / 256:5.1f}%), {len(v)} prompts")
    # how well does the decode profile cover a prefill chunk's accesses
    cov = []
    for n in names:
        p = data[n]["pre_ids"]
        if len(p) < 128:
            continue
        cov.append(None)
    # expert reuse in decode across consecutive tokens vs batch union
    print("\n### union over a batch: distinct experts a layer reads in one round, N sequences (different prompts, lockstep), mean over layers")
    rng = random.Random(0)
    for N in (1, 2, 3, 4, 8):
        u = []
        for _ in range(300):
            grp = rng.sample(names, N)
            t = rng.randrange(0, 250)
            ids = np.stack([data[n]["ids"][min(t, len(data[n]["ids"]) - 1)] for n in grp])
            u.append(np.mean([len(np.unique(ids[:, l, :])) for l in range(NL)]))
        print(f"  N={N}: {np.mean(u):5.2f} experts a layer ({np.mean(u) * 40 * EXPERT_BYTES / 2**20:.0f} MiB a round; {N * 8} row-experts)")
    # lookahead accuracy
    print("\n### lookahead: share of the true top-8 of layer t found in the guessed top-8 (decode tokens, layers >= 2)")
    names_k = ["A0 own router, residual entering the layer", "A1 residual entering layer t-1", "A2 residual entering layer t-2",
               "B1 residual after layer t-1's mixer", "B2 residual after layer t-2's mixer"]
    for k, nm in enumerate(names_k):
        rec = []
        for n in names:
            ids, pr = data[n]["ids"], data[n]["preds"]
            for l in range(2, NL):
                a = ids[:, l, :][:, :, None] == pr[:, l, k, :][:, None, :]
                rec.append(a.any(axis=2).mean())
        print(f"  {nm:48s} recall {100 * np.mean(rec):5.1f}%  (precision = recall: both sets have 8)")
    # expected recall if one guessed the previous token's set / the static hot set
    rec = []
    for n in names:
        ids = data[n]["ids"]
        for l in range(2, NL):
            a = ids[1:, l, :][:, :, None] == ids[:-1, l, :][:, None, :]
            rec.append(a.any(axis=2).mean())
    print(f"  {'previous token, same layer':48s} recall {100 * np.mean(rec):5.1f}%")
    cov8 = np.mean([np.sort(tot[l])[::-1][:8].sum() / tot[l].sum() for l in range(NL)])
    print(f"  {'static hot-8 of the layer':48s} recall {100 * cov8:5.1f}%")


def cmd_all():
    allres = {}
    cmd_locality()
    allres["generalise"] = {k: list(v) for k, v in cmd_generalise().items()}
    allres["sim"] = cmd_sim([1, 2, 4])
    allres["prefetch"] = cmd_prefetch()
    allres["admit"] = cmd_admit()
    cold = pool_map([(dict(policy="lru", budget=5.9, n=1, mode="uniform", cold=True), None), (dict(policy="hybrid", budget=5.9, n=1, mode="uniform", cold=True), None),
                     (dict(policy="static", budget=5.9, n=1, mode="uniform", cold=True), None)])
    print("\n### cold start: LRU, hybrid and static caches emptied at each prompt (5.9 GiB), misses a token by decode position")
    print(f"{'position':>14s}" + "".join(f"{c['cfg']['policy']:>10s}" for c in cold))
    for lo, hi in ((0, 8), (8, 32), (32, 64), (64, 128), (128, 256), (256, 512)):
        row = []
        for c in cold:
            m = sum(v[0] for k, v in c["step_miss"].items() if lo <= k < hi)
            rows = sum(v[1] for k, v in c["step_miss"].items() if lo <= k < hi)
            row.append(m / rows)
        print(f"{f'{lo}-{hi}':>14s}" + "".join(f"{v:>10.1f}" for v in row))
    allres["cold"] = cold
    cmd_long()
    json.dump(allres, open(os.path.join(HERE, "cache_sim_results.json"), "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "all":
        cmd_all()
    elif cmd == "locality":
        cmd_locality()
    elif cmd == "sim":
        cmd_sim([int(x) for x in sys.argv[2:]] or [1, 2, 4])
    elif cmd == "admit":
        cmd_admit()
    elif cmd == "prefetch":
        cmd_prefetch()
    elif cmd == "generalise":
        cmd_generalise()
    elif cmd == "long":
        cmd_long()
