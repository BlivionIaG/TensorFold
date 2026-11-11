#!/usr/bin/env python3
"""Offline replay of real Qwen3.6-35B-A3B decode routing traces against expert-cache policies (numpy only, CPU only).

Model: one global pool of CAP lines (default 2876 = 4.74 GiB of 1,769,472 B experts) shared by 40 layers x 256 experts (key = layer*256+expert).
Every (token, layer) call asks for the layer's top-8 (lockstep mode: the union of the sequences' picks). Residents are hits and are touched
first, then misses are admitted one by one ("admit on miss" = free: the miss read writes through into the chosen line). A line touched or
admitted in the current call is never the victim (the production `stageAt` rule, expert_offload.zig). Policies may also refuse to admit
(TinyLFU, ARC never does). The cache and all policy state are emptied at each prompt (or lockstep group) start.

usage: expert_cache_sim.py [TRACES_DIR] [--long DIR] [--cap N] [--jobs N] [--only SUBSTR]
  TRACES_DIR holds meta.json + *.npz (ids (T,40,8), kind: 1 = decode, 0 = prefill, as written by routing_trace.py); the default is the study
  worktree's traces/. --long defaults to <TRACES_DIR>/../traces_long (set to "none" to skip).
Datasets: short = every prompt of TRACES_DIR on its own; long = long generations; lock4 = short prompts, 4 at a time in lockstep, per-layer union.
"""
import collections
import glob
import heapq
import json
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np

NL, NE, TOPK = 40, 256, 8
NK = NL * NE
CAP_DEFAULT = 2876
DEFAULT_TRACES = os.path.expanduser("~/Projects/Infra/tf-study-moe-cache/docs/proposals/metal-tb4-split-moe-probe/traces")
BUCKETS = [(0, 8), (8, 32), (32, 128), (128, 512), (512, 10**9)]
BUCKET_NAMES = ["0-8", "8-32", "32-128", "128-512", "512+"]
OD = collections.OrderedDict


def load(directory):
    """name -> dict(ids (T,40,8) decode picks, pre (P,40,8) prefill picks, cat)."""
    meta = {m["name"]: m for m in json.load(open(os.path.join(directory, "meta.json")))}
    out = {}
    for f in sorted(glob.glob(os.path.join(directory, "*.npz"))):
        name = os.path.basename(f)[:-4]
        d = np.load(f)
        dec = d["kind"] == 1
        out[name] = dict(ids=d["ids"][dec], pre=d["ids"][~dec], cat=meta[name]["cat"])
    return out


# ------------------------------------------------------------------ frequency estimator
class Freq:
    """Exact counters over the 10,240 keys, halved every W accesses; optional doorkeeper (first sighting only sets a bit)."""

    def __init__(self, W, door):
        self.c = [0] * NK
        self.W, self.n = W, 0
        self.door = set() if door else None

    def touch(self, keys):
        c, door = self.c, self.door
        if door is None:
            for k in keys:
                c[k] += 1
        else:
            for k in keys:
                if k in door:
                    c[k] += 1
                else:
                    door.add(k)

    def est(self, k):
        return self.c[k] + (1 if self.door is not None and k in self.door else 0)

    def tick(self, n):
        self.n += n
        if self.n >= self.W:
            self.n = 0
            self.c = [x >> 1 for x in self.c]
            if self.door is not None:
                self.door.clear()
            return True
        return False


def first_free(od, kset):
    """oldest entry of od not requested in this call, or None"""
    for v in od:
        if v not in kset:
            return v
    return None


# ------------------------------------------------------------------ policies; step(layer, keys) -> number of misses
class LRU:
    def __init__(self, cap):
        self.cap, self.od, self.adm, self.evi = cap, OD(), 0, 0

    def preload(self, keys_oldest_first):
        for k in keys_oldest_first:
            if len(self.od) < self.cap:
                self.od[k] = None

    def step(self, layer, keys):
        od, miss = self.od, []
        for k in keys:
            if k in od:
                od.move_to_end(k)
            else:
                miss.append(k)
        if miss:
            kset = set(keys)
            for k in miss:
                if len(od) >= self.cap:
                    v = first_free(od, kset)
                    del od[v]
                    self.evi += 1
                od[k] = None
                self.adm += 1
        return len(miss)


class LRUPerLayer:
    """equal split: CAP//40 lines per layer, the remainder one extra line to the first layers"""

    def __init__(self, cap):
        base, rem = divmod(cap, NL)
        self.caps = [base + (1 if l < rem else 0) for l in range(NL)]
        self.ods = [OD() for _ in range(NL)]
        self.adm = self.evi = 0

    def step(self, layer, keys):
        od, cap, miss = self.ods[layer], self.caps[layer], []
        for k in keys:
            if k in od:
                od.move_to_end(k)
            else:
                miss.append(k)
        if miss:
            kset = set(keys)
            for k in miss:
                if len(od) >= cap:
                    del od[first_free(od, kset)]
                    self.evi += 1
                od[k] = None
                self.adm += 1
        return len(miss)


class SLRU:
    """probation + protected; a probation hit is promoted, a protected overflow is demoted to probation's MRU end"""

    def __init__(self, cap, pfrac):
        self.cap, self.pcap = cap, int(cap * pfrac)
        self.prob, self.prot, self.adm, self.evi = OD(), OD(), 0, 0

    def step(self, layer, keys):
        prob, prot, miss = self.prob, self.prot, []
        for k in keys:
            if k in prot:
                prot.move_to_end(k)
            elif k in prob:
                del prob[k]
                prot[k] = None
                if len(prot) > self.pcap:
                    d, _ = prot.popitem(last=False)
                    prob[d] = None
            else:
                miss.append(k)
        if miss:
            kset = set(keys)
            for k in miss:
                if len(prob) + len(prot) >= self.cap:
                    v = first_free(prob, kset)
                    if v is not None:
                        del prob[v]
                    else:
                        del prot[first_free(prot, kset)]
                    self.evi += 1
                prob[k] = None
                self.adm += 1
        return len(miss)


class TwoQ:
    """2Q: FIFO A1in (kin of the cap), LRU Am, ghost list A1out of evicted A1in keys (ghost_frac of the cap); a ghost hit goes to Am"""

    def __init__(self, cap, kin, ghost_frac=0.5):
        self.cap, self.kin, self.gcap = cap, max(1, int(cap * kin)), int(cap * ghost_frac)
        self.a1, self.am, self.out, self.adm, self.evi = OD(), OD(), OD(), 0, 0

    def step(self, layer, keys):
        a1, am, out, miss = self.a1, self.am, self.out, []
        for k in keys:
            if k in am:
                am.move_to_end(k)
            elif k not in a1:
                miss.append(k)
        if miss:
            kset = set(keys)
            for k in miss:
                if len(a1) + len(am) >= self.cap:
                    v = first_free(a1, kset) if (len(a1) > self.kin or not am) else None
                    if v is not None:
                        del a1[v]
                        out[v] = None
                        if len(out) > self.gcap:
                            out.popitem(last=False)
                    else:
                        v = first_free(am, kset)
                        if v is None:
                            v = first_free(a1, kset)
                            del a1[v]
                        else:
                            del am[v]
                    self.evi += 1
                if k in out:
                    del out[k]
                    am[k] = None
                else:
                    a1[k] = None
                self.adm += 1
        return len(miss)


class TinyLFU:
    """LRU victim, but a miss is admitted into a full cache only if freq(new) > freq(victim) (>= with ge). Frequency counts every access."""

    def __init__(self, cap, W, door, ge=False):
        self.cap, self.f, self.ge = cap, Freq(W, door), ge
        self.od, self.adm, self.evi = OD(), 0, 0

    def step(self, layer, keys):
        od, f, miss = self.od, self.f, []
        f.touch(keys)
        for k in keys:
            if k in od:
                od.move_to_end(k)
            else:
                miss.append(k)
        if miss:
            kset = set(keys)
            for k in miss:
                if len(od) >= self.cap:
                    v = first_free(od, kset)
                    a, b = f.est(k), f.est(v)
                    if not (a >= b if self.ge else a > b):
                        continue
                    del od[v]
                    self.evi += 1
                od[k] = None
                self.adm += 1
        f.tick(len(keys))
        return len(miss)


class WTinyLFU:
    """window LRU (wfrac of the cap) in front of an LRU main; the window's overflow candidate is admitted to a full main only if it beats main's LRU"""

    def __init__(self, cap, wfrac, W, door):
        self.wcap = max(1, int(cap * wfrac))
        self.mcap = cap - self.wcap
        self.f = Freq(W, door)
        self.win, self.main, self.adm, self.evi = OD(), OD(), 0, 0

    def step(self, layer, keys):
        win, main, f, miss = self.win, self.main, self.f, []
        f.touch(keys)
        for k in keys:
            if k in main:
                main.move_to_end(k)
            elif k in win:
                win.move_to_end(k)
            else:
                miss.append(k)
        for k in miss:
            win[k] = None
            self.adm += 1
            if len(win) > self.wcap:
                cand, _ = win.popitem(last=False)
                if len(main) < self.mcap:
                    main[cand] = None
                else:
                    v = next(iter(main))
                    if f.est(cand) > f.est(v):
                        del main[v]
                        main[cand] = None
                    self.evi += 1  # either the old main line or the candidate leaves the cache
        f.tick(len(keys))
        return len(miss)


class LFU:
    """aged LFU (counts halved every W accesses), LRU tie-break; hist keeps counts of non-resident keys, else a key's count restarts at eviction"""

    def __init__(self, cap, W, hist):
        self.cap, self.W, self.hist, self.n = cap, W, hist, 0
        self.c = [0] * NK
        self.rec, self.b, self.heap, self.inheap = OD(), {}, [], set()
        self.adm = self.evi = 0

    def _add(self, k, cnt):
        d = self.b.get(cnt)
        if d is None:
            d = self.b[cnt] = OD()
            if cnt not in self.inheap:
                self.inheap.add(cnt)
                heapq.heappush(self.heap, cnt)
        d[k] = None

    def _del(self, k, cnt):
        d = self.b[cnt]
        del d[k]
        if not d:
            del self.b[cnt]

    def _victim(self, kset):
        heap, b = self.heap, self.b
        while heap[0] not in b:
            self.inheap.discard(heapq.heappop(heap))
        for k in b[heap[0]]:
            if k not in kset:
                return k
        for lvl in sorted(b):
            for k in b[lvl]:
                if k not in kset:
                    return k

    def step(self, layer, keys):
        rec, c, miss = self.rec, self.c, []
        for k in keys:
            if k in rec:
                old = c[k]
                c[k] = old + 1
                self._del(k, old)
                self._add(k, old + 1)
                rec.move_to_end(k)
            else:
                c[k] += 1
                miss.append(k)
        if miss:
            kset = set(keys)
            for k in miss:
                if len(rec) >= self.cap:
                    v = self._victim(kset)
                    del rec[v]
                    self._del(v, c[v])
                    if not self.hist:
                        c[v] = 0
                    self.evi += 1
                rec[k] = None
                self._add(k, c[k])
                self.adm += 1
        self.n += len(keys)
        if self.n >= self.W:
            self.n = 0
            self.c = c = [x >> 1 for x in c]
            self.b, self.heap, self.inheap = {}, [], set()
            for k in rec:
                self._add(k, c[k])
        return len(miss)


class LRUFreq:
    """evict, among the K oldest evictable lines, the one with the lowest (aged) frequency; ties go to the older line"""

    def __init__(self, cap, K, W):
        self.cap, self.K, self.f = cap, K, Freq(W, False)
        self.od, self.adm, self.evi = OD(), 0, 0

    def step(self, layer, keys):
        od, f, miss = self.od, self.f, []
        f.touch(keys)
        for k in keys:
            if k in od:
                od.move_to_end(k)
            else:
                miss.append(k)
        if miss:
            kset, c, K = set(keys), f.c, self.K
            for k in miss:
                if len(od) >= self.cap:
                    best, bc, n = None, 0, 0
                    for v in od:
                        if v in kset:
                            continue
                        if best is None or c[v] < bc:
                            best, bc = v, c[v]
                        n += 1
                        if n >= K:
                            break
                    del od[best]
                    self.evi += 1
                od[k] = None
                self.adm += 1
        f.tick(len(keys))
        return len(miss)


class ARC:
    """Adaptive Replacement Cache, one global pool; victims skip lines requested in the current call"""

    def __init__(self, cap):
        self.c, self.p = cap, 0
        self.t1, self.t2, self.b1, self.b2 = OD(), OD(), OD(), OD()
        self.adm = self.evi = 0

    def _replace(self, in_b2, kset):
        t1, t2 = self.t1, self.t2
        if len(t1) + len(t2) < self.c:
            return
        prefer1 = len(t1) > 0 and (len(t1) > self.p or (in_b2 and len(t1) == self.p))
        for first in ((t1, self.b1), (t2, self.b2)) if prefer1 else ((t2, self.b2), (t1, self.b1)):
            v = first_free(first[0], kset)
            if v is not None:
                del first[0][v]
                first[1][v] = None
                self.evi += 1
                return

    def step(self, layer, keys):
        t1, t2, b1, b2, c = self.t1, self.t2, self.b1, self.b2, self.c
        miss = []
        for k in keys:
            if k in t1:
                del t1[k]
                t2[k] = None
            elif k in t2:
                t2.move_to_end(k)
            else:
                miss.append(k)
        if miss:
            kset = set(keys)
            for k in miss:
                if k in b1:
                    self.p = min(c, self.p + max(len(b2) // len(b1), 1))
                    self._replace(False, kset)
                    del b1[k]
                    t2[k] = None
                elif k in b2:
                    self.p = max(0, self.p - max(len(b1) // len(b2), 1))
                    self._replace(True, kset)
                    del b2[k]
                    t2[k] = None
                else:
                    l1 = len(t1) + len(b1)
                    if l1 >= c:
                        if len(t1) < c:
                            b1.popitem(last=False)
                            self._replace(False, kset)
                        else:
                            v = first_free(t1, kset)
                            del t1[v]
                            self.evi += 1
                    else:
                        total = l1 + len(t2) + len(b2)
                        if total >= c:
                            if total >= 2 * c:
                                b2.popitem(last=False)
                            self._replace(False, kset)
                    t1[k] = None
                self.adm += 1
                while len(b1) + len(b2) > 2 * c:  # ghost lists bounded (cap hits only through protected skips)
                    (b1 if len(b1) > len(b2) else b2).popitem(last=False)
        return len(miss)


# ------------------------------------------------------------------ configs
def build_configs(cap):
    """(label, group, factory(cap) -> policy, init mode). init: cold | seed | pf | pf+seed | pfreplay (single-sequence datasets only)"""
    C = []

    def add(label, group, fac, init="cold"):
        C.append((label, group, fac, init))

    add("LRU global (baseline)", "1 LRU", lambda: LRU(cap))
    add("LRU per layer, equal split", "2 split", lambda: LRUPerLayer(cap))
    for pf in (0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9):
        add(f"SLRU protected {int(pf * 100)}%", "3 SLRU/2Q", lambda pf=pf: SLRU(cap, pf))
    for kin in (0.1, 0.25, 0.5):
        add(f"2Q A1in {int(kin * 100)}% ghost 50%", "3 SLRU/2Q", lambda kin=kin: TwoQ(cap, kin))
    for W in (2_500, 5_000, 10_000, 40_000, 160_000, 640_000):
        for door in (False, True):
            add(f"TinyLFU W={W / 1000:g}k door={'y' if door else 'n'}", "4 TinyLFU", lambda W=W, door=door: TinyLFU(cap, W, door))
    for door in (False, True):
        add(f"TinyLFU W=40k door={'y' if door else 'n'} admit if >=", "4 TinyLFU", lambda door=door: TinyLFU(cap, 40_000, door, ge=True))
    for wf in (0.01, 0.05, 0.2):
        add(f"W-TinyLFU window {int(wf * 100)}% W=160k door=y", "4 TinyLFU", lambda wf=wf: WTinyLFU(cap, wf, 160_000, True))
    for W in (1_250, 2_500, 5_000, 10_000, 20_000, 40_000, 160_000, 640_000):
        add(f"LFU aged W={W / 1000:g}k (history kept)", "5 LFU", lambda W=W: LFU(cap, W, True))
    for W in (40_000, 160_000):
        add(f"LFU aged W={W / 1000:g}k (count restarts on evict)", "5 LFU", lambda W=W: LFU(cap, W, False))
    for W in (5_000, 10_000, 40_000, 160_000):
        for K in (2, 4, 8, 16, 32):
            add(f"LRU, evict lowest-freq of K={K} oldest, W={W / 1000:g}k", "6 LRU+freq", lambda K=K, W=W: LRUFreq(cap, K, W))
    add("ARC global", "7 ARC", lambda: ARC(cap))
    add("LRU, seeded with other categories' hottest keys", "8 seed", lambda: LRU(cap), "seed")
    add("LRU, warm from own prefill top counts", "9 warm", lambda: LRU(cap), "pf")
    add("LRU, own prefill top counts + others' hottest fill", "9 warm", lambda: LRU(cap), "pf+seed")
    add("LRU, prefill replayed token by token through LRU", "9 warm", lambda: LRU(cap), "pfreplay")
    return C


# ------------------------------------------------------------------ replay
G = {}  # shared with forked workers: configs, datasets, seeds


def keyify(a):
    """(T,40,8) expert ids -> nested lists of global keys, [t][layer] -> list of 8"""
    return (a.astype(np.int32) + (np.arange(NL, dtype=np.int32) * NE)[None, :, None]).tolist()


def stable_seed(name):
    return sum(ord(ch) * (i + 1) for i, ch in enumerate(name))


def prefill_keys(pre, cap, name):
    """the prompt's most frequently routed keys (global top `cap`, ties random), hottest first; keys never routed are not listed"""
    cnt = np.bincount(np.asarray(keyify(pre)).reshape(-1), minlength=NK)
    rnd = np.random.RandomState(stable_seed(name)).random_sample(NK)
    order = np.lexsort((rnd, -cnt))
    order = order[cnt[order] > 0][:cap]
    return order.tolist()


def run_single(job):
    ci, ds, name = job
    label, group, fac, init = G["configs"][ci]
    d = G["data"][ds][name]
    pol = fac()
    cap = G["cap"]
    if init == "seed":
        pol.preload(G["seed"][d["cat"]][:cap][::-1])
    elif init in ("pf", "pf+seed"):
        pk = prefill_keys(d["pre"], cap, name)
        if init == "pf":
            pol.preload(pk[::-1])
        else:
            have = set(pk)
            fill = [k for k in G["seed"][d["cat"]] if k not in have][: max(0, cap - len(pk))]
            pol.preload(fill[::-1] + pk[::-1])
    elif init == "pfreplay":
        for row in keyify(d["pre"]):
            for l in range(NL):
                pol.step(l, row[l])
    pol.adm = pol.evi = 0
    K = keyify(d["ids"])
    T = len(K)
    miss = np.zeros(T, dtype=np.int32)
    step = pol.step
    for t in range(T):
        row, m = K[t], 0
        for l in range(NL):
            m += step(l, row[l])
        miss[t] = m
    return ci, ds, name, T * NL * TOPK, pol.adm, pol.evi, miss


def run_lock(job):
    ci, ds, gi = job
    label, group, fac, init = G["configs"][ci]
    names = G["groups"][gi]
    pol = fac()
    Ks = [keyify(G["data"]["short"][n]["ids"]) for n in names]
    T = max(len(k) for k in Ks)
    acc = rows = misses = 0
    step = pol.step
    for t in range(T):
        live = [k[t] for k in Ks if t < len(k)]
        rows += len(live)
        for l in range(NL):
            keys = sorted({x for r in live for x in r[l]}) if len(live) > 1 else live[0][l]
            acc += len(keys)
            misses += step(l, keys)
    return ci, ds, gi, acc, pol.adm, pol.evi, np.array([misses, rows], dtype=np.int64)


def run_job(job):
    return run_lock(job) if job[1] == "lock4" else run_single(job)


def build_seed(short):
    """leave-category-out: per category, the global hottest keys (most routed share of a prompt, summed over other categories' prompts)"""
    cats = sorted({d["cat"] for d in short.values()})
    out = {}
    for cat in cats:
        acc = np.zeros(NK)
        for d in short.values():
            if d["cat"] == cat:
                continue
            cnt = np.bincount(np.asarray(keyify(d["ids"])).reshape(-1), minlength=NK)
            acc += cnt / len(d["ids"])
        out[cat] = np.argsort(-acc, kind="stable")[: G["cap"]].tolist()
    return out


# ------------------------------------------------------------------ reporting
def main():
    args = sys.argv[1:]
    opt = {"--long": None, "--cap": str(CAP_DEFAULT), "--jobs": str(os.cpu_count() or 4), "--only": ""}
    pos = []
    i = 0
    while i < len(args):
        if args[i] in opt:
            opt[args[i]] = args[i + 1]
            i += 2
        else:
            pos.append(args[i])
            i += 1
    tdir = os.path.expanduser(pos[0]) if pos else DEFAULT_TRACES
    ldir = opt["--long"] or os.path.join(os.path.dirname(os.path.abspath(tdir)), "traces_long")
    cap = int(opt["--cap"])
    t0 = time.time()
    short = load(tdir)
    data = {"short": short}
    if ldir != "none" and os.path.isdir(ldir):
        data["long"] = load(ldir)
    G["cap"], G["data"] = cap, data
    G["seed"] = build_seed(short)
    order = sorted(short)
    rnd = random.Random(0)
    rnd.shuffle(order)
    G["groups"] = [order[i : i + 4] for i in range(0, len(order), 4)]
    configs = [c for c in build_configs(cap) if opt["--only"] in c[0]]
    G["configs"] = configs
    ntok = {ds: sum(len(d["ids"]) for d in v.values()) for ds, v in data.items()}
    print(f"# expert cache replay, {cap} lines ({cap * 1769472 / 2**30:.2f} GiB), cold start per prompt")
    print(f"short: {len(short)} prompts, {ntok['short']} decode tokens; " + (f"long: {len(data['long'])} generations, {ntok['long']} decode tokens; " if "long" in data else "")
          + f"lock4: {len(G['groups'])} groups of 4 short prompts in lockstep\n")

    jobs = []
    for ci, (label, group, fac, init) in enumerate(configs):
        for ds, v in data.items():
            jobs += [(ci, ds, n) for n in sorted(v)]
        if init == "cold":
            jobs += [(ci, "lock4", gi) for gi in range(len(G["groups"]))]
    # heaviest first so the pool stays busy
    jobs.sort(key=lambda j: -(len(data[j[1]][j[2]]["ids"]) if j[1] != "lock4" else 4 * 287))
    res = {}  # (ci, ds) -> accumulators
    ctx = mp.get_context("fork")
    with ctx.Pool(int(opt["--jobs"])) as pool:
        for ci, ds, item, acc, adm, evi, miss in pool.imap_unordered(run_job, jobs, chunksize=4):
            r = res.setdefault((ci, ds), dict(acc=0, adm=0, evi=0, tok=0, miss=0, per={}, pos=np.zeros((len(BUCKETS), 2), dtype=np.int64)))
            r["acc"] += acc
            r["adm"] += adm
            r["evi"] += evi
            if ds == "lock4":
                r["miss"] += int(miss[0])
                r["tok"] += int(miss[1])
            else:
                r["miss"] += int(miss.sum())
                r["tok"] += len(miss)
                r["per"][item] = int(miss.sum()) / len(miss)
                for b, (lo, hi) in enumerate(BUCKETS):
                    seg = miss[lo:hi]
                    r["pos"][b] += (int(seg.sum()), len(seg))
    print(f"(replay took {time.time() - t0:.0f} s on {opt['--jobs']} workers)\n")

    def hit(ci, ds):
        r = res.get((ci, ds))
        return None if r is None else 100.0 * (1 - r["miss"] / r["acc"])

    def mpt(ci, ds):
        r = res.get((ci, ds))
        return None if r is None else r["miss"] / r["tok"]

    def f(x, nd=1):
        return "-" if x is None else f"{x:.{nd}f}"

    base = next((i for i, c in enumerate(configs) if c[0].startswith("LRU global")), None)
    dss = [d for d in ("short", "long") if d in data]
    # ---- table 1
    print("## 1. cold-start replay, every policy admit-on-miss, one global pool unless stated")
    print("hit% = decode expert accesses served from VRAM; miss/tok = link reads per decode token (320 accesses/token); lock4 miss is per sequence-token;")
    print("adm/evi per token on the short set (admissions that evict count in both).\n")
    head = "| policy | " + " | ".join(f"{d} hit% | {d} miss/tok" for d in dss) + " | lock4 hit% | lock4 miss/seq-tok | adm/tok | evict/tok | short Δ vs LRU (pts) |"
    print(head)
    print("|" + "---|" * (head.count("|") - 1))
    cur = None
    for ci, (label, group, fac, init) in enumerate(configs):
        if init != "cold":
            continue
        if group != cur:
            cur = group
        cells = [label]
        for d in dss:
            cells += [f(hit(ci, d)), f(mpt(ci, d), 2)]
        cells += [f(hit(ci, "lock4")), f(mpt(ci, "lock4"), 2)]
        r = res[(ci, "short")]
        cells += [f(r["adm"] / r["tok"], 2), f(r["evi"] / r["tok"], 2)]
        dl = None if base is None else hit(ci, "short") - hit(base, "short")
        cells.append(("%+.2f" % dl) if dl is not None else "-")
        print("| " + " | ".join(cells) + " |")

    # ---- best two non-LRU candidates (hit on short+long weighted by tokens)
    def comb(ci):
        a = b = 0
        for d in dss:
            r = res[(ci, d)]
            a += r["miss"]
            b += r["acc"]
        return 1 - a / b

    cand = [ci for ci, c in enumerate(configs) if c[3] == "cold" and not c[0].startswith("LRU global")]
    best = sorted(cand, key=comb, reverse=True)[:2]
    print("\n## 2. evictions / admissions per token: LRU baseline and the two best other candidates (short+long hit rate)\n")
    print("| policy | " + " | ".join(f"{d} adm/tok | {d} evict/tok | {d} miss/tok" for d in dss) + " |")
    print("|" + "---|" * (1 + 3 * len(dss)))
    for ci in ([base] if base is not None else []) + best:
        cells = [configs[ci][0]]
        for d in dss:
            r = res[(ci, d)]
            cells += [f(r["adm"] / r["tok"], 2), f(r["evi"] / r["tok"], 2), f(r["miss"] / r["tok"], 2)]
        print("| " + " | ".join(cells) + " |")
    if base is not None:
        print("\nper-prompt paired comparison against LRU (miss/token difference, negative = fewer link reads than LRU):\n")
        print("| policy | set | prompts better / equal(<0.05) / worse | mean diff miss/tok | std error |")
        print("|---|---|---|---|---|")
        for ci in best:
            for d in dss:
                diff = np.array([res[(ci, d)]["per"][n] - res[(base, d)]["per"][n] for n in sorted(data[d])])
                se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else float("nan")
                print(f"| {configs[ci][0]} | {d} | {(diff < -0.05).sum()} / {(np.abs(diff) <= 0.05).sum()} / {(diff > 0.05).sum()} | {diff.mean():+.2f} | {se:.2f} |")

    # ---- table 3: seeding / warm start by decode position
    seedrows = [ci for ci, c in enumerate(configs) if c[0].startswith("LRU global") or c[3] != "cold"]
    print("\n## 3. cold vs seeded vs prefill-warm LRU: misses per decode token by decode position (320 accesses/token)\n")
    print("| set | start | " + " | ".join(BUCKET_NAMES) + " | all tokens |")
    print("|" + "---|" * (3 + len(BUCKET_NAMES)))
    names = {"cold": "cold LRU"}
    for d in dss:
        for ci in seedrows:
            r = res[(ci, d)]
            cells = [d, configs[ci][0] if configs[ci][3] != "cold" else "LRU cold start"]
            for b in range(len(BUCKETS)):
                m, n = r["pos"][b]
                cells.append(f"{m / n:.1f} (n={n})" if n else "-")
            cells.append(f"{r['miss'] / r['tok']:.1f}")
            print("| " + " | ".join(cells) + " |")
    # steady-state reference: LRU's miss/token in 512+ / 128-512, and tokens before each warm start catches up
    print("\n## 4. what the warm starts save (LRU cold minus warm, link reads per prompt, summed over decode tokens, per bucket)\n")
    print("| set | start | " + " | ".join(BUCKET_NAMES) + " | total reads saved per prompt | prompts |")
    print("|" + "---|" * (4 + len(BUCKET_NAMES)))
    for d in dss:
        nprompt = len(data[d])
        for ci in seedrows:
            if configs[ci][3] == "cold":
                continue
            cells = [d, configs[ci][0]]
            tot = 0
            for b in range(len(BUCKETS)):
                dm = res[(base, d)]["pos"][b][0] - res[(ci, d)]["pos"][b][0]
                tot += dm
                n = res[(base, d)]["pos"][b][1]
                cells.append(f"{dm / nprompt:+.0f}" if n else "-")
            cells.append(f"{tot / nprompt:+.0f}")
            cells.append(str(nprompt))
            print("| " + " | ".join(cells) + " |")
    print(f"\n(total wall time {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
