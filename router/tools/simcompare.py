#!/usr/bin/env python3
"""simcompare.py: run the ROUTER'S OWN code (tfrouter.policy / backend / prefix / tokens) as a routing policy inside bench/sim.py's
discrete-event simulator and compare it with the simulator's built-in routers on identical seeds.

The router sees only what it sees in production: it decides at arrival, learns the first token and the finish of each request and keeps
its own per-backend bookkeeping (Backend.view: FIFO prefill backlog, decoders, slot wait).  The simulator's own `A+B least-loaded` and
`cost-aware` read the machines' true queues; the gap between them and the router is the price of imperfect knowledge (plus the router's
extra rules: guard, SLO, primary bias).

    python3 tools/simcompare.py tables [--wl chat,mixed,agent,rag,longdoc] [--seeds 6] [--horizon 900]
    python3 tools/simcompare.py batch
    python3 tools/simcompare.py decisions --wl mixed --rate 0.10      # who goes where, by prompt size
    python3 tools/simcompare.py compare [--rates 0.06,0.09,0.12,0.15]  # the E2 policy vs the fixed router on chat (TTFT p95, goodput, share to M)
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
import sim                                                  # noqa: E402
from tfrouter import admission                               # noqa: E402
from tfrouter import config as cfgmod                        # noqa: E402
from tfrouter.backend import Backend, Inflight               # noqa: E402
from tfrouter.policy import ReqCtx, decide                   # noqa: E402
from tfrouter.prefix import PrefixIndex                      # noqa: E402
from tfrouter.tokens import TokenEstimator                   # noqa: E402


class RouterInSim:
    """one router instance driving one Sim"""

    def __init__(self, s, overrides, oracle_out=False, affinity=True):
        ov = {"backends": {"S": {"url": "http://s:1"}, "M": {"url": "http://m:2"}}, "calibration": {"enabled": False}}
        _deep(ov, overrides)
        self.cfg = cfgmod.load(None, ov)
        self.backends = [Backend(c, self.cfg) for c in self.cfg["backends"].values()]
        self.by = {b.name: b for b in self.backends}
        self.prefix = PrefixIndex(self.cfg)
        self.tokens = TokenEstimator(self.cfg)
        self.oracle_out, self.affinity = oracle_out, affinity
        self.s = s
        self.log = []                                      # (n_in, cls, choice, rule)
        self.admit_on = bool(self.cfg["admission"]["enabled"])
        self.gate = admission.WaitQueue()                  # late binding: requests waiting for a lane (the live router's admission.WaitQueue)
        self.held = {}                                     # id(r) -> (waiter, r, original arrival time)

    def _decide(self, r, ahead=None):
        now = self.s.t
        cls = "background" if r.cls == "batch" else "interactive"
        cached = {}
        if self.affinity and r.conv is not None:
            for b in self.backends:
                cached[b.name] = self.prefix.lookup_session(r.conv.id, b.name, now, r.n_in)
        out_est = r.n_out if self.oracle_out else self.tokens.out_est(cls, r.n_out)
        ctx = ReqCtx(id(r), cls, r.n_in, True, out_est, r.n_out, cached)
        ctx.ahead = ahead
        return decide(self.cfg, self.backends, ctx, now), cls, cached, out_est

    def _ready(self, dec, prio, seq):
        """admission.ready on the simulator's backends: the first choice has a free lane and no older waiter wants it"""
        b = self.by[dec.ranking[0]]
        return b.lane_free() and not self.gate.blocked(prio, seq, {b.name})

    def arrive(self, r, orig):
        """Sim.arrive for a router setup: with admission control the request goes in at once if a lane is free (and nobody older could use it), else it waits"""
        if not self.admit_on:
            return orig(self.s, r)
        cls = "background" if r.cls == "batch" else "interactive"
        prio = admission.prio_of(self.cfg, cls)
        r.rt_ahead = self.gate.ahead(prio, admission.INF)
        dec, cls, _, _ = self._decide(r, r.rt_ahead)
        if dec.status == 200 and self._ready(dec, prio, admission.INF):
            return orig(self.s, r)
        w = self.gate.add(prio, set(dec.ranking), {dec.ranking[0]} if dec.ranking else ())
        self.held[w[1]] = (w, r, self.s.t)

    def drain(self):
        """a lane freed: every waiter in order decides again on fresh state (the live bind loop)"""
        for w in list(self.gate.waiters):
            _, r, t_arr = self.held[w[1]]
            r.rt_ahead = self.gate.ahead(w[0], w[1])
            dec, cls, _, _ = self._decide(r, r.rt_ahead)
            if dec.status != 200 or self._ready(dec, w[0], w[1]):
                self.gate.remove(w)
                del self.held[w[1]]
                self._orig_arrive(self.s, r)
                r.t_arr = t_arr                                # TTFT counts the wait in the router
            else:
                w[3], w[4] = frozenset(dec.ranking), frozenset({dec.ranking[0]})

    def route(self, r):
        s, now = self.s, self.s.t
        dec, cls, cached, out_est = self._decide(r, getattr(r, "rt_ahead", None))
        name = dec.choice or "S"
        if not s.feasible(r, name, name):
            name = "S"
        b = self.by[name]
        pred = dec.preds.get(name)
        rec = Inflight(id(r), cls, True, r.n_in, max(1, r.n_in - cached.get(name, 0)), cached.get(name, 0), out_est)
        if pred is None:                                   # forced fallback: price it with the model
            from tfrouter.model import predict
            pred = predict(b.profile, b.view(now), rec.rows, out_est, 1.0)
        b.admit(rec, now, pred.ttft, pred.pf)
        r.rt_rec, r.rt_name = rec, name
        self.log.append((r.n_in, cls, name, dec.rule))
        return (name, name)

    def on_first(self, r, now):
        rec = getattr(r, "rt_rec", None)
        if rec is not None:
            self.by[r.rt_name].first_token(rec, now)

    def on_finish(self, r, now):
        rec = getattr(r, "rt_rec", None)
        if rec is None:
            return
        self.by[r.rt_name].finish(rec, now, None)
        cls = rec.cls
        self.tokens.observe_out(cls, r.n_out)
        if r.conv is not None and self.affinity:
            self.prefix.record_session(r.conv.id, r.rt_name, r.n_in + r.n_out, now)


def _deep(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep(dst[k], v)
        else:
            dst[k] = v


_orig_first, _orig_finish, _orig_arrive = sim.Machine.first_token, sim.Sim.finish, sim.Sim.arrive
RouterInSim._orig_arrive = staticmethod(_orig_arrive)


def _first(m, r):
    rt = getattr(m.sim, "rt", None)
    if rt is not None:
        rt.on_first(r, m.sim.t)
    _orig_first(m, r)


def _finish(s, r, Dm):
    rt = getattr(s, "rt", None)
    if rt is not None:
        rt.on_finish(r, s.t)
    _orig_finish(s, r, Dm)
    if rt is not None and rt.admit_on and rt.gate.waiters:       # after the machine released its slot: the waiter that takes the lane starts at once
        rt.drain()


def _arrive(s, r):
    make = getattr(s.setup["policy"], "make", None)
    if make is None:                                       # one of sim.py's own routers
        return _orig_arrive(s, r)
    if getattr(s, "rt", None) is None:
        s.rt = make(s)
    s.rt.arrive(r, _orig_arrive)


sim.Machine.first_token = _first
sim.Sim.finish = _finish
sim.Sim.arrive = _arrive


def router_policy(overrides, **kw):
    def policy(s, r):
        if getattr(s, "rt", None) is None:
            s.rt = RouterInSim(s, overrides, **kw)
        return s.rt.route(r)
    policy.make = lambda s: RouterInSim(s, overrides, **kw)
    return policy


# the policy of the E2 run (2026-11-11): no admission control, every stream outlives the new request, plain output EWMA, primary bias 0.3 s
E2_POLICY = {"policy": {"interf_overlap": False, "primary_bias_frac": 0.0}, "tokens": {"output_warmup": False}, "admission": {"enabled": False}}


SETUP_DEFS = {
    "Router E2 policy (before the fix)": router_policy(E2_POLICY),
    "Router cost (tfrouter default)": router_policy({}),
    "Router ll (policy.mode ll)": router_policy({"policy": {"mode": "ll"}}),
    "Router cost (slo_penalty 0)": router_policy({"policy": {"slo_penalty": 0.0}}),
    "Router cost (slo_penalty 0.3)": router_policy({"policy": {"slo_penalty": 0.3}}),
    "Router cost (slo_penalty 3)": router_policy({"policy": {"slo_penalty": 3.0}}),
    "Router cost (slo_penalty 1000 = hard)": router_policy({"policy": {"slo_penalty": 1000.0}}),
    "Router cost (no SLO rule)": router_policy({"policy": {"classes": {"interactive": {"slo": False}}}}),
    "Router cost (no SLO, no guard)": router_policy({"policy": {"classes": {"interactive": {"slo": False}}}, "backends": {"M": {"guard_rows": None}}}),
    "Router cost (no guard)": router_policy({"backends": {"M": {"guard_rows": None}}}),
    "Router cost (no affinity)": router_policy({}, affinity=False),
    "Router cost (oracle output length)": router_policy({}, oracle_out=True),
    "Router cost (w_int 0)": router_policy({"policy": {"w_int": 0.0}}),
}
for k, fn in SETUP_DEFS.items():
    sim.SETUPS[k] = dict(policy=fn, S_profile="prod")

BASELINES = ["A  S alone", "A+B round-robin", "A+B least-loaded", "A+B overflow (M only if S slots full)", "A+B cost-aware (no remote)"]
ROUTERS = list(SETUP_DEFS)


def tables(wls, seeds, horizon, only=None, setups=None):
    sim.SEEDS = tuple(range(1, seeds + 1))
    sim.HORIZON = horizon
    for wl in wls:
        for rate in (only or sim.WORKLOADS[wl][1]):
            sim.table(wl, rate, setups or (BASELINES + ROUTERS))


def batch():
    out = {}
    print("\n**batch: 24 docs of 8-16k in / 400 out, all at t=0, class background**\n")
    print("| setup | makespan s | out tok/s | util S | util M |\n|---|---|---|---|---|")
    for n in BASELINES + ROUTERS:
        sm = sim.run_one("batch", 0.0, sim.SETUPS[n], sim.PARAM, 1)
        r = sim.summarize([sm], "batch")
        print("| %s | %.0f | %.1f | %.0f%% | %.0f%% |" % (n, r["makespan"], r["out_tps"], r["utilS"] * 100, r["utilM"] * 100))


def decisions(wl, rate, seeds, horizon):
    sim.HORIZON = horizon
    print("\n**%s @ %g/s: share of requests the router sends to M, by prompt size (sim.py routers: from the simulated placement)**\n" % (wl, rate))
    bins = [(0, 4096), (4096, 12000), (12000, 30000), (30000, 10 ** 9)]
    print("| setup | " + " | ".join("n_in %d-%s" % (a, b if b < 10 ** 9 else "inf") for a, b in bins) + " | all |\n|---|" + "---|" * (len(bins) + 1))
    for n in ["A+B least-loaded", "A+B cost-aware (no remote)", "A+B overflow (M only if S slots full)"] + ROUTERS[:2]:
        cnt = {i: [0, 0] for i in range(len(bins) + 1)}
        for sd in range(1, seeds + 1):
            sm = sim.run_one(wl, rate, sim.SETUPS[n], sim.PARAM, sd, horizon)
            for q in sm.reqs:
                if q.D is None or q.t_arr < 0.15 * horizon or q.t_arr > horizon:
                    continue
                for i, (a, b) in enumerate(bins):
                    if a <= q.n_in < b:
                        cnt[i][0] += 1
                        cnt[i][1] += q.D == "M"
                cnt[len(bins)][0] += 1
                cnt[len(bins)][1] += q.D == "M"
        print("| %s | " % n + " | ".join("%.0f%% (n=%d)" % (100.0 * c[1] / c[0] if c[0] else float("nan"), c[0]) for c in cnt.values()) + " |")


def m_share(sims):
    """share of the measured window's requests placed on M (a sim.py router: from the simulated placement)"""
    n = m = 0
    for s in sims:
        for q in s.reqs:
            if q.D is None or q.t_arr < 0.15 * sim.HORIZON or q.t_arr > sim.HORIZON:
                continue
            n += 1
            m += q.D == "M"
    return m / max(n, 1)


def compare(rates, seeds, horizon, wl="chat", slos=((4.0, 0.15), (2.0, 0.10))):
    """the E2 policy vs the fixed one (and S alone) at the E2 chat rates: TTFT p95, goodput, share to M.  Goodput under the gate-E2 SLO (TTFT 4 s, ITL p95 150 ms)
    and under sim.py's own chat SLO (2 s, 100 ms)."""
    sim.SEEDS = tuple(range(1, seeds + 1))
    sim.HORIZON = horizon
    names = ["A  S alone", "Router E2 policy (before the fix)", "Router cost (tfrouter default)"]
    print("\n**%s, %d seeds x %.0f s: the E2 policy vs the fixed router**\n" % (wl, seeds, horizon))
    print("| rate /s | setup | TTFT p95 s | E2E p95 s | " + " | ".join("goodput req/s @ %gs/%gms" % (a, b * 1000) for a, b in slos) + " | share to M | finished |")
    print("|---|---|---|---|" + "---|" * len(slos) + "---|---|")
    for rate in rates:
        for n in names:
            sims = [sim.run_one(wl, rate, sim.SETUPS[n], sim.PARAM, sd, horizon) for sd in sim.SEEDS]
            rs = [sim.summarize(sims, wl, horizon, slo=slo) for slo in slos]
            print("| %.2f | %s | %.2f | %.1f | %s | %.0f%% | %.0f%% |" % (rate, n, rs[0]["ttft95"], rs[0]["e2e95"], " | ".join("%.3f" % (r["good"] / 60.0) for r in rs),
                                                                   100 * m_share(sims), 100 * rs[0]["fin"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("tables", "batch", "decisions", "compare"))
    ap.add_argument("--rates", default="0.06,0.09,0.12,0.15", help="compare: chat rates")
    ap.add_argument("--wl", default="chat,mixed,agent,rag,longdoc")
    ap.add_argument("--rate", type=float)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--horizon", type=float, default=900.0)
    ap.add_argument("--setups", help="semicolon-separated setup names (default: sim baselines + all router variants)")
    a = ap.parse_args()
    if a.cmd == "tables":
        tables(a.wl.split(","), a.seeds, a.horizon, [a.rate] if a.rate else None, a.setups.split(";") if a.setups else None)
    elif a.cmd == "compare":
        compare([float(x) for x in a.rates.split(",")], a.seeds, a.horizon, a.wl.split(",")[0])
    elif a.cmd == "batch":
        batch()
    else:
        decisions(a.wl.split(",")[0], a.rate or sim.WORKLOADS[a.wl.split(",")[0]][1][1], a.seeds, a.horizon)


if __name__ == "__main__":
    main()
