"""The routing decision: min cost over the backends that can take the request, with the stage-0 rules of setups-simulation.md s6.

  1. context: n_in + output must fit the backend's context (an ESTIMATE must clear the limit by `estimate_ctx_margin`)  -> prompts above M's 40,960 go to S
  2. breaker: an open breaker removes the backend
  3. guard (interactive): a backend with `guard_rows` (M: 4096) does not take a prompt with more uncached rows than that while its streams are
     decoding (a long prefill stops all of them: worst gap 15-77 s in the simulator), unless it is idle
  4. cost: policy.mode "cost" = own E2E + w_int * delay imposed on the running streams (sim.py 'cost-aware'); "ll" = predicted TTFT only (sim.py 'least-loaded')
  5. SLO (interactive): each second of predicted TTFT above the SLO costs `policy.slo_penalty` seconds of cost (a huge value = hard filter)
  6. prefix affinity is not a rule: a backend holding the prefix simply has fewer uncached rows, hence a lower predicted TTFT and cost
  7. background class: no SLO, higher w_int (protect running interactive streams), a bonus for the secondary role (M) while its queue is short
  8. primary_bias_s + primary_bias_frac: the primary (S) wins unless the secondary is cheaper by that many seconds plus that share of the primary's cost
"""
from . import pd
from .model import predict


class ReqCtx:
    """everything the policy needs to know about one request"""
    __slots__ = ("id", "cls", "n_in", "n_exact", "out_est", "ctx_out", "cached", "slo_ttft", "force", "session", "s_only", "sm_off", "ahead")

    def __init__(self, rid, cls, n_in, n_exact, out_est, ctx_out, cached, slo_ttft=None, force=None, session=None, s_only=None, sm_off=False):
        self.id, self.cls, self.n_in, self.n_exact, self.out_est, self.ctx_out = rid, cls, n_in, n_exact, out_est, ctx_out
        self.cached, self.slo_ttft, self.force, self.session = cached, slo_ttft, force, session     # cached: {backend name: rows it holds}
        self.s_only, self.sm_off = s_only, sm_off       # PD: why only S may serve this request (pd.features) / the SM path is switched off (fallbacks)
        self.ahead = None               # {backend name: requests waiting in the router for that backend ahead of this one} (admission control)

    def ahead_for(self, name):
        return self.ahead.get(name, 0) if self.ahead else 0


class Decision:
    __slots__ = ("choice", "ranking", "rule", "cands", "error", "status", "preds", "paths", "retry_after")

    def __init__(self):
        self.choice, self.ranking, self.rule, self.cands, self.error, self.status = None, [], None, {}, None, 200
        self.preds = {}                 # path name (= backend name, or "SM") -> model.Pred (for the live bookkeeping; not part of the explain record)
        self.paths = {}                 # path name -> pd.Path (which backend gets the request, which one prefills, the body directive)
        self.retry_after = None         # seconds: set with a 503 by admission control

    def as_dict(self):
        return {"choice": self.choice, "ranking": self.ranking, "rule": self.rule, "candidates": self.cands, "error": self.error}


def slo_ttft_for(cfg, ctx):
    if ctx.slo_ttft is not None:
        return ctx.slo_ttft
    s = cfg["policy"]["slo"]
    return s["ttft_base_s"] + s["ttft_per_ktok_s"] * ctx.n_in / 1000.0


def decide(cfg, backends, ctx, now):
    """backends: list of tfrouter.backend.Backend.  Returns a Decision with per-backend predictions (explain data)."""
    pol = cfg["policy"]
    cls = pol["classes"].get(ctx.cls) or pol["classes"][cfg["default_class"]]
    w_int = cls["w_int"] if cls.get("w_int") is not None else pol["w_int"]
    margin = pol["estimate_ctx_margin"]
    d = Decision()
    slo_on = bool(cls.get("slo"))
    slo = slo_ttft_for(cfg, ctx) if slo_on else None
    itl_slo = pol["slo"]["itl_s"]

    n_ctx = ctx.n_in + ctx.ctx_out
    for b in backends:
        c = {"role": b.role}
        d.cands[b.name] = c
        if not b.enabled:
            c["excluded"] = "disabled"
            continue
        if ctx.force and b.name != ctx.force:
            c["excluded"] = "not-forced"
            continue
        limit = b.profile.max_context
        need = n_ctx if ctx.n_exact else n_ctx * (1.0 + margin)
        if need > limit:
            c["excluded"] = "context"
            continue
        if ctx.s_only and ctx.s_only != "exact" and b.name == cfg["pd"]["decode_backend"] and not ctx.force:
            c["excluded"] = "s-only:" + ctx.s_only      # the secondary cannot serve it as asked: greedy-only decode, grammar fields, images
            continue
        if not b.available(now):
            c["excluded"] = "breaker-" + b.breaker.state
            continue
        cached = min(ctx.cached.get(b.name, 0), max(0, ctx.n_in - 1))
        rows = max(1, ctx.n_in - cached)
        v = b.view(now, ctx.ahead_for(b.name))
        p = predict(b.profile, v, rows, ctx.out_est, w_int)
        c.update({"cached": cached, "rows": rows, "view": v.as_dict(), "pred": p.as_dict(), "_p": p, "_v": v})
        c["slo_ok"] = (slo is None or p.ttft <= slo) and p.step1 <= itl_slo

    pd.add_paths(cfg, backends, ctx, d, w_int, slo, itl_slo, now)

    live = [n for n, c in d.cands.items() if "_p" in c]
    ctx_fail = [n for n, c in d.cands.items() if c.get("excluded") == "context"]
    if not live:
        if ctx_fail and all(c.get("excluded") in ("context", "not-forced", "disabled") for n, c in d.cands.items() if n != pd.SM):
            d.status, d.error, d.rule = 413, "prompt (%d tokens + %d output) exceeds every backend's context" % (ctx.n_in, ctx.ctx_out), "no-backend-fits"
        else:
            d.status, d.error, d.rule = 503, "no backend available", "no-backend"
        _clean(d)
        return d

    # guard: long uncached prompt onto a backend whose streams would stall (interactive only; relaxed if it would leave nothing)
    guarded = []
    for b in backends:
        c = d.cands[b.name]
        if "_p" not in c or b.guard_rows is None or not slo_on:
            continue
        if c["rows"] > b.guard_rows and c["_v"].k > 0:
            c["guard"] = "long-prompt-while-decoding"
            guarded.append(b.name)
    pool = [n for n in live if n not in guarded] or live
    if guarded and len(pool) == len(live):
        d.rule = "guard-relaxed"

    lam = pol["slo_penalty"]

    def key(n):
        c = d.cands[n]
        p = c["_p"]
        base = p.ttft if pol["mode"] == "ll" else p.cost
        k = base
        if slo is not None:
            viol = max(0.0, p.ttft - slo)
            c["slo_violation_s"] = round(viol, 3)
            k += lam * viol
        b = next(x for x in backends if x.name == d.paths[n].backend)
        k += pd.extra_cost(cfg, d, n, p)
        if b.role == "primary":
            k -= pol["primary_bias_s"] + pol["primary_bias_frac"] * base
        elif cls.get("prefer_role") == b.role:
            cap = cls.get("secondary_max_backlog_s")
            if cap is None or c["_v"].backlog <= cap:
                k -= cls.get("secondary_bonus_s") or 0.0
        c["key"] = round(k, 3)
        return k

    if ctx.force:
        order = list(pool)
        d.rule = "forced"
    else:
        order = sorted(pool, key=key)
        order += sorted([n for n in live if n not in order], key=key)       # guarded candidates: last-resort failover targets
        d.rule = d.rule or ("min-ttft" if pol["mode"] == "ll" else "min-cost")
        if slo is not None and not any(d.cands[n]["slo_ok"] for n in pool):
            d.rule += "+slo-miss"
        if len(order) == 1:
            d.rule += "+only-candidate"
    d.ranking = order
    d.choice = order[0]
    rej = pol.get("reject_ttft_s")
    if rej and d.cands[d.choice]["_p"].ttft > rej and not ctx.force:
        d.status, d.error, d.rule = 503, "predicted TTFT %.1f s exceeds reject_ttft_s" % d.cands[d.choice]["_p"].ttft, "overload"
    _clean(d)
    return d


def _clean(d):
    """move the private objects out of the explain dict"""
    for n, c in d.cands.items():
        p = c.pop("_p", None)
        c.pop("_v", None)
        if p is not None:
            d.preds[n] = p
