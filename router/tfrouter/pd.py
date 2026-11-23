"""PD+ fleet: the third path (SM), per-request prefill directive, S shadow bookkeeping (designs/pd-plus-fleet.md).

Three paths per request (P = where the prompt is prefilled, D = where it is decoded):
  S   (SS): the S backend alone                       no directive
  M   (MM): the M backend alone                       body `prefill: "never"`   (M's own server never pulls it)
  SM      : S prefills with replay export, M pulls it and decodes     body `prefill: "force"`   (the request goes to the M backend)

SM is not a backend: it is a candidate of `policy.decide` that names the M backend with a directive, and whose cost model is
    tP   = S.backlog + pf_S(n_in) * export_factor (+ S's decode interleave windows)       S prefills FIFO, export included
    H    = link_extra + wire + landing + one tail step                                    [designs/pd-plus-fleet.md 3.1 H(n)]
    ttft = max(M.slot_wait, tP) + H                                                       the M lane is needed at landing, not before
    e2e  = ttft + (n_out - 1) * tok_M(k_M + 1)
    interf = k_S * pf_S (S's running streams stall during the export)  +  k_M * (landing stall + slowdown of M's streams)
    cost = e2e + w_int * interf + mu * (S-seconds)
No I/O, no clock, no asyncio: the same functions run in tests and in the live proxy.  (Late binding / the wait queue is admission.py.)
"""

from .model import Pred

SM = "SM"
S_PATH = "S"
M_PATH = "M"


class Path:
    """a candidate of the decision: where the request is sent (`backend`), which backend prefills (`prefill`), the body directive"""
    __slots__ = ("name", "backend", "prefill", "directive")

    def __init__(self, name, backend, prefill=None, directive=None):
        self.name, self.backend, self.prefill, self.directive = name, backend, prefill, directive

    def as_dict(self):
        return {"backend": self.backend, "prefill": self.prefill or self.backend, "directive": self.directive}


def enabled(cfg):
    return bool(cfg["pd"]["enabled"])


def backend_paths(cfg, backends):
    """name -> Path for every backend; with pd on, the decode backend's own path carries `never` (the server must not pull on its own)"""
    pd = cfg["pd"]
    out = {}
    for b in backends:
        d = None
        if enabled(cfg) and b.name == pd["decode_backend"] and pd["mm_never"]:
            d = "never"
        out[b.name] = Path(b.name, b.name, None, d)
    return out


def handoff_s(pd, n_in):
    """seconds from the end of S's prefill to M's first token: link round trips, the wire, landing, one tail step (design 3.1 H(n); 0.26 s at 2k)"""
    h = pd["handoff"]
    wire = h["kv_bytes_per_token"] * n_in * 8.0 / (h["link_gbit"] * 1e9)
    return h["link_extra_s"] + wire + landing_s(pd, n_in) + h["tail_s"]


def landing_s(pd, n_in):
    h = pd["handoff"]
    return h["landing_base_s"] + h["landing_per_token_s"] * n_in


def predict_sm(pd, ps, pm, vs, vm, n_in, n_out, w_int):
    """Pred of an SM request: S prefills n_in rows (export from 0, no cache), M decodes.  ps / pm: Profiles of S and M; vs / vm: their Views."""
    tot, ch = ps.pf_seconds(n_in)
    tot *= pd["export_factor"]
    ch *= pd["export_factor"]
    nch = max(1.0, tot / max(ch, 1e-9))
    infl = ps.infl(vs.k, nch)
    tp = vs.backlog + tot + infl
    hs = handoff_s(pd, n_in)
    ttft = max(vm.slot_wait, tp) + hs
    step1 = pm.tok_s(vm.k + 1)
    e2e = ttft + (max(1, n_out) - 1) * step1
    stall_s = vs.k * tot * ps.stall_factor
    stall_m = vm.k * landing_s(pd, n_in) * pm.stall_factor
    slow_m = vm.k * (step1 - pm.tok_s(max(vm.k, 1))) * n_out
    interf = stall_s + stall_m + slow_m
    return Pred(ttft, e2e, interf, e2e + w_int * interf, tot, infl, step1, tot * ps.stall_factor)


IMAGE_TYPES = ("image_url", "image", "input_image")


def features(cfg, body, exact_value=None):
    """-> reason or None.  What M (greedy-only today, G2) or the SM hand-off cannot serve: sampled, grammar/logprobs fields, images -> S only;
    `exact` (X-Exactness header) only rules out the hand-off.  Applies with the hand-off off too: policy.decide keeps those off M."""
    pd = cfg["pd"]
    t = body.get("temperature")
    try:
        if t is not None and float(t) > 0 and pd["greedy_only_decode"]:
            return "sampled"
    except (TypeError, ValueError):
        pass
    for k in pd["s_only_fields"]:
        v = body.get(k)
        if v not in (None, False, [], {}, 0):
            return "field:" + k
    for m in body.get("messages") or ():
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list) and any(isinstance(x, dict) and x.get("type") in IMAGE_TYPES for x in c):
            return "images"
    if exact_value and exact_value.strip().lower() == "exact":
        return "exact"
    return None


def add_paths(cfg, backends, ctx, d, w_int, slo, itl_slo, now):
    """extend a Decision under construction (d.cands filled for every backend) with the SM candidate.  Called by policy.decide."""
    pd = cfg["pd"]
    d.paths = backend_paths(cfg, backends)
    if not enabled(cfg):
        return
    by = {b.name: b for b in backends}
    sb, mb = by.get(pd["prefill_backend"]), by.get(pd["decode_backend"])
    c = {"role": "pd", "prefill": pd["prefill_backend"], "decode": pd["decode_backend"]}
    d.cands[SM] = c
    why = None
    if ctx.force and ctx.force not in (SM,):
        why = "not-forced"
    elif sb is None or mb is None or not sb.enabled or not mb.enabled:
        why = "disabled"
    elif ctx.s_only:
        why = "s-only:" + ctx.s_only
    elif not sb.available(now):
        why = "breaker-" + sb.breaker.state + "-prefill"
    elif not mb.available(now):
        why = "breaker-" + mb.breaker.state
    elif ctx.sm_off:
        why = "sm-fallbacks"
    else:
        n_ctx = ctx.n_in + ctx.ctx_out
        need = n_ctx if ctx.n_exact else n_ctx * (1.0 + cfg["policy"]["estimate_ctx_margin"])
        cached_m = min(ctx.cached.get(mb.name, 0), max(0, ctx.n_in - 1))
        if need > mb.profile.max_context or need > pd["max_export_tokens"]:
            why = "context"
        elif ctx.n_in - cached_m < pd["min_rows"]:
            why = "too-few-rows"
    if why:
        c["excluded"] = why
        return
    vs, vm = sb.view(now, ctx.ahead_for(sb.name)), mb.view(now, ctx.ahead_for(mb.name))
    p = predict_sm(pd, sb.profile, mb.profile, vs, vm, ctx.n_in, ctx.out_est, w_int)
    c.update({"cached": min(ctx.cached.get(mb.name, 0), max(0, ctx.n_in - 1)), "rows": ctx.n_in, "view": vm.as_dict(), "view_prefill": vs.as_dict(),
              "pred": p.as_dict(), "_p": p, "_v": vm, "handoff_s": round(handoff_s(pd, ctx.n_in), 3)})
    c["slo_ok"] = (slo is None or p.ttft <= slo) and p.step1 <= itl_slo
    d.paths[SM] = Path(SM, mb.name, sb.name, "force")


def extra_cost(cfg, d, name, p):
    """the shadow price of S-seconds (mu) of a candidate: SS and SM spend S's prefill, MM does not"""
    pd = cfg["pd"]
    if not enabled(cfg) or not pd["mu"]:
        return 0.0
    path = d.paths.get(name)
    if path is None:
        return 0.0
    if (path.prefill or path.backend) == pd["prefill_backend"]:
        return pd["mu"] * p.pf
    return 0.0


class State:
    """per-router PD bookkeeping: SM fallbacks (the M server prefilled a forced prompt itself), outcomes, counters"""

    def __init__(self, pd):
        self.pd = pd
        self.fallbacks = 0              # consecutive SM requests that M did not pull
        self.off_until = 0.0
        self.counts = {"SS": 0, "MM": 0, "SM": 0, "sm_pulled": 0, "sm_fallback": 0, "sm_disabled": 0}
        self.sm_ttft_err = []

    def count_path(self, name):
        key = {S_PATH: "SS", M_PATH: "MM", SM: "SM"}.get(name)
        if key:
            self.counts[key] += 1

    def sm_outcome(self, info, now):
        """info: usage_info of a finished SM request.  A forced request that M prefilled itself is a fallback; N in a row disable SM for a while."""
        pulled = info.get("pulled_prefix") or info.get("remote_prefix") or 0
        if pulled > 0:
            self.fallbacks = 0
            self.counts["sm_pulled"] += 1
            return True
        self.fallbacks += 1
        self.counts["sm_fallback"] += 1
        if self.fallbacks >= int(self.pd["fallback"]["failures"]):
            self.off_until = now + float(self.pd["fallback"]["cooldown_s"])
            self.fallbacks = 0
            self.counts["sm_disabled"] += 1
        return False

    def as_dict(self, now):
        return {"counts": dict(self.counts), "sm_fallbacks_in_a_row": self.fallbacks, "sm_off_for_s": round(max(0.0, self.off_until - now), 1)}


# ---------------------------------------------------------------------------------------------------------------------------------
# live helpers (the Router passes itself: clock, backends, state)
# ---------------------------------------------------------------------------------------------------------------------------------
def available(router, name, dec, now):
    """the decode backend of a path takes the request and, for SM, the prefill backend is up too"""
    path = dec.paths.get(name)
    if path is None:
        return router.backends[name].available(now)
    if not router.backends[path.backend].available(now):
        return False
    return path.prefill is None or router.backends[path.prefill].available(now)


def admit(router, rec, name, dec, now, pred, queue_ahead_fn=None):
    """book the request on its path: an own-prefill path is a plain admit; SM is a lane on the decode backend (pending) plus a shadow prefill on S"""
    path = dec.paths.get(name)
    b = router.backends[path.backend if path else name]
    router.pd.count_path(name)
    if path is None or path.prefill is None:
        b.admit(rec, now, pred.ttft, pred.pf)
        return b
    sb = router.backends[path.prefill]
    shadow = type(rec)(rec.id + "/p", rec.cls, rec.stream, rec.n_in, rec.n_in, 0, rec.out_est)
    vs = sb.view(now)
    rec.path = SM
    sb.shadow_admit(shadow, now, pred.pf)
    b.admit_pending(rec, now, pred.ttft)
    rec.queue_ahead = len(sb.fifo) - 1 + (1 if vs.k else 0)         # prefills ahead on S + S's decoders (interleave windows): a clean sample needs both 0
    rec.link = lambda t, sb=sb, sh=shadow: sb.shadow_done(sh, t)
    return b


def complete(router, b, rec, now, info):
    """an SM request finished: was it pulled, and what did S's prefill really cost (feeds S's prefill scale from the first-token time)"""
    if rec.path != SM:
        return
    pd = router.cfg["pd"]
    if not router.pd.sm_outcome(info, now):
        return
    sb = router.backends[pd["prefill_backend"]]
    cal = router.cfg["calibration"]
    if not (cal["enabled"] and rec.stream and rec.t_first is not None and rec.queue_ahead == 0 and rec.n_in >= cal["min_rows"]):
        return
    s_sec = (rec.t_first - rec.t_admit) - handoff_s(pd, rec.n_in)
    base = sb.profile.pf_table(rec.n_in) * pd["export_factor"]
    if s_sec <= 0 or base <= 0:
        return
    router.pd.sm_ttft_err.append(round((rec.t_first - rec.t_admit) / max(1e-6, rec.est_first - rec.t_admit), 3))
    del router.pd.sm_ttft_err[:-200]
    s = sb.ew_pf_scale.update(min(4.0, max(0.25, s_sec / base)))
    sb.profile.pf_scale = min(cal["max_scale"], max(cal["min_scale"], s))
