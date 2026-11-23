"""Cost model: predicted TTFT / E2E / interference of placing one request on one backend.

This is bench/sim.py `predict()` for P == D (two independent full servers), parameterised by a per-backend Profile:

    ttft   = max(slot_wait, backlog) + prefill(uncached rows) + decode_interleave_inflation
    e2e    = ttft + (n_out - 1) * seconds_per_token(k + pending + 1)
    interf = k * stall + decode slowdown for the current and reserved future streams
             with View.rem set (live router): the slowdown term uses each stream's remaining output budget;
             slot_wait counts the requests waiting in the router ahead of this one (View built by Backend.view(now, ahead))
    cost   = e2e + w_int * interf

No I/O and no clock: everything is an argument, so the same functions run inside the discrete-event simulator (tools/simcompare.py)
and in the live proxy.
"""


def interp(pts, x):
    """piecewise-linear through pts, flat below the first point, linear extrapolation above the last (as sim.py interp)"""
    if x <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    return y1 + (y1 - y0) / (x1 - x0) * (x - x1)


class Profile:
    """timing profile of one backend.  pf_scale / dec_scale are the runtime calibration factors (EWMA of measured / table)."""

    def __init__(self, cfg):
        self.name = cfg["name"]
        self.slots = int(cfg["slots"])
        self.max_context = int(cfg["max_context"])
        self.pf_pts = [tuple(map(float, p)) for p in cfg["prefill_pts"]]
        self.round_pts = [tuple(map(float, p)) for p in cfg["decode"]["round_ms_pts"]]
        self.tpe_pts = [tuple(map(float, p)) for p in cfg["decode"]["tokens_per_round_pts"]]
        pf = cfg["prefill"]
        self.mode = pf["mode"]
        self.chunk_rows = int(pf["chunk_rows"] or 0)
        self.window_s = float(pf["window_s"] or 0.0)
        self.stall_factor = float(pf["stall_factor"])
        self.pf_scale = 1.0     # >1: the backend prefills slower than the table
        self.dec_scale = 1.0    # >1: the backend decodes faster than the table

    # ---- timing primitives -------------------------------------------------------------------------------------
    def pf_table(self, rows):
        return interp(self.pf_pts, rows)

    def pf_seconds(self, rows):
        """(total seconds, seconds of one full chunk) of an uncached prompt pass of `rows` rows (same split as sim Machine.pf_seconds)"""
        tot = self.pf_table(rows) * self.pf_scale
        if not self.chunk_rows or rows <= self.chunk_rows:
            return tot, tot
        return tot, tot * self.chunk_rows / rows

    def step_s(self, k):
        """decode ROUND time with k streams (one round delivers tpe(k) tokens per stream)"""
        return interp(self.round_pts, max(1, k)) / 1000.0 / self.dec_scale

    def tpe(self, k):
        return max(1.0, interp(self.tpe_pts, max(1, k)))

    def tok_s(self, k):
        """seconds per token per stream with k streams decoding"""
        return self.step_s(k) / self.tpe(k)

    def stream_tps(self, k):
        return 1.0 / self.tok_s(k)

    def infl(self, k, nch):
        """time the decode interleave adds to a prefill of nch chunks while k streams are decoding (S: a 250 ms window per chunk; M: none, it blocks them)"""
        if k <= 0 or self.mode != "interleave":
            return 0.0
        return nch * self.window_s


class View:
    """Current decoders and reserved future decoders. Pending streams affect decode cost, not current prefill stalls.
    rem is each current decoder's remaining output budget (None means unknown); pending_rem holds the admitted
    requests still prefilling, including remotely prefilled requests, but never prefill-only shadow work."""
    __slots__ = ("k", "slot_wait", "backlog", "slots_used", "queued", "rem", "pending_rem")

    def __init__(self, k=0, slot_wait=0.0, backlog=0.0, slots_used=0, queued=0, rem=None, pending_rem=()):
        self.k, self.slot_wait, self.backlog, self.slots_used, self.queued, self.rem = k, slot_wait, backlog, slots_used, queued, rem
        self.pending_rem = pending_rem

    def as_dict(self):
        return {"k": self.k, "slot_wait": round(self.slot_wait, 3), "backlog": round(self.backlog, 3), "slots_used": self.slots_used, "queued": self.queued}


class Pred:
    __slots__ = ("ttft", "e2e", "interf", "cost", "pf", "infl", "step1", "stall")

    def __init__(self, ttft, e2e, interf, cost, pf, infl, step1, stall):
        self.ttft, self.e2e, self.interf, self.cost, self.pf, self.infl, self.step1, self.stall = ttft, e2e, interf, cost, pf, infl, step1, stall

    def as_dict(self):
        return {"ttft": round(self.ttft, 3), "e2e": round(self.e2e, 3), "interf": round(self.interf, 3), "cost": round(self.cost, 3),
                "pf": round(self.pf, 3), "step1_ms": round(self.step1 * 1000, 2)}


def predict(profile, view, rows, n_out, w_int):
    """predicted cost of placing a request (rows uncached prompt rows, n_out output tokens) on this backend now"""
    rows = max(1, int(rows))
    tot, ch = profile.pf_seconds(rows)
    nch = max(1.0, tot / ch)
    k = view.k
    infl = profile.infl(k, nch)
    ttft = max(view.slot_wait, view.backlog) + tot + infl
    stall = tot * profile.stall_factor
    decode_k = k + len(view.pending_rem)
    step1 = profile.tok_s(decode_k + 1)
    e2e = ttft + (max(1, n_out) - 1) * step1
    slow = step1 - profile.tok_s(max(decode_k, 1))
    running_overlap = k * n_out if view.rem is None else sum(min(r, n_out) for r in view.rem)
    pending_overlap = sum(min(r, n_out) for r in view.pending_rem)
    interf = k * stall + slow * (running_overlap + pending_overlap)
    return Pred(ttft, e2e, interf, e2e + w_int * interf, tot, infl, step1, stall)
