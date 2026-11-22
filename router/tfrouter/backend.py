"""Per-backend live state: in-flight requests, the router's model of the backend's prefill queue and decoders, measured EWMAs,
circuit breaker, API key, health and external-load readings.

All methods take `now` (router seconds); nothing here reads a clock, so the discrete-event comparison can drive it with simulated time.

The router never sees inside a backend.  What it can observe per request is: admit (it decided), first token (first SSE event with
content), finish (usage / timings in the last event or the JSON body), abort.  From these it keeps
  * waiting prefills (admitted, no first token yet) in arrival order: the backend prefills FIFO [MEAS baselines.md s4], so
        backlog = sum(predicted prefill seconds of the waiting) - progress of the head;
  * decoding streams (k) and each one's expected remaining time (for the slot wait).
"""
import os
import re
import time
from urllib.parse import urlsplit

from .model import Profile, View, interp

REM_FLOOR = 0.25      # a running stream that has already produced its estimated output is assumed to have this share of the estimate left (estimates are means, half the streams outlive them)


class Ewma:
    __slots__ = ("value", "alpha", "n")

    def __init__(self, init, alpha):
        self.value, self.alpha, self.n = init, alpha, 0

    def update(self, x):
        self.value = x if self.n == 0 and self.value is None else self.value + self.alpha * (x - self.value)
        self.n += 1
        return self.value


class KeyFile:
    """API key from a TensorFold-format file (one key per line, 'label: key', '#' comments), re-read when the mtime changes"""
    _line = re.compile(r"^[A-Za-z0-9._\-]+(\[[^\]]*\])?\s*:\s*(\S+)$")

    def __init__(self, path=None, env=None):
        self.path, self.env = path, env
        self._key, self._mtime, self._checked = None, None, 0.0
        self.reload(force=True)

    def reload(self, force=False):
        if self.env and os.environ.get(self.env):
            self._key = os.environ[self.env].split(",")[0].strip()
            return
        if not self.path:
            return
        try:
            mt = os.stat(self.path).st_mtime_ns
        except OSError:
            self._key = None
            return
        if not force and mt == self._mtime:
            return
        self._mtime = mt
        key = None
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                m = self._line.match(line)
                key = m.group(2) if m else line
                break
        self._key = key

    def get(self):
        if self.path and time.monotonic() - self._checked >= 1.0:       # rotation: at most one stat a second (wall clock on purpose)
            self._checked = time.monotonic()
            self.reload()
        return self._key


class Breaker:
    """closed -> (threshold consecutive failures) -> open -> (cooldown) -> half_open (one trial) -> closed | open (cooldown doubles)"""

    def __init__(self, cfg):
        self.threshold = int(cfg["failure_threshold"])
        self.base_cooldown = float(cfg["cooldown_s"])
        self.max_cooldown = float(cfg["max_cooldown_s"])
        self.half_open_max = int(cfg["half_open_max"])
        self.state = "closed"
        self.fails = 0
        self.cooldown = self.base_cooldown
        self.opened_at = 0.0
        self.trials = 0
        self.trips = 0

    def _tick(self, now):
        if self.state == "open" and now - self.opened_at >= self.cooldown:
            self.state, self.trials = "half_open", 0

    def allows(self, now):
        self._tick(now)
        if self.state == "closed":
            return True
        if self.state == "half_open":
            return self.trials < self.half_open_max
        return False

    def reserve(self, now):
        """called when a request is actually sent to the backend"""
        self._tick(now)
        if self.state == "half_open":
            self.trials += 1

    def success(self, now):
        self.fails = 0
        if self.state != "closed":
            self.state, self.trials, self.cooldown = "closed", 0, self.base_cooldown

    def failure(self, now):
        self._tick(now)
        self.fails += 1
        if self.state == "half_open":
            self.cooldown = min(self.cooldown * 2, self.max_cooldown)
            self._open(now)
        elif self.state == "closed" and self.fails >= self.threshold:
            self._open(now)

    def force_open(self, now):
        self._open(now)

    def _open(self, now):
        self.state, self.opened_at, self.trials = "open", now, 0
        self.trips += 1

    def release_trial(self):
        if self.state == "half_open" and self.trials > 0:
            self.trials -= 1

    def as_dict(self, now):
        self._tick(now)
        return {"state": self.state, "fails": self.fails, "cooldown_s": self.cooldown, "trips": self.trips,
                "reopens_in_s": round(max(0.0, self.opened_at + self.cooldown - now), 2) if self.state == "open" else 0.0}


class Inflight:
    """one request the router has sent to this backend"""
    __slots__ = ("id", "cls", "stream", "n_in", "rows", "cached_est", "out_est", "pf_pred", "t_admit", "t_first", "est_first", "k_at_admit",
                 "queue_ahead", "k_int0", "epoch0", "done", "meta", "path", "link")

    def __init__(self, rid, cls, stream, n_in, rows, cached_est, out_est):
        self.id, self.cls, self.stream, self.n_in, self.rows, self.cached_est, self.out_est = rid, cls, stream, n_in, rows, cached_est, out_est
        self.pf_pred = 0.0
        self.t_admit = self.t_first = self.est_first = None
        self.k_at_admit = 0
        self.queue_ahead = 0
        self.k_int0 = 0.0
        self.epoch0 = 0
        self.done = False
        self.meta = None                                  # the explain record of this request (the router adds the outcome)
        self.path = None                                  # PD: "SM" for a hand-off request (M's first token is S's prefill + landing: no prefill calibration of M)
        self.link = None                                  # PD: called once with the time the request leaves the prefill phase (first token / drop): ends S's shadow prefill


class Backend:
    def __init__(self, cfg, global_cfg):
        self.cfg = cfg
        self.name = cfg["name"]
        self.role = cfg["role"]
        self.model = cfg["model"]
        self.enabled = cfg["enabled"]
        self.profile = Profile(cfg)
        self.guard_rows = cfg.get("guard_rows")
        self.interf_overlap = bool(global_cfg["policy"]["interf_overlap"])
        self.max_inflight = int(cfg["max_inflight"] or self.profile.slots)       # admission control: requests in flight here (the router holds the rest)
        self.queue_max = int(cfg["queue_max"])
        u = urlsplit(cfg["url"] or "http://invalid:0")
        self.scheme, self.host = u.scheme, u.hostname
        self.port = u.port or (443 if u.scheme == "https" else 80)
        self.base = u.path.rstrip("/")
        self.key = KeyFile(cfg.get("api_key_file"), cfg.get("api_key_env"))
        self.breaker = Breaker(global_cfg["breaker"])
        cal = global_cfg["calibration"]
        self.cal = cal
        self.ew_pf_scale = Ewma(1.0, cal["alpha"])
        self.ew_dec_scale = Ewma(1.0, cal["alpha"])
        self.ew_pf_tps = Ewma(None, cal["alpha"])         # measured prefill rows/s
        self.ew_dec_tps = Ewma(None, cal["alpha"])        # measured per-stream decode tokens/s
        self.ew_dec_k = Ewma(None, cal["alpha"])          # mean concurrency those decode samples ran at
        self.inflight = {}
        self.fifo = []                                    # admitted, no first token yet, arrival order
        self.head_start = 0.0
        self.k_int = 0.0                                  # integral of decoding streams over time
        self.k_t = 0.0
        self.admitted_total = 0
        self.counts = {"requests": 0, "ok": 0, "failed": 0, "cancelled": 0, "failovers_from": 0, "chosen": 0, "cached_hits": 0, "bytes_out": 0}
        # health / load readings
        self.health_ok = None
        self.health_ts = None
        self.live = {}
        self.ext_active = 0
        self.ext_seen = 0
        self.load_raw = None
        self.load_off = False                             # /v1/load not served: use /health live.connections instead
        self.slots_reported = None
        self.pending = []                                 # PD: SM requests admitted here whose prompt S is prefilling: a lane is theirs, M's own prefill queue is not
        self.ext_pf_tokens = 0                            # PD: prompt tokens other clients of the server have waiting or filling (/v1/load)
        self.shadows = []                                 # PD: prefills of SM requests queued here (a request on this server that is not in `inflight`)
        self.on_free = None                               # PD: called when a request leaves this backend (a decode lane freed): wakes the router's hold queue

    # ---- availability ----------------------------------------------------------------------------------------------------
    def available(self, now):
        return self.enabled and self.breaker.allows(now)

    def in_use(self):
        """decode lanes taken: the router's own requests plus other clients' requests the server reports (same rule as `view`)"""
        ext = self.ext_active if self.ext_seen >= 2 else 0
        return len(self.inflight) + ext

    def lane_free(self):
        """admission control: another request may be sent here without exceeding max_inflight"""
        return self.in_use() < self.max_inflight

    # ---- phase bookkeeping -----------------------------------------------------------------------------------------------
    def _decoders(self):
        return [r for r in self.inflight.values() if r.t_first is not None]

    def _integrate(self, now):
        n = sum(1 for r in self.inflight.values() if r.t_first is not None)
        self.k_int += n * max(0.0, now - self.k_t)
        self.k_t = now

    def _mark_first(self, rec, t):
        self._integrate(t)
        rec.t_first = t
        rec.k_int0 = self.k_int
        rec.epoch0 = self.admitted_total
        if rec in self.pending:
            self.pending.remove(rec)
        if rec.link is not None:
            rec.link(t)
            rec.link = None
        if rec in self.fifo:
            was_head = self.fifo[0] is rec
            self.fifo.remove(rec)
            if was_head:
                self.head_start = t

    def _advance(self, now):
        """non-streaming requests give no first-token event: assume it came when the model predicted"""
        for rec in list(self.fifo) + list(self.pending):
            if not rec.stream and rec.est_first is not None and now >= rec.est_first and rec.t_first is None:
                self._mark_first(rec, rec.est_first)

    # ---- the router's view -------------------------------------------------------------------------------------------------
    def view(self, now, ahead=0):
        """the state the model prices a new request against; `ahead` = requests already waiting in the router for this backend (they take the next lanes)"""
        self._advance(now)
        pf = self.profile
        dec = self._decoders()
        waiting = self.fifo
        backlog = 0.0
        if waiting:
            backlog = sum(r.pf_pred for r in waiting) - min(max(0.0, now - self.head_start), waiting[0].pf_pred)
            backlog = max(0.0, backlog)
        if self.ext_pf_tokens > 0:
            rows = 4096.0
            backlog += self.ext_pf_tokens / (rows / pf.pf_seconds(rows)[0])
        ext = self.ext_active if self.ext_seen >= 2 else 0
        k = len(dec) + ext
        in_use = len(self.inflight) + ext
        lim = min(pf.slots, self.max_inflight)
        total = in_use + ahead
        queued = max(0, total - lim)
        slot_wait = 0.0
        if total >= lim and dec:
            rate = pf.tok_s(max(1, k))
            full = [r.out_est * rate for r in dec]
            rem = sorted(max(0.0, f - (now - r.t_first)) for f, r in zip(full, dec))
            if queued < len(rem):
                slot_wait = rem[queued]
            else:                                    # more waiting than running: the lanes turn over again, one mean request length per `len(rem)` requests
                slot_wait = rem[-1] + (queued - len(rem) + 1) * (sum(full) / len(full)) / len(rem)
        rem_tok = None
        if self.interf_overlap:
            rate = pf.tok_s(max(1, k))
            rem_tok = tuple(max(r.out_est - (now - r.t_first) / rate, REM_FLOOR * r.out_est) for r in dec) + (1e12,) * ext
        pending_rem = tuple(r.out_est for r in self.inflight.values() if r.t_first is None)
        return View(k=k, slot_wait=slot_wait, backlog=backlog, slots_used=min(in_use, pf.slots), queued=queued,
                    rem=rem_tok, pending_rem=pending_rem)

    # ---- request lifecycle --------------------------------------------------------------------------------------------------
    def admit(self, rec, now, pred_ttft, pf_pred):
        """the router decided to send rec here.  pred_ttft / pf_pred come from the cost model at decision time."""
        v = self.view(now)
        rec.t_admit = now
        rec.pf_pred = pf_pred
        rec.k_at_admit = v.k
        rec.queue_ahead = len(self.fifo)
        rec.est_first = now + pred_ttft
        if not self.fifo:
            self.head_start = now
        self.fifo.append(rec)
        self.inflight[rec.id] = rec
        self.admitted_total += 1
        self.counts["requests"] += 1

    def admit_pending(self, rec, now, pred_ttft):
        """PD: an SM request.  It holds a lane here from now on but its prompt is prefilled elsewhere: it is not in this backend's prefill FIFO."""
        v = self.view(now)
        rec.t_admit, rec.pf_pred, rec.k_at_admit, rec.queue_ahead = now, 0.0, v.k, 0
        rec.est_first = now + pred_ttft
        self.pending.append(rec)
        self.inflight[rec.id] = rec
        self.admitted_total += 1
        self.counts["requests"] += 1

    def shadow_admit(self, rec, now, pf_pred):
        """PD: the prefill of an SM request queues on this backend (FIFO, export included) but takes no decode lane here"""
        rec.t_admit, rec.pf_pred = now, pf_pred
        if not self.fifo:
            self.head_start = now
        self.fifo.append(rec)
        self.shadows.append(rec)

    def shadow_done(self, rec, now):
        if rec in self.shadows:
            self.shadows.remove(rec)
        if rec in self.fifo:
            was_head = self.fifo[0] is rec
            self.fifo.remove(rec)
            if was_head:
                self.head_start = now

    def first_token(self, rec, now):
        if rec.t_first is None and not rec.done:
            self._mark_first(rec, now)

    def _drop(self, rec, now):
        self._integrate(now)
        if rec in self.pending:
            self.pending.remove(rec)
        if rec.link is not None:
            rec.link(now)
            rec.link = None
        if rec in self.fifo:
            was_head = self.fifo[0] is rec
            self.fifo.remove(rec)
            if was_head:
                self.head_start = now
        self.inflight.pop(rec.id, None)
        rec.done = True
        if self.on_free is not None:
            self.on_free()

    def abort(self, rec, now, cancelled=False, failed=False):
        if rec.done:
            return
        self._drop(rec, now)
        if cancelled:
            self.counts["cancelled"] += 1
        if failed:
            self.counts["failed"] += 1

    def finish(self, rec, now, info=None):
        """request completed.  info (all optional): prompt_tokens, completion_tokens, cached_tokens, prefill_seconds, decode_seconds"""
        if rec.done:
            return
        self._advance(now)
        first_real = rec.t_first if rec.stream else None      # a non-stream request has no observed first token
        self._integrate(now)
        k_int_now, epoch_now = self.k_int, self.admitted_total
        self._drop(rec, now)
        self.counts["ok"] += 1
        if info and self.cal["enabled"]:
            self._calibrate(rec, now, info, first_real, k_int_now, epoch_now)

    # ---- calibration ----------------------------------------------------------------------------------------------------------
    def _calibrate(self, rec, now, info, first_real, k_int_now, epoch_now):
        cal, pf = self.cal, self.profile
        pt, ct = info.get("prompt_tokens"), info.get("completion_tokens")
        cached = info.get("cached_tokens") or 0
        if cached > 0:
            self.counts["cached_hits"] += 1
        # prefill: only when the cached share is known (a backend that reports 0 while it really hits would fake a faster prefill)
        if pt and rec.path != "SM" and (cached > 0 or rec.cached_est == 0):
            rows = max(1, pt - cached)
            sec = info.get("prefill_seconds")
            if sec is None and first_real is not None and rec.queue_ahead == 0 and (pf.mode == "block" or rec.k_at_admit == 0):
                sec = first_real - rec.t_admit
            if sec and sec > 0 and rows >= cal["min_rows"]:
                self.ew_pf_tps.update(rows / sec)
                ratio = sec / max(1e-6, pf.pf_table(rows))
                s = self.ew_pf_scale.update(min(4.0, max(0.25, ratio)))
                pf.pf_scale = min(cal["max_scale"], max(cal["min_scale"], s))
        # decode: only a "clean" stream (no other request was admitted while it decoded: no stall from someone else's prefill in the sample)
        if ct and ct - 1 >= cal["min_tokens"] and rec.t_first is not None and epoch_now == rec.epoch0:
            dt = info.get("decode_seconds")
            if dt is None and first_real is not None:
                dt = now - first_real
            if dt and dt > 0.2:
                tps = (ct - 1) / dt
                span = now - rec.t_first
                kmean = max(1.0, (k_int_now - rec.k_int0) / span) if span > 1e-6 else float(max(1, rec.k_at_admit + 1))
                self.ew_dec_tps.update(tps)
                self.ew_dec_k.update(kmean)
                pred_tps = 1.0 / (interp(pf.round_pts, kmean) / 1000.0 / pf.tpe(kmean))     # table value, dec_scale = 1
                s = self.ew_dec_scale.update(min(4.0, max(0.25, tps / pred_tps)))
                pf.dec_scale = min(cal["max_scale"], max(cal["min_scale"], s))

    # ---- health / load ---------------------------------------------------------------------------------------------------------------
    def set_health(self, ok, now, live=None, slots=None):
        self.health_ok, self.health_ts = ok, now
        if live is not None:
            self.live = live
        if slots:
            self.slots_reported = slots

    def set_external(self, http_active):
        """http_active: the backend's own count of model requests in progress (/v1/load); ours are subtracted"""
        if http_active is None:
            return
        ext = max(0, int(http_active) - len(self.inflight) - len(self.shadows))
        self.ext_seen = self.ext_seen + 1 if ext > 0 else 0
        self.ext_active = ext

    def set_prefill_load(self, tokens):
        """tokens: the server's own count of prompt tokens waiting or filling (/v1/load, all clients); ours (the FIFO's rows) are subtracted"""
        if tokens is None:
            self.ext_pf_tokens = 0
            return
        self.ext_pf_tokens = max(0, int(tokens) - sum(r.rows for r in self.fifo))

    def state(self, now):
        v = self.view(now)
        pf = self.profile
        return {
            "name": self.name, "role": self.role, "url": "%s://%s:%s%s" % (self.scheme, self.host, self.port, self.base), "enabled": self.enabled,
            "breaker": self.breaker.as_dict(now), "health": {"ok": self.health_ok, "age_s": None if self.health_ts is None else round(now - self.health_ts, 1), "live": self.live},
            "inflight": len(self.inflight), "queued": v.queued, "decoding": v.k, "waiting_prefill": len(self.fifo), "backlog_s": round(v.backlog, 3),
            "slot_wait_s": round(v.slot_wait, 3), "external_active": self.ext_active,
            "ewma": {"prefill_tps": None if self.ew_pf_tps.value is None else round(self.ew_pf_tps.value, 1),
                     "decode_tps_per_stream": None if self.ew_dec_tps.value is None else round(self.ew_dec_tps.value, 2),
                     "decode_k_mean": None if self.ew_dec_k.value is None else round(self.ew_dec_k.value, 2),
                     "pf_scale": round(pf.pf_scale, 3), "dec_scale": round(pf.dec_scale, 3)},
            "counts": dict(self.counts), "slots": pf.slots, "max_context": pf.max_context,
        }
