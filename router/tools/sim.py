#!/usr/bin/env python3
"""Discrete-event simulator for DSV4.1-Flash serving on S (2x GB10 pair, CUDA) + M (M5 Ultra, Metal) + link.

python3 stdlib only.  Usage:
    python3 sim.py validate                 # model vs measured single-request numbers
    python3 sim.py tables [wl ...]          # setup x metric tables per workload (3 rates)
    python3 sim.py sens                     # sensitivity / break-even analyses
    python3 sim.py c3                       # single-request split (C3) analytic table
    python3 sim.py winners                  # winner per metric, per workload x load
    python3 sim.py limits                   # max context / slots table
    python3 sim.py all                      # everything (setups-simulation.md was generated from this; ~3.5 min)

EVERYTHING that the owner may want to change is in PARAM (one block).  Tags:
    [MEAS file]   measured, with the notes file it comes from
    [CALIB file]  derived from a measured calibration line / a fit to measured points
    [ASSUMED]     my assumption (no measurement exists)
    [INFER]       derived by reasoning / spec arithmetic
baselines.md (BaselineBench, measured 2026-11-11 on the LIVE production service and on M `--parallel 1/2/4/8`) now exists and PARAM follows it: S prefill table,
S decode round times + tokens per round at 1/2/4 streams, 2048-row chunks with a 250 ms decode window, M prefill table, M decode rounds 25/39/66 ms and
M having NO decode interleave during a prefill (running streams stop for the whole prompt), measured GPU-side power.  The older `main21ea` S profile
(spark-baseline.md) is kept as PROFILE_MAIN21EA for sensitivity.  ITL in the tables is the gap between stream EVENTS (what a client sees; S emits ~1.6-1.8 tokens per event).
Still NOT measured (assumed/derived): replay-export speed, hand-off after first-token shipping, S batching beyond 4 streams, workload shapes, wall power.
Model in one paragraph: two machines S and M, each a non-preemptive round scheduler (S: one 2048-row prefill chunk, then decode rounds for 250 ms;
M: a whole prompt prefill blocks M's decode).  A decode round advances every stream by tpe(k) tokens in step(k) seconds.  A request is placed as (prefill on P,
decode on D); P != D adds link + hand-off (M engine blocked for the landing/tail forward).  Slots are acquired on D first.  Open Poisson arrivals,
6 seeds x 900 s, 15% warm-up dropped; common random numbers across setups.  Differences below ~20% in TTFT p95 / goodput are inside seed noise.
"""
import heapq, math, random, sys, copy
from collections import deque

# =====================================================================================================================
# PARAMETER BLOCK
# =====================================================================================================================
PARAM = dict(
    # ---------------- S (two GB10 Sparks, TP=2): the LIVE production service, baselines.md (2026-11-11) -------------------
    # prefill TTFT (s) of an n-row prompt, medians.  [MEAS baselines.md s2: 1035->0.84, 4055->2.22, 8275->4.47, 16652->8.49, 32912->16.36, 67915->33.6, 134093->69.8;
    # 39-token first-token floor 0.15-0.16 s]
    S_prod_pf=[(0, 0.15), (39, 0.16), (1035, 0.84), (4055, 2.22), (8275, 4.47), (16652, 8.49), (32912, 16.36), (67915, 33.6), (134093, 69.8)],
    # export profile (whole prompt prefill + export, replicated KV).  [MEAS prefill-tuning.md / e2e-final.md "Spark prefill+export"] -- an older build, 1.0-1.2k tok/s
    S_export_pf=[(0, 0.15), (1009, 0.973), (4096, 4.05), (8192, 7.6), (16384, 14.9), (30208, 27.4), (60846, 50.8)],
    # replay-export (being implemented elsewhere): live prefill table x this factor.  [ASSUMED] 5% for decoder-ring replay + capture
    S_replay_export_mult=1.05,
    S_chunk_rows=2048,          # [MEAS baselines.md s1: prompts in 2048-row chunks]
    S_dec_window_s=0.25,        # [MEAS baselines.md s1: TF_DS_PREFILL_INTERLEAVE_MS=250 -> 250 ms of decode rounds between two chunks]
    S_pf_per_dec=None,          # (only used when S_dec_window_s is None: chunks per decode round)
    S_slots=4,                  # [MEAS baselines.md: 4 streams, a 5th queues].  Export profile has 1 slot.
    # S decode: round time (ms) for k streams = ITL p50 of the event gaps [MEAS baselines.md s4: k=1 30.8 (31), k=2 41.3, k=4 56.5];
    # k>4 [ASSUMED] = the shape of the engine verify calibration (spark-baseline.md) scaled to k=4 -- only used when S_slots>4 is swept
    S_round_pts_ms=[(1, 30.8), (2, 41.3), (4, 56.5), (8, 79.0), (16, 122.6), (32, 193.1)],
    # tokens per stream per round (draft acceptance) = agg tok/s x round / k [MEAS baselines.md: 58.2/82/110 tok/s at 1/2/4 streams -> 1.79/1.69/1.55]
    S_tpe_pts=[(1, 1.79), (2, 1.69), (4, 1.55), (32, 1.55)],
    S_export_dec_toks=34.8,     # [MEAS spark-baseline.md run A: defaults/export profile decode], 1 slot
    S_dec_speedup=1.0,          # sensitivity knob (uniform)
    # ---------------- M (M5 Ultra, Metal) --------------------------------------------------------------------------
    # whole-prompt prefill TTFT (s), medians.  [MEAS baselines.md s2 B column: 39 tok 0.72-0.78 (constant ~0.7 s first-token), 1008->1.62, 4184->4.58, 8280->8.74,
    # 16433->17.05, 33157->34.1, 65022->67.2, 131870->144.5; ~0.97k tok/s]
    M_pf_pts=[(0, 0.70), (39, 0.75), (1008, 1.62), (4184, 4.58), (8280, 8.74), (16433, 17.05), (33157, 34.1), (65022, 67.2), (131870, 144.5)],
    M_pf_b=1.0e-3,              # slope s/row used by the C3b layer-split estimate [MEAS baselines.md: 8k-33k slope 1.02 ms/row]
    # tail forward on a landed prefix: 0.55 s + 0.9 ms/row [MEAS landing-profile.md s2]
    M_tail_a=0.55, M_tail_b=0.90e-3,
    M_chunk_rows=4096,          # [MEAS baselines.md s1: TF_DSV41_PREFILL_STEP=4096]
    M_step_eff={512: 0.55, 1024: 0.86, 2048: 0.95, 4096: 1.0},   # prefill throughput vs step [MEAS metal-lane.md: 575/890/1050 tok/s]
    # M decode streams STOP for the whole prefill of another request (9.8 s for a 9k prompt) [MEAS baselines.md s5] -> prefill has absolute priority
    M_pf_per_dec=10 ** 9,
    M_slots=4,                  # [MEAS --parallel 4; 8 slots add nothing (59 tok/s)]
    M_dec_k1_toks=39.4,         # [MEAS baselines.md s4: 39.1-39.4 tok/s (round 25 ms); drafts switch themselves off]
    M_dec_k4_agg=60.6,          # [MEAS baselines.md s4: round 66 ms at 4 streams (steady 59.8-60.6 tok/s); 39 ms at 2; 104 at 6; 135 at 8]
    M_dec_speedup=1.0,          # sensitivity: uniform speed-up of M's overhead-bound decode (2.0 => k=1 80 tok/s)
    M_row_ms=None,              # sensitivity: if set, per-extra-stream cost (ms) of M's batched decode (measured fit: 13.5)
    M_tpe_pts=[(1, 1.0), (32, 1.0)],   # tokens per stream per decode round on M.  [MEAS baselines.md: 1.0, its drafts switch off after 8 rounds]; sens: drafts working = S's table
    # ---------------- decode roofline (min(roofline, overhead)) ----------------------------------------------------
    # bytes per token step as STORED in pack/Mia-2.9 (safetensors headers): 8.06 GB = 3.06 routed experts (6 of 384, 40 layers) + 5.0 dense
    # (shared 0.80 + attention 3.21 + head 0.50 + gate/hc/engram-wkv 0.49).  [MEAS-HEADERS setup-survey.md s3.1; study-decode-roofline.md said 6.9 GB]
    B_dense_GB=5.0, B_exp_GB=3.06, n_experts=384, topk=6,
    S_BW_GBs=2 * 273.0,         # [SPEC NVIDIA DGX Spark user guide: 273 GB/s per Spark] TP=2 reads half the bytes on each
    M_BW_GBs=1228.8,            # [SPEC via setup-survey.md: Apple Support 128107, M5 Ultra 80-core GPU 1.2 TB/s]; I could not verify it independently (apple.com shows M5 Max 460 GB/s)
    bw_eff=0.85,                # [ASSUMED] fraction of peak a perfect kernel sustains
    # ---------------- limits ---------------------------------------------------------------------------------------
    S_ctx=1048576,              # [MEAS baselines.md s6: config = model window (served up to 134k tokens, 4 streams share the pool)]
    S_export_pool=65536,        # [MEAS prefill-tuning.md: validated 65,536; 60,846-token export ok]
    M_ctx=40960,                # [MEAS e2e-final.md, baselines.md: 4 slots tested at 40,960; `--context 262144 --parallel 1` booted and served 147k; KV is 1402 B/token, memory is not the limit]
    # ---------------- link and hand-off ----------------------------------------------------------------------------
    kb_per_tok=3.7,             # [MEAS e2e-final.md cable check 3.7-3.9 KB/token]
    link_gbps=29.0,             # [MEAS payload ceiling writes; 50 reads; task: 3-29]
    link_overlap=0.0,           # fraction of the transfer hidden behind S prefill (streaming; 0 = measured pull-after-export)
    # hand-off scenarios: link_extra = HTTP+queue+pull latency off M's engine; land = M engine time (landing fill + tail forward)
    H_now=dict(link_extra=0.25, land_fixed=0.04, land_per_tok=1.1e-5, tail='grid512'),   # [MEAS e2e-final.md "other" 0.1-0.55, M land+tail 0.62-1.06]
    H_ship1=dict(link_extra=0.15, land_fixed=0.04, land_per_tok=1.1e-5, tail='one'),     # [CALIB firsttok-design.md: tail=1 -> one ~45 ms step]; total ~0.3 s
    H_best=dict(link_extra=0.05, land_fixed=0.02, land_per_tok=0.6e-5, tail='one'),      # [ASSUMED] ~0.2 s total (replay + shipping + streaming)
    one_step_s=0.045,           # decode-class one-row step on M (25-45 ms) [MEAS firsttok-design.md]
    remote_min=1024,            # minimum prompt for a remote prefill. Today's TF_PREFILL_MIN default is 4096 [MEAS study-handoff.md]; 1024 = the pipeline the owner asked about
    # ---------------- C3: split prefill of one request across S and M ------------------------------------------------
    c3_lag_frac=0.10,           # M may start c3_lag_frac * (S part duration) after S starts. 1.0 = chunk-major (no overlap), ~1/20 = layer-skewed
    c3_min_rows=4096, c3_tail_s=0.15, c3_land_s=0.20,
    # C3b layer-split chunk pipeline (study-chunk-pipeline.md B): S runs the first c3b_f of the encoder layers on every segment, M the rest + exact cone
    c3b_f=0.70,                 # [INFER study-chunk-pipeline.md: balanced cut 14/20]
    c3b_cone_s=0.7,             # M exact decoder cone [INFER study-chunk-pipeline.md]
    c3b_seg_rows=4096, c3b_min_rows=8192,
    # ---------------- policy ----------------------------------------------------------------------------------------
    w_int=1.0,                  # weight of the delay imposed on other streams in the adaptive policies' cost
    len_thresh=4096,            # A+B length router
    # ---------------- power (only what is documented) --------------------------------------------------------------
    # measured GPU-side power [MEAS baselines.md s7: nvidia-smi per Spark x2; powermetrics GPU on M]; wall power NOT measured
    W_S_idle=25.0, W_S_pf=156.0, W_S_dec=135.0,     # S pair; decode 126 W (1 stream) .. 142 W (4 streams) -> 135 [INFER]
    W_M_idle=36.5, W_M_pf=138.0, W_M_dec=104.0,
)


# the OLDER S profile (spark-baseline.md main21ea: 1.60k/2.16k/2.53k tok/s, 45.5 tok/s, 4096-row pieces 1:1 with one decode step) for sensitivity
PROFILE_MAIN21EA = dict(
    S_prod_pf=[(0, 0.10), (64, 0.20), (1009, 0.629), (8111, 3.749), (32427, 12.834), (128912, 53.1)],
    S_chunk_rows=4096, S_dec_window_s=None, S_pf_per_dec=1,
    S_round_pts_ms=[(k, v * 22.0 / 21.5) for k, v in [(1, 21.5), (2, 25.8), (4, 35.3), (6, 42.7), (8, 49.6), (16, 76.9), (24, 100.8), (32, 118.6)]],
    S_tpe_pts=[(1, 1.0), (32, 1.0)],
)


def prm_with(**kw):
    p = copy.deepcopy(PARAM)
    p.update(kw)
    return p


def interp(pts, x):
    if x <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    return y1 + (y1 - y0) / (x1 - x0) * (x - x1)


# =====================================================================================================================
# Decode model: step_s(k) = max(bandwidth roofline(k), overhead-bound(k));  k streams x 1 row, one token each
# =====================================================================================================================
def uniq(prm, k):
    """expected distinct routed experts per layer for k independent tokens, in units of one token's 6 experts (upper bound on bytes)"""
    n, t = prm['n_experts'], prm['topk']
    return n * (1 - (1 - t / n) ** k) / t


def roof_s(prm, k, bw):
    gb = prm['B_dense_GB'] + prm['B_exp_GB'] * uniq(prm, k)
    return gb / (bw * prm['bw_eff'])


def tpe_S(prm, k):
    return max(1.0, interp(prm['S_tpe_pts'], k))


def step_S(prm, k, export_profile=False):
    """S decode ROUND time for k streams (one round delivers tpe_S(k) tokens per stream)"""
    if export_profile:
        return 1.0 / prm['S_export_dec_toks']
    over = interp(prm['S_round_pts_ms'], k) / 1000.0 / prm['S_dec_speedup']
    # the measured table already contains the bandwidth cost; the floor is the 1-row weight-read roofline
    return max(over, roof_s(prm, 1, prm['S_BW_GBs']))


def M_over_coeffs(prm):
    one, four = 1.0 / prm['M_dec_k1_toks'], 4.0 / prm['M_dec_k4_agg']
    row = (four - one) / 3.0
    if prm['M_row_ms'] is not None:
        row = prm['M_row_ms'] / 1000.0
    over = max(one - row, 0.0)
    return over / prm['M_dec_speedup'], row / prm['M_dec_speedup']


def step_M(prm, k):
    a, b = M_over_coeffs(prm)
    return max(a + b * k, roof_s(prm, k, prm['M_BW_GBs']))


def tail_rows(prm, H, n):
    return 1 if H['tail'] == 'one' else ((n - 1) % 512) + 1


# =====================================================================================================================
# DES core
# =====================================================================================================================
class Job:
    __slots__ = ('kind', 'req', 'rem', 'total', 'chunk_s', 'not_before', 'on_done', 'on_start', 'started', 'tag')

    def __init__(self, kind, req, total, chunk_s, on_done, not_before=0.0, on_start=None, tag=None):
        self.kind, self.req, self.rem, self.total, self.chunk_s = kind, req, total, chunk_s, chunk_s
        self.chunk_s = chunk_s
        self.not_before, self.on_done, self.on_start, self.started, self.tag = not_before, on_done, on_start, None, tag


class Machine:
    def __init__(m, sim, name, slots, chunk_rows, pf_per_dec, step_fn, pf_fn, export_profile=False, can_decode=True, tpe_fn=None, dec_window=None):
        m.sim, m.name, m.slots, m.chunk_rows, m.pf_per_dec = sim, name, slots, chunk_rows, (pf_per_dec or 1)
        m.step_fn, m.pf_fn, m.export_profile, m.can_decode = step_fn, pf_fn, export_profile, can_decode
        m.tpe_fn, m.dec_window, m.win_left = tpe_fn or (lambda k: 1.0), dec_window, 0.0
        m.queue, m.active, m.slots_used, m.admit_q = [], [], 0, deque()
        m.busy, m.pf_since_dec = False, 0
        m.busy_time = m.pf_time = m.dec_time = 0.0
        m.tokens_out = 0

    def step_s(m, k):
        return m.step_fn(k)

    def tok_s(m, k):
        """seconds per token per stream with k streams active"""
        return m.step_fn(k) / m.tpe_fn(k)

    def infl(m, k, nch):
        """time the decode interleave adds to one of this machine's own prefills (k streams live)"""
        if k <= 0:
            return 0.0
        if m.dec_window is not None:
            return nch * m.dec_window
        return nch * m.step_s(k) * m.pf_per_dec if m.pf_per_dec < 1e6 else 0.0

    def pf_seconds(m, rows, export=False):
        """(total seconds, chunk seconds) of a prompt pass of `rows` rows"""
        tot = m.pf_fn(rows, export)
        if not m.chunk_rows or rows <= m.chunk_rows:
            return tot, tot
        return tot, tot * m.chunk_rows / rows        # full chunks at the average row rate; the last chunk is the remainder

    def backlog(m):
        return sum(j.rem for j in m.queue)

    def slot_wait(m):
        if m.slots_used < m.slots:
            return 0.0
        rem = [r.n_out - r.tokens for r in m.active]
        return (min(rem) if rem else 0) * m.tok_s(max(1, len(m.active)))

    # slots ----------------------------------------------------------------------------------------------------
    def acquire(m, cont):
        if m.slots_used < m.slots:
            m.slots_used += 1
            cont()
        else:
            m.admit_q.append(cont)

    def release(m):
        m.slots_used -= 1
        if m.admit_q:
            m.slots_used += 1
            m.admit_q.popleft()()

    # queue ----------------------------------------------------------------------------------------------------
    def add_job(m, job, front=False):
        if front:
            m.queue.insert(0, job)
        else:
            m.queue.append(job)
        m.kick()

    def kick(m):
        if not m.busy:
            m.start_round()

    def start_round(m):
        if m.busy:
            return
        s = m.sim
        now = s.t
        job = None
        for j in m.queue:
            if j.not_before <= now + 1e-12:
                job = j
                break
        k = len(m.active)
        go_pf = job is not None and (k == 0 or not m.can_decode or (m.win_left <= 1e-12 if m.dec_window is not None else m.pf_since_dec < m.pf_per_dec))
        if go_pf:
            d = min(job.chunk_s, job.rem)
            if job.started is None:
                job.started = now
                if job.on_start:
                    job.on_start(now)
            m.busy = True
            s.at(now + d, m.end_round, 'pf', job, d)
        elif k > 0:
            d = m.step_s(k)
            m.busy = True
            s.at(now + d, m.end_round, 'dec', None, d)
        else:
            nb = [j.not_before for j in m.queue if j.not_before > now]
            if nb:
                s.at(min(nb), m.kick)

    def end_round(m, kind, job, d):
        s = m.sim
        m.busy = False
        m.busy_time += d
        if kind == 'pf':
            m.pf_since_dec += 1
            m.pf_time += d
            m.win_left = m.dec_window or 0.0
            job.rem -= d
            if job.rem <= 1e-9:
                m.queue.remove(job)
                job.on_done(s.t)
        else:
            m.pf_since_dec = 0
            m.dec_time += d
            m.win_left -= d
            t = s.t
            tp = m.tpe_fn(len(m.active))
            done = []
            for r in m.active:
                r.acc += tp
                whole = int(r.acc + 1e-9)
                r.acc -= whole
                nt = min(whole, r.n_out - r.tokens)
                r.tokens += nt
                r.gaps.append(t - r.last_tok)
                r.last_tok = t
                m.tokens_out += nt
                if r.tokens >= r.n_out:
                    done.append(r)
            for r in done:
                m.active.remove(r)
                s.finish(r, m)
        m.start_round()

    def first_token(m, r):
        s = m.sim
        r.t_first = s.t
        r.tokens = 1
        r.last_tok = s.t
        m.tokens_out += 1
        if r.tokens >= r.n_out:
            s.finish(r, m)
        else:
            m.active.append(r)
            m.kick()


class Req:
    def __init__(r, rid, t, n_in, n_out, cls, conv=None, turn=0):
        r.id, r.t_arr, r.n_in, r.n_out, r.cls, r.conv, r.turn = rid, t, n_in, n_out, cls, conv, turn
        r.t_first = None
        r.t_done = None
        r.tokens = 0
        r.gaps = []
        r.acc = 0.0
        r.last_tok = 0.0
        r.P = r.D = None
        r.plan = None


class Conv:
    def __init__(c, cid, turns, thinks):
        c.id, c.turns, c.thinks = cid, turns, thinks      # turns: [(n_in, n_out)], thinks: seconds before next turn
        c.held = {'S': 0, 'M': 0}


class Sim:
    def __init__(s, prm, setup, reqs, convs, horizon, seed=0):
        s.prm, s.setup, s.t, s.heap, s.seq = prm, setup, 0.0, [], 0
        s.horizon = horizon
        s.reqs = []
        s.rr = 0
        s.rng = random.Random(seed + 777)
        prof = setup['S_profile']
        exp = prof == 'export'
        s.Sprof = prof
        s.M = Machine(s, 'M', prm['M_slots'], prm['M_chunk_rows'] or 0, prm['M_pf_per_dec'],
                      lambda k: step_M(prm, k), s._pf_M, tpe_fn=lambda k: max(1.0, interp(prm['M_tpe_pts'], k)))
        s.S = Machine(s, 'S', 1 if exp else prm['S_slots'], 0 if exp else prm['S_chunk_rows'], prm['S_pf_per_dec'],
                      (lambda k: step_S(prm, k, True)) if exp else (lambda k: step_S(prm, k)), s._pf_S,
                      export_profile=exp, can_decode=not exp, tpe_fn=None if exp else (lambda k: tpe_S(prm, k)),
                      dec_window=None if exp else prm['S_dec_window_s'])
        s.mach = {'S': s.S, 'M': s.M}
        s.H = prm[setup.get('H', 'H_now')]
        for r in reqs:
            s.reqs.append(r)
            s.at(r.t_arr, s.arrive, r)
        for c in convs:
            r = c.first
            s.reqs.append(r)
            s.at(r.t_arr, s.arrive, r)

    # ------------------------------------------------------------------ time
    def at(s, t, fn, *a):
        s.seq += 1
        heapq.heappush(s.heap, (t, s.seq, fn, a))

    def run(s, t_end):
        while s.heap:
            t, _, fn, a = heapq.heappop(s.heap)
            if t > t_end:
                break
            s.t = t
            fn(*a)

    # ------------------------------------------------------------------ prefill time models
    def _pf_S(s, rows, export):
        p = s.prm
        if s.Sprof == 'export':
            return interp(p['S_export_pf'], rows)
        t = interp(p['S_prod_pf'], rows)
        return t * p['S_replay_export_mult'] if export else t

    def _pf_M(s, rows, export):
        p = s.prm
        eff = interp([(k, v) for k, v in sorted(p['M_step_eff'].items())], p['M_chunk_rows'] or 4096)
        return interp(p['M_pf_pts'], rows) / eff

    # ------------------------------------------------------------------ cache
    def cached(s, r, name):
        if r.conv is None:
            return 0
        return min(r.conv.held[name], r.n_in - 1)

    # ------------------------------------------------------------------ feasibility
    def feasible(s, r, P, D):
        p = s.prm
        n = r.n_in + r.n_out
        ctx = {'S': p['S_ctx'], 'M': p['M_ctx']}
        if n > ctx[D] or r.n_in > ctx[P]:
            return False
        if P == 'S' and D == 'M' and r.n_in > p['S_export_pool']:
            return False
        if D == 'S' and s.Sprof == 'export':
            return False
        if P == 'M' and D == 'S':
            return False          # needs a CUDA import path: not implemented (kept off in every setup)
        return True

    # ------------------------------------------------------------------ arrival / routing
    def arrive(s, r):
        r.t_arr = s.t
        plan = s.setup['policy'](s, r)
        r.plan = plan
        if plan[0] in ('split', 'lsplit'):
            s.mach[plan[2]].acquire(lambda: s.begin_split(r, plan))
        else:
            P, D = plan
            r.P, r.D = P, D
            s.mach[D].acquire(lambda: s.begin(r, P, D))

    def link_s(s, ntok):
        p = s.prm
        return ntok * p['kb_per_tok'] * 1e3 * 8 / (p['link_gbps'] * 1e9) * (1 - p['link_overlap'])

    def land_s(s, n, tail):
        p, H = s.prm, s.H
        t = H['land_fixed'] + H['land_per_tok'] * n
        t += p['one_step_s'] if tail <= 1 else p['M_tail_a'] + p['M_tail_b'] * tail
        return t

    def remote_split(s, r, rows):
        """(rows the producer computes, tail rows the decoder computes) for a remote prefix"""
        tail = tail_rows(s.prm, s.H, r.n_in)
        return max(1, rows - tail), tail

    def begin(s, r, P, D):
        Pm, Dm = s.mach[P], s.mach[D]
        rows = max(1, r.n_in - s.cached(r, P))
        if P == D:
            tot, ch = Pm.pf_seconds(rows)
            Pm.add_job(Job('pf', r, tot, ch, lambda t: Dm.first_token(r)))
        else:
            rp, tail = s.remote_split(r, rows)
            tot, ch = Pm.pf_seconds(rp, export=True)

            def exported(t):
                land = s.land_s(r.n_in, tail)
                dt = s.H['link_extra'] + s.link_s(r.n_in)
                s.at(t + dt, lambda: Dm.add_job(Job('land', r, land, land, lambda t2: Dm.first_token(r)), front=True))
            Pm.add_job(Job('pf', r, tot, ch, exported))

    def begin_split(s, r, plan):
        """one request prefilled on S and M at once.  plan = ('split', x, D) row split (S takes the first x of the rows, M starts c3_lag_frac*T_S
        after S starts) or ('lsplit', f, 'M') layer-split chunk pipeline (every row on both: S the first f of the layers, M the rest + exact cone)."""
        p = s.prm
        kind, x, D = plan
        n = r.n_in
        if kind == 'split':
            rows_S = max(1, int(n * x))
            totS, chS = s.S.pf_seconds(rows_S, export=True)
            totM, chM = s.M.pf_seconds(max(1, n - rows_S))
            lag_of = lambda tS: p['c3_lag_frac'] * tS
        else:
            tS, _ = s.S.pf_seconds(n, export=True)
            nseg = max(1, math.ceil(n / p['c3b_seg_rows']))
            totS, chS = x * tS, x * tS / nseg
            totM = (1 - x) * p['M_pf_b'] * n + p['c3b_cone_s']
            chM = totM / nseg
            lag_of = lambda tS_: x * interp(p['S_prod_pf'], min(n, p['c3b_seg_rows'])) * p['S_replay_export_mult']
        r.P, r.D = 'S+M', D
        st = {'S': None, 'M': None}
        Dm = s.mach[D]

        def joined():
            if st['S'] is None or st['M'] is None:
                return
            tail = p['c3_tail_s']
            t0 = max(st['S'] + (tail if D == 'M' else 0.0), st['M'] + (tail if D == 'S' else 0.0), s.t)
            land = p['c3_land_s']
            s.at(t0, lambda: Dm.add_job(Job('land', r, land, land, lambda t2: Dm.first_token(r)), front=True))

        def done(who):
            def f(t):
                st[who] = t
                joined()
            return f
        jM = Job('pf', r, totM, chM, done('M'), not_before=1e18)

        def startS(t):
            jM.not_before = t + lag_of(totS)
            s.at(jM.not_before, s.M.kick)
        s.S.add_job(Job('pf', r, totS, chS, done('S'), on_start=startS))
        s.M.add_job(jM)

    # ------------------------------------------------------------------ completion
    def finish(s, r, Dm):
        r.t_done = s.t
        Dm.release()
        c = r.conv
        if c is not None:
            n = r.n_in + r.n_out
            if r.P == 'S+M':
                c.held['S'] = c.held['M'] = 0
                c.held[r.D] = n
            else:
                c.held[r.D] = n
                if r.P != r.D:
                    c.held[r.P] = max(c.held[r.P], r.n_in)
            nxt = r.turn + 1
            if nxt < len(c.turns):
                n_in, n_out = c.turns[nxt]
                r2 = Req((c.id, nxt), s.t + c.thinks[r.turn], n_in, n_out, r.cls, c, nxt)
                s.reqs.append(r2)
                s.at(s.t + c.thinks[r.turn], s.arrive, r2)


# =====================================================================================================================
# Prediction + policies
# =====================================================================================================================
def predict(s, r, P, D, w_int):
    """predicted cost (request-seconds) of placing r with prefill on P and decode on D: own E2E + delay imposed on D's streams"""
    p = s.prm
    Pm, Dm = s.mach[P], s.mach[D]
    rows = max(1, r.n_in - s.cached(r, P))
    k = len(Dm.active)
    if P == D:
        tot, ch = Pm.pf_seconds(rows)
        nch = max(1, tot / ch)
        infl = Dm.infl(k, nch)
        ttft = max(Dm.slot_wait(), Pm.backlog()) + tot + infl
        stall = tot
    else:
        rp, tail = s.remote_split(r, rows)
        tot, ch = Pm.pf_seconds(rp, export=True)
        land = s.land_s(r.n_in, tail)
        ttft = max(Dm.slot_wait(), Pm.backlog()) + tot + s.H['link_extra'] + s.link_s(r.n_in) + Dm.backlog() + land
        stall = land
    step1 = Dm.tok_s(k + 1)
    e2e = ttft + (r.n_out - 1) * step1
    interf = k * stall + k * (step1 - Dm.tok_s(max(k, 1))) * r.n_out
    return e2e + w_int * interf, ttft, e2e


def predict_split(s, r, D, w_int):
    """best split fraction x (S takes the first x of the rows) and predicted cost."""
    p = s.prm
    n = r.n_in
    S_, M_ = s.S, s.M
    bS, bM = S_.backlog(), M_.backlog()

    def fin(x):
        rs, rm = max(1, int(n * x)), max(1, n - int(n * x))
        tS = S_.pf_seconds(rs, export=True)[0]
        tM = M_.pf_seconds(rm)[0]
        fS = bS + tS
        fM = max(bM, bS + p['c3_lag_frac'] * tS) + tM
        return fS, fM, tS, tM
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        fS, fM, _, _ = fin(mid)
        if fS < fM:
            lo = mid
        else:
            hi = mid
    x = (lo + hi) / 2
    fS, fM, tS, tM = fin(x)
    ready = max(fS + p['c3_tail_s'], fM) + p['c3_land_s']
    Dm = s.mach[D]
    k = len(Dm.active)
    ttft = max(ready, Dm.slot_wait())
    step1 = Dm.tok_s(k + 1)
    e2e = ttft + (r.n_out - 1) * step1
    interf = len(S_.active) * tS + len(M_.active) * tM + k * (step1 - Dm.tok_s(max(k, 1))) * r.n_out
    return x, e2e + w_int * interf, ttft


def predict_lsplit(s, r, w_int):
    p = s.prm
    n, f = r.n_in, p['c3b_f']
    tS = f * s.S.pf_seconds(n, export=True)[0]
    tM = (1 - f) * p['M_pf_b'] * n + p['c3b_cone_s']
    lag = f * interp(p['S_prod_pf'], min(n, p['c3b_seg_rows'])) * p['S_replay_export_mult']
    bS, bM = s.S.backlog(), s.M.backlog()
    fS = bS + tS
    fM = max(bM, bS + lag) + tM
    ready = max(fS + p['c3_tail_s'], fM) + p['c3_land_s']
    Dm = s.M
    k = len(Dm.active)
    ttft = max(ready, Dm.slot_wait())
    step1 = Dm.tok_s(k + 1)
    e2e = ttft + (r.n_out - 1) * step1
    interf = len(s.S.active) * tS + k * tM + k * (step1 - Dm.tok_s(max(k, 1))) * r.n_out
    return e2e + w_int * interf, ttft


def make_choose(cands, split=None, with_len_min=False, local_cap=None):
    """cost-based placement among candidate (P,D) pairs"""
    def policy(s, r):
        w = s.prm['w_int']
        best, bestc = None, 1e30
        for P, D in cands:
            if not s.feasible(r, P, D):
                continue
            if P != D and r.n_in < s.prm['remote_min'] and with_len_min:
                continue
            if P == D == 'M' and local_cap is not None and r.n_in - s.cached(r, 'M') > local_cap and s.feasible(r, 'S', 'M'):
                continue
            c, _, _ = predict(s, r, P, D, w)
            if c < bestc:
                best, bestc = (P, D), c
        if split in ('row', 'layer') and s.Sprof != 'export':
            n = r.n_in
            hit = max(s.cached(r, 'S'), s.cached(r, 'M'))
            fits_M = n + r.n_out <= s.prm['M_ctx']
            if split == 'row' and n >= s.prm['c3_min_rows'] and hit < 0.5 * n and n <= s.prm['S_export_pool']:
                for D in ('M', 'S'):
                    if D == 'M' and not fits_M:
                        continue
                    x, c, _ = predict_split(s, r, D, w)
                    if c < bestc:
                        best, bestc = ('split', x, D), c
            if split == 'layer' and n >= s.prm['c3b_min_rows'] and hit < 0.5 * n and n <= s.prm['S_export_pool'] and fits_M:
                c, _ = predict_lsplit(s, r, w)
                if c < bestc:
                    best, bestc = ('lsplit', s.prm['c3b_f'], 'M'), c
        return best if best is not None else ('S', 'S')
    return policy


def pol_fixed(P, D):
    def policy(s, r):
        return (P, D) if s.feasible(r, P, D) else (('S', 'S') if s.feasible(r, 'S', 'S') else (P, D))
    return policy


def pol_rr(s, r):
    s.rr ^= 1
    P = 'S' if s.rr else 'M'
    return (P, P) if s.feasible(r, P, P) else ('S', 'S')


def pol_ll(s, r):
    """two independent servers; least predicted TTFT (cache aware)"""
    best, bt = None, 1e30
    for m in ('S', 'M'):
        if not s.feasible(r, m, m):
            continue
        _, ttft, _ = predict(s, r, m, m, 0.0)
        if ttft < bt:
            best, bt = (m, m), ttft
    return best or ('S', 'S')


def pol_len(s, r):
    P = 'S' if r.n_in >= s.prm['len_thresh'] else 'M'
    return (P, P) if s.feasible(r, P, P) else ('S', 'S')



def pol_overflow(s, r):
    """'A for everything, M only for overflow': everything goes to S while S has a free decode slot (and the request fits S);
    when S is slot-full M takes it if it fits M (a cost-free, signal-free rule: no TTFT prediction)."""
    if s.feasible(r, 'S', 'S') and s.S.slots_used < s.S.slots:
        return ('S', 'S')
    if s.feasible(r, 'M', 'M') and s.M.slots_used < s.M.slots:
        return ('M', 'M')
    return ('S', 'S')


def pol_c(s, r):
    """today's C: remote prefill when n >= TF_PREFILL_MIN, otherwise M prefills locally. No producer-busy fallback."""
    if r.n_in >= s.prm['remote_min'] and s.feasible(r, 'S', 'M'):
        return ('S', 'M')
    return ('M', 'M') if s.feasible(r, 'M', 'M') else ('S', 'S')


def pol_c4(s, r):
    """class based: interactive -> decode on S (low ITL), batch-like -> decode on M (S frees up for prefill)."""
    inter = r.cls in ('chat', 'agent')
    order = [('S', 'S'), ('S', 'M'), ('M', 'M')] if inter else [('S', 'M'), ('S', 'S'), ('M', 'M')]
    for P, D in order:
        if not s.feasible(r, P, D):
            continue
        if D == 'S' and s.S.slots_used >= s.S.slots and s.M.slots_used < s.M.slots:
            continue
        if D == 'M' and s.M.slots_used >= s.M.slots and s.S.slots_used < s.S.slots:
            continue
        if P == 'S' and D == 'M' and r.n_in < s.prm['remote_min']:
            continue
        return (P, D)
    return ('S', 'S')


SETUPS = {
    # name: dict(policy, S_profile, H)
    'A  S alone':                 dict(policy=pol_fixed('S', 'S'), S_profile='prod'),
    'B  M alone':                 dict(policy=pol_fixed('M', 'M'), S_profile='prod'),
    'A+B round-robin':            dict(policy=pol_rr, S_profile='prod'),
    'A+B least-loaded':           dict(policy=pol_ll, S_profile='prod'),
    'A+B length (>=4k->S)':       dict(policy=pol_len, S_profile='prod'),
    'A+B overflow (M only if S slots full)': dict(policy=pol_overflow, S_profile='prod'),
    'A+B cost-aware (no remote)': dict(policy=make_choose([('S', 'S'), ('M', 'M')], local_cap=None), S_profile='prod'),
    'C  now (export 1.1k, H=1.0)': dict(policy=pol_c, S_profile='export', H='H_now'),
    'C  replay+ship1 (H=0.3)':    dict(policy=pol_c, S_profile='replay', H='H_ship1'),
    'C2 conditional disagg':      dict(policy=make_choose([('S', 'M'), ('M', 'M')], local_cap=4096), S_profile='replay', H='H_ship1'),
    'C5 S+M, S also decodes':     dict(policy=make_choose([('S', 'S'), ('S', 'M'), ('M', 'M')]), S_profile='replay', H='H_ship1'),
    'C4 class (chat->S dec)':     dict(policy=pol_c4, S_profile='replay', H='H_ship1'),
    'C3a C5 + row split (layer-major)': dict(policy=make_choose([('S', 'S'), ('S', 'M'), ('M', 'M')], split='row'), S_profile='replay', H='H_ship1'),
    'C3b C5 + layer-split pipeline': dict(policy=make_choose([('S', 'S'), ('S', 'M'), ('M', 'M')], split='layer'), S_profile='replay', H='H_ship1'),
}


# =====================================================================================================================
# Workloads
# =====================================================================================================================
def U(rng, a, b):
    return int(rng.uniform(a, b))


def gen_oneshot(rng, rate, horizon, mix):
    """mix: list of (weight, cls, n_in range, n_out range)"""
    reqs, t, i = [], 0.0, 0
    w = [m[0] for m in mix]
    while True:
        t += rng.expovariate(rate)
        if t > horizon:
            break
        m = rng.choices(mix, w)[0]
        reqs.append(Req(i, t, U(rng, *m[2]), U(rng, *m[3]), m[1]))
        i += 1
    return reqs, []


def gen_agent(rng, rate, horizon):
    convs, t, i = [], 0.0, 0
    while True:
        t += rng.expovariate(rate)
        if t > horizon:
            break
        T = rng.randint(6, 12)
        n = U(rng, 3000, 6000)
        turns, th = [], []
        for j in range(T):
            o = U(rng, 100, 300)
            turns.append((n, o))
            th.append(rng.uniform(1.0, 4.0))
            n = n + o + U(rng, 300, 1500)
        c = Conv(i, turns, th)
        c.first = Req((i, 0), t, turns[0][0], turns[0][1], 'agent', c, 0)
        convs.append(c)
        i += 1
    return [], convs


def gen_batch(rng, rate, horizon):
    return [Req(i, 0.0, U(rng, 8000, 16000), 400, 'batch') for i in range(24)], []


WORKLOADS = {
    # name: (generator, [3 loads: req/s (agent: conversations/s)], description, SLO (TTFT s, per-request ITL p95 s))
    'chat':    (lambda rng, rate, H: gen_oneshot(rng, rate, H, [(1, 'chat', (1500, 2500), (256, 768))]), [0.06, 0.12, 0.20], 'chat 2k in / 512 out', (2.0, 0.100)),
    'rag':     (lambda rng, rate, H: gen_oneshot(rng, rate, H, [(1, 'rag', (12000, 20000), (200, 300))]), [0.03, 0.06, 0.09], 'RAG 16k in / 256 out', (10.0, 0.100)),
    'agent':   (gen_agent, [0.02, 0.05, 0.09], 'agent loop (rate = conversations/s; 6-12 turns, ctx 3-6k growing +0.5-1.8k/turn, think 1-4 s, prefix hits)', (5.0, 0.100)),
    'longdoc': (lambda rng, rate, H: gen_oneshot(rng, rate, H, [(1, 'long', (32000, 40000), (200, 300))]), [0.02, 0.04, 0.06], 'long-doc 32-40k in / 256 out (<= M ctx 40,960)', (20.0, 0.100)),
    'mixed':   (lambda rng, rate, H: gen_oneshot(rng, rate, H, [(0.6, 'chat', (1500, 2500), (256, 768)), (0.25, 'rag', (12000, 20000), (200, 300)),
                                                                (0.1, 'long', (28000, 38000), (200, 300)), (0.05, 'short', (200, 800), (64, 256))]),
                [0.06, 0.10, 0.14], 'mixed 60% chat 2k/512, 25% RAG 16k/256, 10% long 33k/256, 5% short', (10.0, 0.100)),
    'batch':   (gen_batch, [0.0], 'batch summarisation: 24 docs of 8-16k in / 400 out, all at t=0', (1e9, 1e9)),
}

SEEDS = (1, 2, 3, 4, 5, 6)
HORIZON = 900.0
DRAIN = 600.0


def pct(xs, q):
    if not xs:
        return float('nan')
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(math.ceil(q / 100 * len(xs))) - 1))
    return xs[i]


def run_one(wl, rate, setup, prm, seed, horizon=None):
    horizon = horizon or HORIZON
    gen = WORKLOADS[wl][0]
    rng = random.Random(seed)
    reqs, convs = gen(rng, rate, horizon)
    sim = Sim(prm, setup, reqs, convs, horizon, seed)
    end = horizon + DRAIN if wl != 'batch' else 3 * 3600
    sim.run(end)
    return sim


def summarize(sims, wl, horizon=None, slo=None):
    slo = slo or WORKLOADS[wl][3]
    horizon = horizon or HORIZON
    """pool the seeds; window = arrivals after 15% of the horizon"""
    ttft, e2e, gaps, maxg, ok_n, done_n, tot_n = [], [], [], [], 0, 0, 0
    out_tok = in_tok = 0
    span = 0.0
    sb = {'S': 0.0, 'M': 0.0}
    stall2 = 0
    for sim in sims:
        w0 = 0.15 * horizon if wl != 'batch' else 0.0
        endt = max([r.t_done or 0 for r in sim.reqs] + ([horizon] if wl != 'batch' else [0.0]))
        span += (horizon - w0) if wl != 'batch' else max(r.t_done or 0 for r in sim.reqs)
        for r in sim.reqs:
            if r.t_arr < w0 or r.t_arr > horizon:
                continue
            tot_n += 1
            if r.t_done is None:
                continue
            done_n += 1
            ttft.append(r.t_first - r.t_arr)
            e2e.append(r.t_done - r.t_arr)
            gaps.extend(r.gaps)
            if r.gaps:
                maxg.append(max(r.gaps))
                stall2 += max(r.gaps) > 1.0
            out_tok += r.n_out
            in_tok += r.n_in
            if r.t_first - r.t_arr <= slo[0] and (not r.gaps or pct(r.gaps, 95) <= slo[1]):
                ok_n += 1
        for k in sb:
            sb[k] += sim.mach[k].busy_time / max(endt, 1e-9)
    n = len(sims)
    res = dict(
        fin=done_n / max(1, tot_n), n=tot_n,
        ttft50=pct(ttft, 50), ttft95=pct(ttft, 95),
        itl50=pct(gaps, 50) * 1000, itl95=pct(gaps, 95) * 1000, itl99=pct(gaps, 99) * 1000, maxg95=pct(maxg, 95),
        stall2=stall2 / max(1, done_n),
        e2e50=pct(e2e, 50), e2e95=pct(e2e, 95),
        out_tps=out_tok / max(span, 1e-9), in_tps=in_tok / max(span, 1e-9), good=ok_n / max(span, 1e-9) * 60,
        utilS=sb['S'] / n, utilM=sb['M'] / n,
        makespan=span / n,
    )
    return res


def evaluate(wl, rate, setup, prm=None, seeds=None, slo=None):
    prm = prm or PARAM
    seeds = seeds or SEEDS
    slo = slo or WORKLOADS[wl][3]
    sims = [run_one(wl, rate, setup, prm, sd) for sd in seeds]
    return summarize(sims, wl, slo=slo)


# =====================================================================================================================
# Reports
# =====================================================================================================================
def fmt(v, f='{:.1f}'):
    return 'nan' if v != v else f.format(v)


COLS = [('TTFT p50', 'ttft50', '{:.1f}'), ('TTFT p95', 'ttft95', '{:.1f}'), ('ITL p50 ms', 'itl50', '{:.0f}'), ('ITL p95 ms', 'itl95', '{:.0f}'),
        ('worst-gap p95 s', 'maxg95', '{:.1f}'), ('stall>1s /req', 'stall2', '{:.2f}'), ('E2E p50', 'e2e50', '{:.1f}'), ('E2E p95', 'e2e95', '{:.1f}'),
        ('out tok/s', 'out_tps', '{:.1f}'), ('in tok/s', 'in_tps', '{:.0f}'), ('good req/min', 'good', '{:.1f}'),
        ('fin %', 'fin', '{:.0%}'), ('util S', 'utilS', '{:.0%}'), ('util M', 'utilM', '{:.0%}')]
BEST = {'ttft50': min, 'ttft95': min, 'itl50': min, 'itl95': min, 'maxg95': min, 'stall2': min, 'e2e50': min, 'e2e95': min,
        'out_tps': max, 'in_tps': max, 'good': max}


def table(wl, rate, setups=None, prm=None, slo=None, out=print):
    slo = slo or WORKLOADS[wl][3]
    setups = setups or list(SETUPS)
    rows = {n: evaluate(wl, rate, SETUPS[n], prm, slo=slo) for n in setups}
    # a winner only counts if the setup finished >=95% of its requests (otherwise the percentiles are censored)
    stable = [n for n in setups if rows[n]['fin'] >= 0.95] or setups
    win = {}
    for key, fn in BEST.items():
        vals = {n: rows[n][key] for n in stable if rows[n][key] == rows[n][key]}
        if vals:
            bv = fn(vals.values())
            win[key] = [n for n, v in vals.items() if abs(v - bv) <= 1e-9 + 0.02 * abs(bv)]
    out(f'\n**{wl}, {WORKLOADS[wl][2]}, load {rate} /s** (SLO for goodput: TTFT<={slo[0]:g}s and per-request ITL p95<={slo[1]*1000:.0f}ms)\n')
    out('| setup | ' + ' | '.join(c[0] for c in COLS) + ' |')
    out('|---|' + '---|' * len(COLS))
    for n in setups:
        r = rows[n]
        cells = []
        for _, key, f in COLS:
            c = fmt(r[key], f)
            if key in win and n in win[key]:
                c = f'**{c}**'
            cells.append(c)
        out(f'| {n} | ' + ' | '.join(cells) + ' |')
    return rows, win


def batch_table(prm=None, out=print):
    setups = list(SETUPS)
    out('\n**batch summarisation: 24 docs, 8-16k in / 400 out, all submitted at t=0** (makespan = time until the last token)\n')
    out('| setup | makespan s | out tok/s | in tok/s | req/min | TTFT p50 (queue incl.) | ITL p95 ms | util S | util M |')
    out('|---|---|---|---|---|---|---|---|---|')
    best = None
    res = {}
    for n in setups:
        sim = run_one('batch', 0.0, SETUPS[n], prm or PARAM, 1)
        r = summarize([sim], 'batch')
        r['req_min'] = 24 / r['makespan'] * 60
        res[n] = r
    bm = min(r['makespan'] for r in res.values())
    for n in setups:
        r = res[n]
        mk = f"**{r['makespan']:.0f}**" if abs(r['makespan'] - bm) < 0.02 * bm else f"{r['makespan']:.0f}"
        out(f"| {n} | {mk} | {r['out_tps']:.1f} | {r['in_tps']:.0f} | {r['req_min']:.1f} | {r['ttft50']:.0f} | {r['itl95']:.0f} | {r['utilS']:.0%} | {r['utilM']:.0%} |")
    return res


# ---------------------------------------------------------------------------------------------------------- validation
def single(n, n_out, setup_name, prm=None):
    prm = prm or PARAM
    rng = random.Random(0)
    r = Req(0, 0.0, n, n_out, 'chat')
    sim = Sim(prm, SETUPS[setup_name], [r], [], 1, 0)
    sim.run(3600)
    return r


def validate(out=print):
    p = PARAM
    out('\n**Validation 1: idle single-request TTFT (s)**.  A and B are the model INPUTS (tables from baselines.md), so they agree by construction; the C-now row is an independent check (export profile + H_now + M landing vs the e2e-final.md landing re-run medians)\n')
    out('| system | n | model | measured |')
    out('|---|---|---|---|')
    A = [(1035, 0.84), (4055, 2.22), (8275, 4.47), (16652, 8.49), (32912, 16.36), (67915, 33.6)]
    B = [(1008, 1.62), (4184, 4.58), (8280, 8.74), (16433, 17.05), (33157, 34.1)]      # (65k/128k: 67.2 / 144.5 s measured; the sim keeps B at <= 40,960 rows)
    C = [(4326, 4.93), (8141, 8.88), (16592, 16.10), (30488, 29.12)]
    for lab, rows, nm in (('A (S alone)', A, 'A  S alone'), ('B (M alone)', B, 'B  M alone'), ('C-now (export 1.1k, H_now)', C, 'C  now (export 1.1k, H=1.0)')):
        for n, meas in rows:
            out(f'| {lab} | {n} | {single(n, 1, nm).t_first:.2f} | {meas:.2f} |')
    out('\n**Validation 2: decode (S and M inputs; the 1/2/4-stream S and M rows are measured, baselines.md s4)**\n')
    out('| quantity | model | measured |')
    out('|---|---|---|')
    for k, agg, rnd in ((1, 58.2, 30.8), (2, 82.0, 41.3), (4, 110.0, 56.5)):
        out(f"| S decode, {k} streams: aggregate tok/s / event gap ms | {k * tpe_S(p, k) / step_S(p, k):.1f} / {step_S(p, k) * 1000:.1f} | {agg} / {rnd} |")
    for k, agg, rnd in ((1, 39.4, 25.4), (2, 51.2, 39.0), (4, 60.6, 66.0)):
        out(f"| M decode, {k} streams: aggregate tok/s / round ms | {k / step_M(p, k):.1f} / {step_M(p, k) * 1000:.1f} | {agg} / {rnd} |")
    out(f"| M decode, 8 streams (linear model; slots stay <= 4 in every run; measured saturates) | {8 / step_M(p, 8):.1f} / {step_M(p, 8) * 1000:.0f} | 59.3 / 135 |")
    out(f"| M 4096-row prefill chunk (s) | {Sim(p, SETUPS['B  M alone'], [], [], 1)._pf_M(4096, False):.2f} | ~4.6 (baselines B 4184 rows 4.58 s) |")
    sim = Sim(p, SETUPS['C  now (export 1.1k, H=1.0)'], [], [], 1)
    n = 8192
    tail = tail_rows(p, sim.H, n)
    ov = sim.H['link_extra'] + sim.link_s(n) + sim.land_s(n, tail)
    out(f"| hand-off overhead after the export, 8k, tail {tail} (s) | {ov:.2f} | 1.0 (0.3-1.4; e2e-final.md 'other' + 'M land+tail') |")
    for nm in ('H_ship1', 'H_best'):
        sim.H = p[nm]
        tail = tail_rows(p, sim.H, n)
        ov = sim.H['link_extra'] + sim.link_s(n) + sim.land_s(n, tail)
        out(f'| hand-off overhead, scenario {nm} (s) | {ov:.2f} | target 0.3 / 0.2 (not measured) |')
    out('')


def loaded_validate(out=print):
    """emergent behaviour of the scheduler vs baselines.md s4/s5 (these are NOT inputs)"""
    out('\n**Validation 3: head-of-line (a long stream is decoding, a prompt arrives at t=3 s)** (baselines.md s5)\n')
    out('| server | newcomer n | running stream: max event gap s / p95 gap during the prefill s | newcomer TTFT (alone) s | measured |')
    out('|---|---|---|---|---|')
    for nm, who, n2, meas in (('A  S alone', 'A (S)', 8700, 'max stall 1.2 s, p95 1.05 s, ~10 tok/s during; newcomer TTFT 5.65 / 5.09 (4.0-4.5 alone)'),
                              ('A  S alone', 'A (S)', 16900, 'max stall 1.2 s, p95 1.06 s; TTFT 10.85 / 10.31 (8.5 alone)'),
                              ('B  M alone', 'B (M)', 9200, 'running stream stops for the whole prefill: 9.76 s; TTFT 9.76'),
                              ('B  M alone', 'B (M)', 17100, 'stops 17.66 s; TTFT 17.66')):
        r1 = Req(0, 0.0, 300, 3000, 'chat')
        r2 = Req(1, 3.0, n2, 1, 'rag')
        sim = Sim(PARAM, SETUPS[nm], [r1, r2], [], 100, 0)
        sim.run(3600)
        t, during = r1.t_first, []
        for g in r1.gaps:
            t += g
            if 3.0 < t <= r2.t_first + 0.05:
                during.append(g)
        alone = single(n2, 1, nm).t_first
        out(f'| {who} | {n2} | {max(r1.gaps):.2f} / {pct(during, 95):.2f} | {r2.t_first - r2.t_arr:.2f} ({alone:.2f}) | {meas} |')
    out('\n**Validation 4: prefill is FIFO-serialised (4 prompts of 4.3k at once; TTFT of each)** (baselines.md s4)\n')
    out('| server | model TTFTs s | measured TTFTs s |')
    out('|---|---|---|')
    for nm, who, meas in (('A  S alone', 'A (S)', '2.2, 4.4, 7.0, 9.6'), ('B  M alone', 'B (M, --parallel 4)', '5.6, 10.4, 15.4, 20.3')):
        rs = [Req(i, 0.0 + 1e-3 * i, 4300, 1, 'rag') for i in range(4)]
        sim = Sim(PARAM, SETUPS[nm], rs, [], 100, 0)
        sim.run(3600)
        out(f'| {who} | ' + ', '.join(f'{r.t_first - r.t_arr:.1f}' for r in rs) + f' | {meas} |')


# ---------------------------------------------------------------------------------------------------------- sensitivity
def single_ttft_c_vs_a(n, rate_exp, H, prm=PARAM):
    """analytic idle TTFT: C = n_remote/rate + H ; A = production"""
    tail = tail_rows(prm, H, n)
    land = H['land_fixed'] + H['land_per_tok'] * n + (prm['one_step_s'] if tail <= 1 else prm['M_tail_a'] + prm['M_tail_b'] * tail)
    ov = H['link_extra'] + n * prm['kb_per_tok'] * 1e3 * 8 / (prm['link_gbps'] * 1e9) + land
    return (n - tail) / rate_exp + ov, ov


def c3_ttfts(n, lag, prm=None):
    """idle TTFT of C3a (row split, lag fraction) and C3b (layer split) for an n-token prompt, decode on M"""
    pp = prm_with(c3_lag_frac=lag) if prm is None else dict(prm, c3_lag_frac=lag)
    r = Req(0, 0.0, n, 1, 'chat')
    s2 = Sim(pp, SETUPS['C3a C5 + row split (layer-major)'], [r], [], 1)
    _, _, ta = predict_split(s2, r, 'M', 0.0)
    _, tb = predict_lsplit(s2, r, 0.0)
    return ta, tb


def c3_idle_table(out=print):
    p = PARAM
    out('\n**C3 idle single-request TTFT (s)**: S = replay-export prod table (x1.05), M = whole-prompt table; C3a: x chosen to equalise finish times, M starts lag x (S part) after S; C3b: layer-split chunk pipeline (S 70% of the layers, M 30% + 0.7 s exact cone, study-chunk-pipeline.md)\n')
    out('| n | A (S prod) | B (M) | C replay+ship1 | C3a lag 1.0 (segment-major: no overlap) | C3a lag 0.5 | C3a lag 0.2 | C3a lag 0.05 (layer-skewed) | C3a lag 0 | C3b layer-split | study-chunk-pipeline B / A(ideal) |')
    out('|---|---|---|---|---|---|---|---|---|---|---|')
    study = {4096: '3.44 / 2.21', 8192: '4.54 / 3.41', 16384: '6.91 / 5.83', 32768: '10.95 / 10.21', 65536: '20.02 / 19.66'}
    for n in (2048, 4096, 8192, 16384, 32768, 40960, 65536):
        sim = Sim(p, SETUPS['C3a C5 + row split (layer-major)'], [], [], 1)
        ta = interp(p['S_prod_pf'], n)
        tb = interp(p['M_pf_pts'], n)
        sim.H = p['H_ship1']
        tc = interp(p['S_prod_pf'], n - 1) * p['S_replay_export_mult'] + sim.H['link_extra'] + sim.link_s(n) + sim.land_s(n, 1)
        cells = []
        for lag in (1.0, 0.5, 0.2, 0.05, 0.0):
            cells.append(c3_ttfts(n, lag)[0])
        tlb = c3_ttfts(n, 0.05)[1]
        out(f'| {n} | {ta:.2f} | {tb:.2f} | {tc:.2f} | ' + ' | '.join(f'{c:.2f}' for c in cells) + f' | {tlb:.2f} | {study.get(n, "-")} |')
    out('\nBreak-even prompt length n* above which C3 beats S alone (idle TTFT; "never" if not within 4k..65k):\n')
    out('| variant | n* |')
    out('|---|---|')

    def nstar(fn):
        lo, hi = 1024, 65536
        if not fn(hi) < interp(p['S_prod_pf'], hi):
            return 'never'
        if fn(lo) < interp(p['S_prod_pf'], lo):
            return '<1k'
        for _ in range(20):
            mid = (lo + hi) / 2
            if fn(mid) < interp(p['S_prod_pf'], mid):
                hi = mid
            else:
                lo = mid
        return f'{hi:.0f}'
    for lag in (1.0, 0.5, 0.2, 0.05, 0.0):
        out(f'| C3a lag {lag:g} | {nstar(lambda n, l=lag: c3_ttfts(n, l)[0])} |')
    out(f'| C3b layer-split | {nstar(lambda n: c3_ttfts(n, 0.05)[1])} |')
    PS, PM = 32768 / (interp(p['S_prod_pf'], 32768) * p['S_replay_export_mult']), 1.0 / p['M_pf_b']
    out(f'\nAsymptotic combined prefill rate of C3a = P_S + (1-lag) x P_M (P_S = replay-export rate {PS:.0f} tok/s at 32k, P_M = 1/{p["M_pf_b"]*1000:.2f} ms = {PM:.0f} tok/s): '
        + ', '.join(f'lag {l:g}: {PS + (1 - l) * PM:.0f}' for l in (1.0, 0.5, 0.2, 0.05, 0.0)) + f' tok/s; the sum of two independent servers is {PS + PM:.0f}.')


def sens_breakeven(out=print):
    out('\n**Idle single-request break-even (analytic, exact in this model)**\n')
    out('Export rate (tok/s, producer side, for the whole prompt) that C needs to match TTFT of A (S prod, same GPUs) / B (M alone):\n')
    out('| n | A TTFT | rate needed H=1.0 | H=0.3 | H=0.2 | B TTFT | rate needed H=1.0 | H=0.3 | H=0.2 |')
    out('|---|---|---|---|---|---|---|---|---|')
    p = PARAM
    for n in (4096, 8192, 16384, 32768):
        ta = interp(p['S_prod_pf'], n)
        tb = interp(p['M_pf_pts'], n)
        cells = []
        for T in (ta, tb):
            cells.append(f'{T:.2f}')
            for H in (1.0, 0.3, 0.2):
                cells.append(f'{n / (T - H):.0f}' if T > H else 'never')
        out(f'| {n} | ' + ' | '.join(cells) + ' |')


def sens_run(title, wl, rate, sets, variants, out=print, slo=None):
    """variants: list of (label, prm-overrides) -> table of selected metrics per setup"""
    keys = [('ttft95', 'TTFT p95', '{:.1f}'), ('itl95', 'ITL p95 ms', '{:.0f}'), ('e2e50', 'E2E p50', '{:.1f}'), ('out_tps', 'out tok/s', '{:.1f}'), ('good', 'good/min', '{:.1f}')]
    out(f'\n**{title}** ({wl}, load {rate}/s; cells = TTFT p95 s / ITL p95 ms / E2E p50 s / out tok/s / goodput req/min)\n')
    out('| variant | ' + ' | '.join(sets) + ' |')
    out('|---|' + '---|' * len(sets))
    for label, ov in variants:
        prm = prm_with(**ov)
        cells = []
        for sname in sets:
            r = evaluate(wl, rate, SETUPS[sname], prm, slo=slo)
            cells.append(' / '.join(f.format(r[k]) for k, _, f in keys) + ('' if r['fin'] >= 0.95 else ' (sat)'))
        out(f'| {label} | ' + ' | '.join(cells) + ' |')


def breakeven_search(wl, rate, setup_name, vs_names, key, param_fn, lo, hi, better, prm=None, n_iter=7, slo=None):
    """smallest/largest parameter value in [lo,hi] at which setup_name beats ALL vs_names on key (monotone assumption); returns None if never"""
    def beats(v):
        pr = param_fn(v)
        a = evaluate(wl, rate, SETUPS[setup_name], pr, slo=slo)
        if a['fin'] < 0.95:
            return False
        for o in vs_names:
            b = evaluate(wl, rate, SETUPS[o], pr, slo=slo)
            if b['fin'] >= 0.95 and not better(a[key], b[key]):
                return False
        return True
    if not beats(hi):
        return None
    if beats(lo):
        return lo
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        if beats(mid):
            hi = mid
        else:
            lo = mid
    return hi



KEYS_BE = [('ttft95', min, 'TTFT p95'), ('itl95', min, 'ITL p95'), ('e2e50', min, 'E2E p50'), ('good', max, 'goodput')]
CMP = ['A  S alone', 'A+B least-loaded']


def sens_m_decode(out=print):
    """how fast must M decode be (single stream / batching / working drafts) for C (and C5) to beat A and A+B-LL"""
    sets = ['A  S alone', 'A+B least-loaded', 'A+B overflow (M only if S slots full)', 'C  replay+ship1 (H=0.3)', 'C5 S+M, S also decodes']
    k1 = PARAM['M_dec_k1_toks']
    for wl, rate in (('chat', 0.12), ('mixed', 0.10), ('rag', 0.06)):
        vs = [(f'M {k1*f:.0f} tok/s k=1 ({4/step_M(prm_with(M_dec_speedup=f),4):.0f} agg @k=4)', dict(M_dec_speedup=f)) for f in (1.0, 1.5, 2.0, 3.0)]
        vs += [('M drafts work: 1.5 tok/round at k=1 (-> 59 tok/s), 1.25 at k=2, 1.0 from k=4', dict(M_tpe_pts=[(1, 1.5), (2, 1.25), (4, 1.0), (32, 1.0)])),
               ('M drafts work like S (1.79/1.69/1.55 tok/round; round time unchanged = optimistic)', dict(M_tpe_pts=PARAM['S_tpe_pts']))]
        sens_run(f'M decode single-stream rate (uniform speed-up of M overhead terms, or working drafts; roofline cap {1/roof_s(PARAM,1,PARAM["M_BW_GBs"]):.0f} tok/s at 1 token/round)', wl, rate, sets, vs, out=out)
    for wl, rate in (('chat', 0.12), ('mixed', 0.10)):
        vs = []
        for row in (13.5, 8.0, 4.0, 2.0):
            pr = prm_with(M_row_ms=row)
            vs.append((f'M batching: +{row:g} ms/extra stream (agg k=4 {4/step_M(pr,4):.0f}, k=8 {8/step_M(pr,8):.0f}, k=16 {16/step_M(pr,16):.0f} tok/s; M slots 8)', dict(M_row_ms=row, M_slots=8)))
        sens_run('M batching quality at fixed single-stream 39 tok/s, 8 M slots', wl, rate, sets, vs, out=out)


def sens_S(out=print):
    sets = ['A  S alone', 'A+B least-loaded', 'A+B overflow (M only if S slots full)', 'C  replay+ship1 (H=0.3)', 'C5 S+M, S also decodes']
    for wl, rate in (('chat', 0.12), ('mixed', 0.10)):
        sens_run('S profile / interleave: live service (baselines.md) vs the older main21ea profile; S decode window (TF_DS_PREFILL_INTERLEAVE_MS) and chunk size', wl, rate, sets,
                 [('live service: 2048-row chunks, 250 ms decode window (default)', {}),
                  ('older main21ea profile (22 ms/step, 4096-row pieces, 1 decode step per piece)', dict(PROFILE_MAIN21EA)),
                  ('S decode window 50 ms', dict(S_dec_window_s=0.05)),
                  ('S decode window 1000 ms', dict(S_dec_window_s=1.0)),
                  ('S chunk 1024 rows (assumed free of throughput loss)', dict(S_chunk_rows=1024)),
                  ('S 8 slots (round times beyond k=4 extrapolated, ASSUMED)', dict(S_slots=8))], out=out)
    for wl, rate in (('rag', 0.06), ('mixed', 0.10)):
        sens_run('S export (replay-export) speed: multiplier on the live production prefill time (1.05 assumed)', wl, rate, sets,
                 [(f'export x{f:g} => ~{8192/(interp(PARAM["S_prod_pf"],8192)*f):.0f} tok/s at 8k', dict(S_replay_export_mult=f)) for f in (1.0, 1.05, 1.5, 2.0)], out=out)


def sens_H(out=print):
    sets = ['A  S alone', 'A+B least-loaded', 'C  replay+ship1 (H=0.3)', 'C2 conditional disagg', 'C5 S+M, S also decodes']
    for wl, rate in (('agent', 0.05), ('rag', 0.06)):
        sens_run('hand-off overhead scenario (H_now ~1.2 s incl. tail forward, H_ship1 ~0.33 s, H_best ~0.17 s) and link rate', wl, rate, ['C  replay+ship1 (H=0.3)', 'C5 S+M, S also decodes'],
                 [('H_now', dict(H_ship1=PARAM['H_now'], H_best=PARAM['H_now'])), ('H_ship1', {}),
                  ('H_best', dict(H_ship1=PARAM['H_best'])),
                  ('H_ship1, link 3 Gbit/s', dict(link_gbps=3.0)), ('H_ship1, link 50 Gbit/s', dict(link_gbps=50.0))], out=out)


def sens_c3(out=print):
    sets = ['A  S alone', 'A+B least-loaded', 'C5 S+M, S also decodes', 'C3a C5 + row split (layer-major)', 'C3b C5 + layer-split pipeline']
    for wl, rate in (('longdoc', 0.02), ('rag', 0.03), ('mixed', 0.06)):
        sens_run('C3a feasibility (lag = fraction of the S part M must wait before starting; 1.0 = no overlap possible; C3b is lag-independent)', wl, rate, sets,
                 [(f'lag {l:g}', dict(c3_lag_frac=l)) for l in (1.0, 0.5, 0.2, 0.05)], out=out)



def sens_chunks(out=print):
    sets = ['A  S alone', 'B  M alone', 'A+B least-loaded', 'A+B overflow (M only if S slots full)', 'C  replay+ship1 (H=0.3)', 'C5 S+M, S also decodes']
    for wl, rate in (('chat', 0.12), ('mixed', 0.10)):
        sens_run('M prefill interleave: today M has NONE (a running stream stops for the whole prompt); a fix = 1 decode round per prefill chunk (what study-metal-concurrency.md designed)', wl, rate, sets,
                 [('today: no interleave, 4096-row chunks', {}),
                  ('fix: 1 decode round per 4096-row chunk', dict(M_pf_per_dec=1)),
                  ('fix: 1 decode round per 1024-row chunk (M prefill rate x0.86)', dict(M_pf_per_dec=1, M_chunk_rows=1024)),
                  ('fix: 1 decode round per 512-row chunk (M prefill rate x0.55)', dict(M_pf_per_dec=1, M_chunk_rows=512))], out=out)


def slo_table(out=print):
    sets = ['A  S alone', 'B  M alone', 'A+B least-loaded', 'A+B overflow (M only if S slots full)', 'A+B cost-aware (no remote)', 'C  replay+ship1 (H=0.3)', 'C2 conditional disagg', 'C5 S+M, S also decodes', 'C4 class (chat->S dec)', 'C3b C5 + layer-split pipeline']
    out('\n**Goodput (requests/min meeting the SLO: TTFT <= X and the request\'s own ITL p95 <= Y)**, mixed workload\n')
    for rate in (0.06, 0.10, 0.14):
        out(f'\nmixed @ {rate}/s\n')
        slos = [(2.0, 0.060), (5.0, 0.100), (10.0, 0.100), (10.0, 0.060)]
        out('| setup | ' + ' | '.join(f'TTFT<={a:g}s, ITL<={int(b*1000)}ms' for a, b in slos) + ' |')
        out('|---|' + '---|' * len(slos))
        sims = {n: [run_one('mixed', rate, SETUPS[n], PARAM, sd) for sd in SEEDS] for n in sets}
        for n in sets:
            out(f'| {n} | ' + ' | '.join(f"{summarize(sims[n], 'mixed', slo=sl)['good']:.1f}" for sl in slos) + ' |')



SHORT = {'A  S alone': 'A', 'B  M alone': 'B', 'A+B round-robin': 'A+B rr', 'A+B least-loaded': 'A+B ll', 'A+B length (>=4k->S)': 'A+B len',
         'A+B overflow (M only if S slots full)': 'A+B ovf', 'A+B cost-aware (no remote)': 'A+B cost',
         'C  now (export 1.1k, H=1.0)': 'C-now', 'C  replay+ship1 (H=0.3)': 'C', 'C2 conditional disagg': 'C2', 'C5 S+M, S also decodes': 'C5',
         'C4 class (chat->S dec)': 'C4', 'C3a C5 + row split (layer-major)': 'C3a', 'C3b C5 + layer-split pipeline': 'C3b'}


def winners(out=print):
    """winner(s) per metric for every workload x load (2% ties; only setups that finished >=95% of their requests)"""
    keys = [('ttft50', 'TTFT p50'), ('ttft95', 'TTFT p95'), ('itl95', 'ITL p95'), ('maxg95', 'worst-gap p95'), ('e2e50', 'E2E p50'), ('e2e95', 'E2E p95'),
            ('out_tps', 'out tok/s'), ('good', 'goodput')]
    out('\n| workload @ load | ' + ' | '.join(k[1] for k in keys) + ' |')
    out('|---|' + '---|' * len(keys))
    for wl in ('chat', 'rag', 'agent', 'longdoc', 'mixed'):
        for rate in WORKLOADS[wl][1]:
            _, win = table(wl, rate, out=lambda *a, **k: None)
            cells = []
            for key, _n in keys:
                w = [SHORT[n] for n in win.get(key, [])]
                cells.append(', '.join(w) if len(w) <= 4 else f'{len(w)}-way tie')
            out(f'| {wl} @ {rate} | ' + ' | '.join(cells) + ' |')


def breakevens(out=print):
    out(f'\n**Break-even search**: smallest M decode speed-up (x current: {PARAM["M_dec_k1_toks"]:.0f} tok/s k=1, {4/step_M(PARAM,4):.0f} agg @k=4) at which setup X beats BOTH A and A+B-least-loaded on the metric (bisection, 7 steps, range 1..4x; "never" = still not better at 4x)\n')
    out('| workload @ load | setup | TTFT p95 | ITL p95 | E2E p50 | goodput |')
    out('|---|---|---|---|---|---|')
    for wl, rate in (('chat', 0.12), ('rag', 0.06), ('mixed', 0.10), ('agent', 0.05)):
        for sname in ('C  replay+ship1 (H=0.3)', 'C5 S+M, S also decodes'):
            cells = []
            for key, better, _ in KEYS_BE:
                v = breakeven_search(wl, rate, sname, CMP, key, lambda f: prm_with(M_dec_speedup=f), 1.0, 4.0,
                                     (lambda a, b: a < b - 1e-9) if better is min else (lambda a, b: a > b + 1e-9))
                cells.append('never' if v is None else ('already' if v <= 1.0001 else f'{v:.2f}x ({PARAM["M_dec_k1_toks"]*v:.0f} tok/s)'))
            out(f'| {wl} @ {rate} | {sname} | ' + ' | '.join(cells) + ' |')


def energy_table(out=print):
    p = PARAM
    out('\n**Energy per 1000 output tokens** from simulated busy-seconds x MEASURED GPU-side power (baselines.md s7: S pair idle 25 W / prefill 156 W / decode ~135 W; M idle 36.5 / prefill 138 / decode 104 W; nvidia-smi + powermetrics GPU only, wall power higher and unmeasured)\n')
    out('| setup | S prefill s | S decode s | M prefill s | M decode s | J / 1k out tok, dynamic above idle | J / 1k out tok, both machines powered all run | out tok/s |')
    out('|---|---|---|---|---|---|---|---|')
    for wl, rate in (('mixed', 0.10), ('chat', 0.12), ('rag', 0.06)):
        out(f'| *{wl} @ {rate}* | | | | | | | |')
        for n in ('A  S alone', 'B  M alone', 'A+B least-loaded', 'A+B overflow (M only if S slots full)', 'A+B cost-aware (no remote)', 'C  replay+ship1 (H=0.3)', 'C5 S+M, S also decodes', 'C4 class (chat->S dec)'):
            sims = [run_one(wl, rate, SETUPS[n], PARAM, sd) for sd in SEEDS]
            tok = sum(m.tokens_out for sm in sims for m in (sm.S, sm.M))
            spf, sdec = sum(sm.S.pf_time for sm in sims), sum(sm.S.dec_time for sm in sims)
            mpf, mdec = sum(sm.M.pf_time for sm in sims), sum(sm.M.dec_time for sm in sims)
            dyn = (p['W_S_pf'] - p['W_S_idle']) * spf + (p['W_S_dec'] - p['W_S_idle']) * sdec + (p['W_M_pf'] - p['W_M_idle']) * mpf + (p['W_M_dec'] - p['W_M_idle']) * mdec
            idle = (p['W_S_idle'] + p['W_M_idle']) * HORIZON * len(sims)
            r = summarize(sims, wl)
            sat = '' if r['fin'] >= 0.95 else ' (sat)'
            out(f'| {n} | {spf/tok*1000:.1f} | {sdec/tok*1000:.1f} | {mpf/tok*1000:.1f} | {mdec/tok*1000:.1f} | {dyn/tok*1000:.0f} | {(dyn+idle)/tok*1000:.0f} | {r["out_tps"]:.1f}{sat} |')


def limits_table(out=print):
    p = PARAM
    out('\n| setup | max context (tokens) | decode slots | note |')
    out('|---|---|---|---|')
    out(f"| A S alone | {p['S_ctx']} config (served 134k in baselines.md; 4 streams share the pool) | {p['S_slots']} | [MEAS baselines.md s6] |")
    out(f"| B M alone | 1,048,576 admitted; 262,144 booted (parallel 1), 147k served; 4 slots tested at {p['M_ctx']} | {p['M_slots']} (8 add nothing) | [MEAS baselines.md s6]; KV 1402 B/token, memory not limiting |")
    out(f"| A+B | {p['S_ctx']} (long requests go to S) | {p['S_slots']}+{p['M_slots']} | |")
    out(f"| C S->M | min(export pool {p['S_export_pool']}, M ctx tested {p['M_ctx']}) = {min(p['S_export_pool'], p['M_ctx'])} | {p['M_slots']} decode (1 export at a time) | the M ctx is a test artefact (memory admits 1M); export pool 65,536 [MEAS prefill-tuning.md] |")
    out(f"| C5 / C3 (adaptive) | {p['S_ctx']} (requests that do not fit M stay on S) | {p['S_slots']}+{p['M_slots']} | split prefill: <= min({p['S_export_pool']} export pool, {p['M_ctx']}) |")


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if cmd in ('validate', 'all'):
        validate()
        loaded_validate()
    if cmd in ('tables', 'all'):
        for wl in (sys.argv[2:] or ['chat', 'rag', 'agent', 'longdoc', 'mixed']):
            for rate in WORKLOADS[wl][1]:
                table(wl, rate)
        if not sys.argv[2:]:
            batch_table()
    if cmd in ('winners', 'all'):
        winners()
    if cmd in ('c3', 'all'):
        c3_idle_table()
    if cmd in ('sens', 'all'):
        sens_breakeven()
        sens_m_decode()
        sens_S()
        sens_H()
        sens_c3()
        sens_chunks()
        breakevens()
        energy_table()
        slo_table()
    if cmd in ('limits', 'all'):
        limits_table()
