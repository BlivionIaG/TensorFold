"""Router configuration: ONE JSON file, defaults here (and dumped to router.config.json).

Every default is tagged with where it comes from:
    [MEAS file]   measured (baselines.md = BaselineBench 2026-11-11 on the live service and on the M5 server)
    [SIM]         the value bench/sim.py (setups-simulation.md) uses for the same quantity
    [ASSUMED]     a guess; tune it
User files are deep-merged over DEFAULTS.  Unknown keys are an error (typos must not be silent); keys starting with '_' are comments.
Backends are a dict by name; the names "S" and "M" start from the measured presets below, any other name needs "preset": "S"|"M"
(or a full profile).
"""
import copy
import json

# --------------------------------------------------------------------------------------------------------------------
# measured backend presets (what the cost model predicts with; the runtime EWMAs only rescale them)
# --------------------------------------------------------------------------------------------------------------------
PRESET_S = {
    "_src": "A = two DGX Sparks, production API (head :8888). baselines.md s1-s6, setups-simulation.md PARAM",
    "role": "primary",
    "slots": 4,                       # [MEAS baselines.md s1: 4 streams, a 5th queues]
    "max_context": 1048576,           # [MEAS baselines.md s6: config = model window; 134k served]
    "model": "DeepSeek-V4.1-Flash-1017",   # [MEAS baselines.md s1: --name]; what the backend wants in the "model" field
    # whole-prompt prefill seconds (idle machine, uncached rows) [MEAS baselines.md s2; 39-token floor 0.16 s]
    "prefill_pts": [[0, 0.15], [39, 0.16], [1035, 0.84], [4055, 2.22], [8275, 4.47], [16652, 8.49], [32912, 16.36], [67915, 33.6], [134093, 69.8]],
    # [MEAS baselines.md s4] round (event gap) ms at 1/2/4 streams; 8+ [ASSUMED] sim.py S_round_pts_ms; tokens per stream per round = draft acceptance
    "decode": {"round_ms_pts": [[1, 30.8], [2, 41.3], [4, 56.5], [8, 79.0], [16, 122.6], [32, 193.1]],
               "tokens_per_round_pts": [[1, 1.79], [2, 1.69], [4, 1.55], [32, 1.55]]},
    # S prefills in 2048-row chunks and runs 250 ms of decode rounds between two chunks [MEAS baselines.md s1, TF_DS_PREFILL_INTERLEAVE_MS=250]
    "prefill": {"mode": "interleave", "chunk_rows": 2048, "window_s": 0.25,
                "stall_factor": 1.0},   # [SIM] share of a prefill that counts as a stall for the running streams (real S keeps ~10 tok/s: 0.8 is arguable)
    "guard_rows": None,               # no long-prompt guard on the primary
    "prefix_cache_tokens": 800000,    # pool of the prefix-affinity model [INFERENCE, SessionTier: 1M-position pool, RAM tier, spill off: present after 0 churn, gone after 1.25M tokens of cold prompts]; the cached_tokens feedback corrects it
    "resume_grid": 2048,              # [MEAS SessionTier on live :8888] a hit resumes on a 2048-token grid (TF_DS_REPLAY_FLOOR): cached = floor(h/2048)*2048
    "tokenizer": True,                # serves /v1/tokenize (TensorFold native server: docs/api.md)
}

PRESET_M = {
    "_src": "B = M5 Ultra, TensorFold native server (--parallel 4). baselines.md, setups-simulation.md PARAM",
    "role": "secondary",
    "slots": 4,                       # [MEAS baselines.md: --parallel 4; 8 slots add nothing]
    "max_context": 40960,             # [MEAS baselines.md s6, e2e-final.md: 4 slots tested at 40,960; --context 262144 --parallel 1 also booted]
    "model": "DeepSeek-V4.1-Flash",   # [MEAS baselines.md s1 B command: --name]
    "prefill_pts": [[0, 0.70], [39, 0.75], [1008, 1.62], [4184, 4.58], [8280, 8.74], [16433, 17.05], [33157, 34.1], [65022, 67.2], [131870, 144.5]],
    # [MEAS baselines.md s4] 25.4 / 39 / 66 ms at k = 1 / 2 / 4 (60.6 tok/s aggregate); linear beyond (13.5 ms per extra stream, sim.py step_M); 1 token per round
    "decode": {"round_ms_pts": [[1, 25.380711], [4, 66.006601]], "tokens_per_round_pts": [[1, 1.0], [32, 1.0]]},
    # M's running streams STOP for the whole prompt prefill [MEAS baselines.md s5]; prompts strictly one at a time [MEAS s4]
    "prefill": {"mode": "block", "chunk_rows": 4096, "window_s": 0.0, "stall_factor": 1.0},
    "guard_rows": 4096,               # [SIM setups-simulation.md s6 stage 0 rule 3]: <= ~4k uncached rows (one <= 4.5 s stall) while its streams decode
    "prefix_cache_tokens": 2000000,   # [MEAS baselines.md s1: --prompt-cache-gib 4; KV 1402 B/token]
    "resume_grid": 1,                 # [UNMEASURED] M's resume granularity
    "tokenizer": True,
}

PRESETS = {"S": PRESET_S, "M": PRESET_M}

BACKEND_COMMON = {
    "enabled": True,
    "url": None,                      # http://host:port ; required for an enabled backend
    "api_key_file": None,             # TensorFold format: one key per line, "label: key", '#' comments; first key is used; re-read when the mtime changes
    "api_key_env": None,              # alternative: name of an environment variable holding the key
    "preset": None,                   # "S" or "M" (default: the backend's own name when it is S or M)
    "health_path": "/health",
    "load_path": "/v1/load",          # needs a key unless the server runs --metrics-open; 404/403 just disables external-load sensing
    "tokenize_path": "/v1/tokenize",
    "max_inflight": None,             # admission control: requests the router lets into this backend at once (None = its `slots`); more wait in the router, interactive first
    "queue_max": 32,                  # [ASSUMED] requests that may wait in the router for a lane of this backend; a request whose eligible backends are all this deep gets 503 + Retry-After
}

DEFAULTS = {
    "listen": {"host": "127.0.0.1", "port": 8000},
    "public_model": "DeepSeek-V4.1-Flash",   # what /v1/models lists and clients should send; ':background' etc. select a class
    "accept_any_model": True,                # unknown "model" values are routed anyway (rewritten to the backend's name)
    "class_suffixes": {":interactive": "interactive", ":background": "background", ":batch": "background", "-background": "background"},
    "class_header": "X-Request-Class",       # interactive | background
    "session_header": "X-Session-Id",        # optional explicit conversation id (prefix affinity without hashing the prompt)
    "route_header": "X-Route-To",            # force a backend by name (debug/ops); fails with 503 if it cannot take the request
    "explain_header": "X-Router-Explain",    # '1' -> the decision comes back in the X-Router-Decision response header
    "expected_output_header": "X-Expected-Output-Tokens",
    "slo_ttft_header": "X-SLO-TTFT",
    "default_class": "interactive",
    "client_api_key_file": None,             # if set every route but GET /health needs 'Authorization: Bearer KEY' (same file format)
    "max_body_mb": 64,
    "time_scale": 1.0,                       # TEST ONLY: router seconds per wall second (fake backends run faster than real time)
    "log": {"explain": False, "explain_file": None, "access": True, "ring": 200},
    "mode": {"dry_run": False,               # answer with the decision, contact no backend
             "shadow": None},                # backend name: proxy everything there, only LOG what the policy would have done
    "policy": {
        "mode": "cost",                      # "cost": own E2E + w_int * delay imposed on the streams already running (sim.py 'cost-aware'); "ll": min predicted TTFT (sim.py 'least-loaded')
        "w_int": 1.0,                        # [SIM PARAM w_int]
        "primary_bias_s": 0.3,               # [ASSUMED] the primary wins unless the other backend is cheaper by this many seconds ...
        "primary_bias_frac": 0.15,            # [SIM, E2] ... plus this share of the primary's own cost: M must predict >= 15 % less TTFT+E2E+interference, not 1-2 s on a 10-15 s request.  The E2 run sent 25 % of chat to M at 0.09 req/s (0.3 s only); 0.15: 14-27 % over 0.06-0.15, simulator goodput >= everywhere
        "estimate_ctx_margin": 0.05,         # [ASSUMED] a token ESTIMATE within this fraction below a context limit is treated as not fitting
        "classes": {
            "interactive": {"slo": True, "w_int": None, "prefer_role": None, "secondary_bonus_s": 0.0, "secondary_max_backlog_s": None},
            # background/batch: no SLO, protect running interactive streams (w_int 2), pull work to the secondary (M) until its queue is 240 s deep  [ASSUMED; setups-simulation.md s6 'background to M']
            "background": {"slo": False, "w_int": 2.0, "prefer_role": "secondary", "secondary_bonus_s": 20.0, "secondary_max_backlog_s": 240.0},
        },
        # SLO for the predicted TTFT: base + per_ktok * (prompt tokens / 1000).  Fits the simulator's per-workload SLOs
        # (chat 2k: 2 s, RAG 16k: 10 s, agent 5 s, long-doc 36k: 20 s) [SIM WORKLOADS]; itl_s = per-request ITL p95 SLO [SIM 100 ms]
        "slo_penalty": 3.0,                  # cost seconds per second of predicted TTFT over the SLO (inf-like = hard filter); chosen by tools/simcompare.py, see README
        "interf_overlap": True,              # the slowdown a new request imposes on a running stream lasts only as long as that stream still has tokens to generate (False: sim.py, every stream outlives the new request)
        "slo": {"ttft_base_s": 1.5, "ttft_per_ktok_s": 0.55, "itl_s": 0.100},
        "reject_ttft_s": None,               # admission control: answer 503 + Retry-After when even the best backend predicts a TTFT above this (off)
    },
    "tokens": {
        "mode": "auto",                      # "estimate" | "auto" (ask the backend's /v1/tokenize only near a decision boundary) | "exact" (always)
        "chars_per_token": 3.5,              # [ASSUMED] start value; calibrated online per model (baselines.md used 3.3 on docs+code text; prose ~4)
        "msg_overhead_tokens": 4,            # [ASSUMED] chat template tokens per message
        "base_overhead_tokens": 2,
        "calibration_alpha": 0.1,
        "exact_margin": 0.12,                # auto: ask when the estimate is within this fraction of a boundary (a context limit, a guard size)
        "exact_timeout_s": 1.5,
        "exact_cache": 512,
        "default_output_tokens": 300,        # [SIM: chat 256-768, RAG 200-300, agent 100-300] start value of the per-class output EWMA
        "output_warmup": True,               # output-length EWMA starts as a running mean (False: plain EWMA from default_output_tokens, 25+ requests to converge at alpha 0.1)
        "output_alpha": 0.1,
        "ctx_output_cap": 4096,              # for the context check a client's max_tokens is capped here (it is an upper bound, not a forecast) [ASSUMED]
    },
    "affinity": {
        "enabled": True,
        "ttl_s": 21600.0,                    # safety net only: S has no time expiry (SessionTier), entries go by pool pressure (backends.*.prefix_cache_tokens) or restart
        "max_entries": 200000,
        "stride_chars": 4096,                # anchor every 4096 characters up to linear_until, then every +12.5 %
        "linear_until_chars": 65536,
        "growth": 1.125,
        "min_prefix_tokens": 256,            # ignore hits smaller than this
        "record_reply": False,               # [MEAS SessionTier on S] the cached prefix is the previous request's PROMPT only (the reply is re-prefilled): do not anchor prompt+reply; True = anchor it (a backend that keeps generated tokens)
    },
    "health": {"interval_s": 5.0, "timeout_s": 2.0, "load_poll_s": 2.0, "load_timeout_s": 1.5, "external_load": True},
    "breaker": {"failure_threshold": 3, "cooldown_s": 10.0, "max_cooldown_s": 120.0, "half_open_max": 1},
    "failover": {"max_attempts": 2, "connect_timeout_s": 3.0, "first_byte_timeout_s": 30.0, "first_byte_per_ptok_s": 2.0,   # first byte of the RESPONSE: 30 s + 2 s per predicted TTFT second
                 "idle_timeout_s": 120.0,    # a stream with no byte for this long is cut
                 "retry_statuses": [429, 502, 503, 504]},
    "calibration": {"enabled": True, "alpha": 0.2, "min_scale": 0.4, "max_scale": 3.0, "min_rows": 256, "min_tokens": 16},
    # PD+ fleet (designs/pd-plus-fleet.md): the third path SM = S prefills with replay export, M pulls the KV and decodes.  Needs the M server started with
    # --prefill-from S (the router only tells it per request: body "prefill": "force" for SM, "never" for MM).  Off: the router is the stage-0 A+B router.
    "pd": {
        "enabled": False,
        "prefill_backend": "S",
        "decode_backend": "M",
        "mm_never": True,                    # MM requests carry prefill=never so the M server never pulls on its own
        "export_factor": 1.05,               # [ASSUMED] replay-export prefill seconds / native prefill seconds (ReplayExport measured it at 0.8-1.0 of the live rate)
        "handoff": {                         # H(n) = link_extra + wire + landing + tail  (0.26 s at 2k tokens)
            "link_extra_s": 0.15,            # [ASSUMED for tail=1] trigger + pull handshake over the LAN/RDMA; 0.10-0.55 s measured with the 512-row tail (e2e-final.md)
            "link_gbit": 29.0,               # [MEAS mcdma-bringup.md]
            "kv_bytes_per_token": 3700,      # [MEAS handoff-format.md] 3.7 KB per token on the wire
            "landing_base_s": 0.04,          # [MEAS landing-profile.md] 114.6 ms / 10.5k tokens
            "landing_per_token_s": 1.1e-5,
            "tail_s": 0.045,                 # [MEAS] one decode-class step for the last token
        },
        "min_rows": 4096,                    # M's TF_PREFILL_MIN: a prompt with fewer uncached rows is not worth a pull
        "max_export_tokens": 262144,         # [ASSUMED, gate G4] replicated KV pool of the export profile
        "mu": 1.0,                           # shadow price of one S prefill-second (SS and SM spend it, MM does not)
        "greedy_only_decode": True,          # [MEAS G2] M ignores temperature: sampled requests are S-only until M samples
        "s_only_fields": ["response_format", "guided_json", "guided_regex", "guided_grammar", "logprobs", "top_logprobs", "prompt_logprobs"],
        "exact_header": "X-Exactness",       # 'exact' -> never SM (the hand-off is exact_mode remote_prefix, not bit-exact with a native prefill)
        "fallback": {"failures": 3, "cooldown_s": 60.0},                    # SM requests that M prefilled itself (reply has no pulled_prefix): disable SM after N in a row
    },
    # Admission control: at most `backends.*.max_inflight` (default = slots: S 4, M 4) requests are in flight per backend, the router queues the rest
    # (interactive before background, then oldest) and binds each to a backend only when a lane frees (late binding, decided on fresh state).  A request whose
    # eligible backends all have `queue_max` waiters, or that waits `max_wait_s`, gets 503 + Retry-After.  E2: without it the router drove production past its slots.
    "admission": {"enabled": True, "max_wait_s": 120.0, "poll_s": 0.25, "retry_after_min_s": 1.0, "retry_after_max_s": 30.0},
    "backends": {"S": {}, "M": {}},
}


def _walk_check(defaults, user, path):
    for k, v in user.items():
        if k.startswith("_"):
            continue
        if k not in defaults:
            raise ValueError("unknown config key: " + ".".join(path + [k]))
        if isinstance(defaults[k], dict) and isinstance(v, dict) and k not in ("classes", "class_suffixes", "backends"):
            _walk_check(defaults[k], v, path + [k])


def _merge(dst, src):
    for k, v in src.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if not k.startswith("_")}
    return obj


def backend_cfg(name, user):
    """full backend config: common defaults <- preset <- user"""
    preset_name = user.get("preset") or (name if name in PRESETS else None)
    if preset_name is None:
        raise ValueError("backend %r needs \"preset\": \"S\" or \"M\" (or copy a full profile)" % name)
    if preset_name not in PRESETS:
        raise ValueError("backend %r: unknown preset %r" % (name, preset_name))
    cfg = copy.deepcopy(BACKEND_COMMON)
    _merge(cfg, _strip(PRESETS[preset_name]))
    _merge(cfg, user)
    cfg["name"] = name
    allowed = set(BACKEND_COMMON) | set(PRESET_S) | {"name"}
    for k in cfg:
        if k not in allowed:
            raise ValueError("unknown backend key %r in %r" % (k, name))
    return cfg


def load(path=None, overrides=None):
    """DEFAULTS <- JSON file <- overrides (dict); returns the merged config with resolved per-backend configs"""
    user = {}
    if path:
        with open(path) as f:
            user = json.load(f)
    if overrides:
        user = _merge(copy.deepcopy(user), overrides)
    _walk_check(DEFAULTS, user, [])
    cfg = copy.deepcopy(DEFAULTS)
    ubackends = user.get("backends", {})
    u2 = {k: v for k, v in user.items() if k != "backends"}
    _merge(cfg, u2)
    # the shipped default lists S and M, a user file that names other backends replaces the set
    names = list(ubackends) if ubackends else ["S", "M"]
    cfg["backends"] = {n: backend_cfg(n, ubackends.get(n, {})) for n in names}
    validate(cfg)
    return cfg


def validate(cfg):
    on = [b for b in cfg["backends"].values() if b["enabled"]]
    if not on:
        raise ValueError("no enabled backend")
    for b in on:
        if not b.get("url"):
            raise ValueError("backend %s has no url" % b["name"])
        if b["role"] not in ("primary", "secondary"):
            raise ValueError("backend %s: role must be primary or secondary" % b["name"])
    p = cfg["policy"]
    if p["mode"] not in ("cost", "ll"):
        raise ValueError("policy.mode must be cost or ll")
    if cfg["tokens"]["mode"] not in ("estimate", "auto", "exact"):
        raise ValueError("tokens.mode must be estimate, auto or exact")
    if cfg["default_class"] not in p["classes"]:
        raise ValueError("default_class is not a policy class")
    if cfg["mode"]["shadow"] and cfg["mode"]["shadow"] not in cfg["backends"]:
        raise ValueError("mode.shadow names no backend")
    if cfg["time_scale"] <= 0:
        raise ValueError("time_scale must be > 0")
    pd = cfg["pd"]
    if pd["enabled"]:
        for k in ("prefill_backend", "decode_backend"):
            if pd[k] not in cfg["backends"] or not cfg["backends"][pd[k]]["enabled"]:
                raise ValueError("pd.%s names no enabled backend" % k)
        if pd["prefill_backend"] == pd["decode_backend"]:
            raise ValueError("pd.prefill_backend and pd.decode_backend must differ")
    a = cfg["admission"]
    if a["max_wait_s"] <= 0 or a["poll_s"] <= 0 or a["retry_after_min_s"] < 0 or a["retry_after_max_s"] < a["retry_after_min_s"]:
        raise ValueError("admission: max_wait_s and poll_s must be > 0, 0 <= retry_after_min_s <= retry_after_max_s")
    for b in on:
        if b["max_inflight"] is not None and int(b["max_inflight"]) < 1:
            raise ValueError("backend %s: max_inflight must be >= 1 (or null: its slots)" % b["name"])
        if int(b["queue_max"]) < 0:
            raise ValueError("backend %s: queue_max must be >= 0" % b["name"])


def dump_defaults():
    """the shipped router.config.json: every default, backends without urls (a copy of this file with urls filled in is a valid config)"""
    d = copy.deepcopy(DEFAULTS)
    d["backends"] = {
        "S": {"_comment": "A: the two Sparks' production API. Fill url (e.g. http://HEAD:8888) and api_key_file; everything else is the measured preset below",
              "url": None, "api_key_file": None, "max_inflight": None, "queue_max": BACKEND_COMMON["queue_max"], **{k: v for k, v in PRESET_S.items() if not k.startswith("_")}},
        "M": {"_comment": "B: the M5 Ultra TensorFold server. Fill url (e.g. http://M5:8899) and api_key_file",
              "url": None, "api_key_file": None, "max_inflight": None, "queue_max": BACKEND_COMMON["queue_max"], **{k: v for k, v in PRESET_M.items() if not k.startswith("_")}},
    }
    d["_comment"] = "tfrouter policy parameters, measured defaults; see router/README.md. [MEAS]/[SIM]/[ASSUMED] tags are in tfrouter/config.py"
    return d
