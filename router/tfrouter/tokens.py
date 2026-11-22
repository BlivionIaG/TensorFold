"""Token counting for routing decisions.

Preferred: the backend's own tokenizer endpoint (TensorFold: POST /v1/tokenize -> {"count", "max_model_len", "tokens"}, docs/api.md; for
`messages` it counts what the chat template renders, tools included).  That costs one extra request carrying the whole prompt, so in
mode "auto" it is only used when the estimate is close to a decision boundary (a context limit, the M guard size).
Fallback: chars / chars_per_token + per-message overhead, with chars_per_token calibrated online from usage.prompt_tokens of every
completed request and from every tokenizer answer (EWMA).
"""
import json
import math
from collections import OrderedDict


def content_chars(content):
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        n = 0
        for p in content:
            if isinstance(p, dict):
                n += len(p.get("text", "")) if p.get("type", "text") == "text" else 0
            else:
                n += len(str(p))
        return n
    return len(str(content))


def request_size(endpoint, body):
    """(chars, n_messages) of a chat or completions body"""
    if endpoint == "chat":
        msgs = body.get("messages") or []
        chars = sum(content_chars(m.get("content")) if isinstance(m, dict) else len(str(m)) for m in msgs)
        for m in msgs:
            if isinstance(m, dict) and m.get("tool_calls"):
                chars += len(json.dumps(m["tool_calls"]))
        if body.get("tools"):
            chars += len(json.dumps(body["tools"]))
        return chars, len(msgs)
    p = body.get("prompt", "")
    if isinstance(p, list):
        if p and isinstance(p[0], int):             # token ids: exact
            return -len(p), 0
        p = "".join(x for x in p if isinstance(x, str))
    return len(p), 0


class TokenEstimator:
    def __init__(self, cfg):
        t = cfg["tokens"]
        self.mode = t["mode"]
        self.cpt = {}                                   # model key -> chars per token
        self.cpt0 = float(t["chars_per_token"])
        self.msg_over = int(t["msg_overhead_tokens"])
        self.base_over = int(t["base_overhead_tokens"])
        self.alpha = float(t["calibration_alpha"])
        self.margin = float(t["exact_margin"])
        self.exact_timeout = float(t["exact_timeout_s"])
        self.out_alpha = float(t["output_alpha"])
        self.out_default = float(t["default_output_tokens"])
        self.out = {}                                   # class -> EWMA of completion tokens
        self.out_n = {}                                 # class -> samples seen
        self.out_warmup = bool(t["output_warmup"])
        self.cache = OrderedDict()
        self.cache_max = int(t["exact_cache"])
        self.samples = 0
        self.exact_calls = self.exact_fail = 0

    def cpt_of(self, key="default"):
        return self.cpt.get(key, self.cpt0)

    def estimate(self, chars, nmsgs, key="default"):
        if chars < 0:                                    # token ids
            return -chars
        return int(math.ceil(chars / self.cpt_of(key))) + nmsgs * self.msg_over + self.base_over

    def observe(self, chars, nmsgs, tokens, key="default"):
        """a ground-truth (chars, tokens) pair: usage.prompt_tokens or a tokenizer answer"""
        if chars < 256 or tokens < 64:
            return
        body_tokens = tokens - nmsgs * self.msg_over - self.base_over
        if body_tokens <= 0:
            return
        sample = chars / body_tokens
        if not 0.5 <= sample <= 12.0:                    # tool schemas, code, other languages: still bounded
            return
        old = self.cpt_of(key)
        self.cpt[key] = old + self.alpha * (sample - old)
        self.samples += 1

    def near_boundary(self, n_est, limits):
        """True when the estimate is within `margin` of any limit (context sizes, guard rows): worth an exact count"""
        for lim in limits:
            if lim and abs(n_est - lim) <= self.margin * lim:
                return True
        return False

    def want_exact(self, n_est, limits):
        if self.mode == "exact":
            return True
        if self.mode == "auto":
            return self.near_boundary(n_est, limits)
        return False

    def cache_get(self, key):
        v = self.cache.get(key)
        if v is not None:
            self.cache.move_to_end(key)
        return v

    def cache_put(self, key, n):
        self.cache[key] = n
        while len(self.cache) > self.cache_max:
            self.cache.popitem(last=False)

    # ---- output length ------------------------------------------------------------------------------------------------
    def out_est(self, cls, max_tokens, hint=None):
        if hint:
            return max(1, int(hint))
        ew = self.out.get(cls, self.out_default)
        if max_tokens:
            return max(1, int(min(max_tokens, ew)))
        return max(1, int(ew))

    def observe_out(self, cls, n):
        """completion tokens of a finished request.  The first samples are a running mean (the start value is a guess and carries no weight):
        with alpha 0.1 an EWMA needed ~25 requests to leave the 300-token start value for a 512-token workload (E2), mispricing every decode term meanwhile"""
        if n <= 0:
            return
        k = self.out_n.get(cls, 0)
        self.out_n[cls] = k + 1
        a = max(self.out_alpha, 1.0 / (k + 1)) if self.out_warmup else self.out_alpha
        old = self.out.get(cls, self.out_default)
        self.out[cls] = old + a * (n - old)
