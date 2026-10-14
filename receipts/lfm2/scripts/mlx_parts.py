"""mlx-lm LFM2 prefill timed by part: each module's output evaluated and timed, summed over 2,048-token chunks."""
import json, sys, time
from collections import defaultdict
import mlx.core as mx
import mlx_lm.models.lfm2_moe as L
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

model, tok = load(sys.argv[1])
n = int(sys.argv[2])
ids = json.load(open(sys.argv[3]))[:n]
times = defaultdict(float)
on = [False]

def timed(name, fn):
    def run(*a, **k):
        if not on[0]:
            return fn(*a, **k)
        mx.eval(a[0] if a and isinstance(a[0], mx.array) else [])
        t = time.perf_counter()
        y = fn(*a, **k)
        mx.eval(y)
        times[name] += time.perf_counter() - t
        return y
    return run

sdpa = L.scaled_dot_product_attention
L.scaled_dot_product_attention = timed("attention kernel (sdpa)", sdpa)
for i, layer in enumerate(model.layers):
    if layer.is_attention_layer:
        layer.self_attn.q_proj.__call__  # noqa
        for p in ("q_proj", "k_proj", "v_proj", "out_proj"):
            m = getattr(layer.self_attn, p)
            setattr(layer.self_attn, p, type("T", (), {"__call__": staticmethod(timed("attention projections", m.__call__))})())
    else:
        c = layer.conv
        for p in ("in_proj", "out_proj"):
            m = getattr(c, p)
            setattr(c, p, type("T", (), {"__call__": staticmethod(timed("conv projections", m.__call__))})())
        c.conv = type("T", (), {"__call__": staticmethod(timed("conv kernel", c.conv.__call__))})()
    ff = layer.feed_forward
    name = "dense FFN" if isinstance(ff, L.MLP) else "experts (router + switch_mlp + combine)"
    layer.feed_forward = type("T", (), {"__call__": staticmethod(timed(name, ff.__call__))})()
    for p in ("operator_norm", "ffn_norm"):
        m = getattr(layer, p)
        setattr(layer, p, type("T", (), {"__call__": staticmethod(timed("norms", m.__call__))})())

def prefill(timing):
    cache = make_prompt_cache(model)
    on[0] = timing
    t = time.perf_counter()
    for at in range(0, n, 2048):
        y = model(mx.array(ids[at:at + 2048])[None], cache=cache)
        mx.eval(y, [c.state for c in cache])
    total = time.perf_counter() - t
    on[0] = False
    mx.clear_cache()
    return total

prefill(False)  # warm up kernels
plain = prefill(False)
times.clear()
parts_total = prefill(True)
print(json.dumps({"tokens": n, "plain_s": round(plain, 3), "plain_tok_s": round(n / plain, 1), "timed_run_s": round(parts_total, 3),
                  "parts_s": {k: round(v, 3) for k, v in sorted(times.items(), key=lambda x: -x[1])}}, indent=1))
