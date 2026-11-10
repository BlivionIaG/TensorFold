"""Metal throughput of Qwen3.6-35B-A3B's routed experts on this Mac, random weights, real shapes. No eGPU.

run: /tmp/mlxenv/bin/python mlx_expert_probe.py   (mlx 0.32.4; `uv venv /tmp/mlxenv && uv pip install mlx`)
Shapes from the checkpoint: 256 experts, top-8, hidden 2048, moe_intermediate 512, MLX affine 4-bit group 64.
A layer's routed block is gate+up (N=512,K=2048) and down (N=2048,K=512) over the 8 picked experts; the chain
below feeds each layer's output to the next so the GPU work is serial, as in a decode step.
"""
import statistics
import sys
import time

import mlx.core as mx

E, K, N, TOPK, GS, BITS = 256, 2048, 512, 8, 64, 4
LAYERS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SETS = 32  # distinct pick sets per layer, so a repeat never hits the system cache


def qweights(n, k):
    w = mx.random.randint(0, 2**31 - 1, (E, n, k * BITS // 32), dtype=mx.int32).astype(mx.uint32)
    s = (mx.random.uniform(shape=(E, n, k // GS)) * 0.01 + 0.001).astype(mx.bfloat16)
    b = (mx.random.uniform(shape=(E, n, k // GS)) * 0.001).astype(mx.bfloat16)
    return w, s, b


def expert_bytes(n, k):
    return n * k // 2 + 2 * 2 * n * (k // GS)


per_expert = 2 * expert_bytes(N, K) + expert_bytes(K, N)
print(f"bytes per expert {per_expert} ({per_expert / 2**20:.4f} MiB); per layer 8 experts {TOPK * per_expert / 2**20:.2f} MiB")

layers = []
for _ in range(LAYERS):
    g, u, d = qweights(N, K), qweights(N, K), qweights(K, N)
    layers.append((g, u, d))
mx.eval([t for l in layers for p in l for t in p])
picks = [[mx.array(sorted(__import__("random").sample(range(E), TOPK)), dtype=mx.uint32).reshape(1, 1, TOPK) for _ in range(SETS)] for _ in range(LAYERS)]
mx.eval(picks)


def gqm(x, p, idx):
    w, s, b = p
    return mx.gather_qmm(x, w, s, b, rhs_indices=idx, transpose=True, group_size=GS, bits=BITS)


def layer(x, l, idx):
    g, u, d = layers[l]
    xe = mx.expand_dims(x, (-2, -3))  # (1,1,1,1,K)
    a = gqm(xe, g, idx)
    c = gqm(xe, u, idx)
    h = (a * mx.sigmoid(a)) * c  # (1,1,8,1,N)
    y = gqm(h, d, idx)  # (1,1,8,1,K)
    return x + y.squeeze(-2).sum(axis=-2) * 0.01


def step(x, it):
    for l in range(LAYERS):
        x = layer(x, l, picks[l][it % SETS])
    return x


x0 = mx.random.normal((1, 1, K)).astype(mx.bfloat16)
for it in range(4):
    mx.eval(step(x0, it))  # warm-up and pipeline compile
times = []
for it in range(60):
    t0 = time.perf_counter()
    mx.eval(step(x0, it + 4))
    times.append(time.perf_counter() - t0)
med = statistics.median(times)
nbytes = LAYERS * TOPK * per_expert
print(f"{LAYERS}-layer routed chain, decode row: median {med * 1e3:.2f} ms (min {min(times) * 1e3:.2f}, max {max(times) * 1e3:.2f}), "
      f"{nbytes / 2**20:.0f} MiB weights -> {nbytes / med / 1e9:.1f} GB/s, {med / LAYERS * 1e6:.0f} us per layer")

# one layer alone, synced each time (what a per-layer crossing would see if Metal were invoked a layer at a time)
one = []
for it in range(300):
    t0 = time.perf_counter()
    mx.eval(layer(x0, it % LAYERS, picks[it % LAYERS][it % SETS]))
    one.append(time.perf_counter() - t0)
print(f"single layer, eval() each (graph build + submit + wait): median {statistics.median(one[20:]) * 1e6:.0f} us, p10 {sorted(one[20:])[28] * 1e6:.0f}, p90 {sorted(one[20:])[-28] * 1e6:.0f}")

# 4 rows (an MTP verify window): up to 4*8 distinct experts, 4 independent rows
rows = 4
xr = mx.random.normal((1, rows, K)).astype(mx.bfloat16)
pr = [mx.array([sorted(__import__("random").sample(range(E), TOPK)) for _ in range(rows)], dtype=mx.uint32).reshape(1, rows, TOPK) for _ in range(SETS)]
mx.eval(pr)


def layer_rows(x, l, idx):
    g, u, d = layers[l]
    xe = mx.expand_dims(x, (-2, -3))
    a = gqm(xe, g, idx); c = gqm(xe, u, idx); h = (a * mx.sigmoid(a)) * c; y = gqm(h, d, idx)
    return x + y.squeeze(-2).sum(axis=-2) * 0.01


def stepr(x, it):
    for l in range(LAYERS):
        x = layer_rows(x, l, pr[(it + l) % SETS])
    return x


for it in range(4):
    mx.eval(stepr(xr, it))
tr = []
for it in range(40):
    t0 = time.perf_counter(); mx.eval(stepr(xr, it + 4)); tr.append(time.perf_counter() - t0)
medr = statistics.median(tr)
print(f"{rows}-row window ({rows * TOPK} (row,expert) pairs a layer, random picks, gather_qmm per pair): median {medr * 1e3:.2f} ms for {LAYERS} layers -> {medr / LAYERS * 1e6:.0f} us per layer")
