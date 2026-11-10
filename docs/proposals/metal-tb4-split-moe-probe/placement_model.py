"""Placement planner sketch and decode tok/s model for the TB4 offload proposal (docs/proposals/metal-tb4-split-moe.md).

Pure arithmetic, no device. Every constant is either a measurement recorded in results.txt / the proposal (M),
a published peak (P), or an assumption (A) the first prototype milestones must replace.
run: python3 placement_model.py            (prints the tables quoted in the proposal)
"""
from dataclasses import dataclass

MiB, GiB = 2**20, 2**30


def q4(n, k, bits=4, g=64):  # MLX affine: packed codes + bf16 scale + bf16 bias per group
    return n * k * bits // 8 + n * (k // g) * 4


@dataclass
class Layer:
    resident: int  # weights that must be somewhere
    dense: int  # attention/mixer/shared/router bytes read per token
    experts: int  # routed-expert bytes resident (0 for dense models)
    experts_read: int  # routed-expert bytes read per decode token (top-k)
    kv_per_tok: int  # KV bytes per context token (read per token and resident)
    state: int  # recurrent state bytes (resident; read+written per token)
    dispatches: int  # kernel launches per token on a launch-bound device


@dataclass
class Model:
    name: str
    layers: list
    embed: int
    head: int
    scratch: int = 400 * MiB

    def weights(self):
        return sum(l.resident + l.experts for l in self.layers) + self.embed + self.head


def qwen36_a3b():  # mlx-community/Qwen3.6-35B-A3B-4bit @38740b8: config.json geometry
    H, E, K, I, SI, V = 2048, 256, 8, 512, 512, 248320
    shared = 3 * q4(SI, H) + q4(1, H, 8)
    router = q4(E, H, 8)
    expert = 3 * q4(I, H)
    lin = q4(8192, H) + q4(4096, H) + q4(H, 4096) + 2 * q4(32, H) + 8192 * 4 * 2
    full = q4(8192, H) + 2 * q4(512, H) + q4(H, 4096)
    ls = []
    for i in range(40):
        att = full if i % 4 == 3 else lin
        st = 0 if i % 4 == 3 else 32 * 128 * 128 * 4 + 3 * 8192 * 2
        ls.append(Layer(att + shared + router + 4 * 2048 * 2, att + shared + router, E * expert, K * expert,
                        2048 if i % 4 == 3 else 0, st, 22))
    return Model("Qwen3.6-35B-A3B 4-bit", ls, q4(V, H), q4(V, H))


def qwen35_4b():  # mlx-community/Qwen3.5-4B-MLX-4bit @32f3e8e: hidden 2560, mlp 9216, 24 DeltaNet + 8 attention, tied head
    H, I, V = 2560, 9216, 248320
    mlp = 3 * q4(I, H)
    lin = q4(8192, H) + q4(4096, H) + q4(H, 4096) + 2 * q4(32, H) + 8192 * 4 * 2
    full = q4(8192, H) + 2 * q4(1024, H) + q4(H, 4096)
    ls = []
    for i in range(32):
        att = full if i % 4 == 3 else lin
        st = 0 if i % 4 == 3 else 32 * 128 * 128 * 4 + 3 * 8192 * 2
        ls.append(Layer(att + mlp + 2 * H * 2, att + mlp, 0, 0, 4096 if i % 4 == 3 else 0, st, 12))
    return Model("Qwen3.5-4B 4-bit (dense)", ls, q4(V, H), 0, 300 * MiB)  # tied: the head reads the embedding table


@dataclass
class Gpu:
    name: str
    mem: float  # usable bytes
    peak: float  # B/s
    eff: float  # fraction of peak the decode kernels reach end to end
    launch: float  # s per kernel launch that cannot overlap


# Mac: M4 Pro, measured end to end: 91.6 tok/s on the A3B and 66 tok/s on the 4B (mlx_lm 0.32.2) = 164 and 180 GB/s of weights+state
MAC = Gpu("M4 Pro (Metal)", 37.4 * GiB, 273e9, 0.62, 0)
PROFILES = [
    Gpu("RX Vega 56 8 GiB (today)", 7.9 * GiB, 410e9, 0.37, 4.4e-6),  # eff: 4B whole token 135 GB/s incl. launches (M)
    Gpu("RX 7900 XTX 24 GiB", 23.0 * GiB, 960e9, 0.50, 2.5e-6),  # A
    Gpu("RTX 5090 32 GiB", 31.0 * GiB, 1792e9, 0.55, 1.5e-6),  # A, GB10 decode reaches 0.55 of its peak (recipe: 88 tok/s)
]
# crossing costs per direction (M = measured in results.txt, A = assumed until M1/M2 on the card)
TO_MAC = 90e-6  # eGPU flag visible to the host (5) + event signal to a pre-committed command buffer and wake (65) + flush (20)   M+A
TO_GPU = 15e-6  # host store + doorbell/poll on the eGPU side                                                                      A
STATE_RW = 2.0  # recurrent state is read and written once per token


def tdev(model, idx, g, ctx, head=False):
    b = 0.0
    n = 0
    for i in idx:
        l = model.layers[i]
        b += l.dense + l.experts_read + l.kv_per_tok * ctx + STATE_RW * l.state
        n += l.dispatches
    if head:
        b += model.head or model.embed
        n += 3
    return b / (g.peak * g.eff) + n * g.launch


def fits_gpu(model, idx, g, ctx, streams, head, embed):
    w = sum(model.layers[i].resident + model.layers[i].experts + model.layers[i].kv_per_tok * ctx * streams + model.layers[i].state * streams for i in idx)
    w += model.scratch + (model.embed if embed else 0) + (model.head if head else 0)
    return w, w <= g.mem


def plan(model, gpu, ctx, mac_budget=MAC.mem, streams=1):
    L = len(model.layers)
    out = []
    allmac = tdev(model, range(L), MAC, ctx, True)
    w, ok = fits_gpu(model, range(L), MAC, ctx, streams, True, True)
    out.append(("all-Metal", L, 0, allmac, 1 / allmac, w <= mac_budget, f"Mac {w / GiB:.1f} GiB"))
    # fully resident on the eGPU, the Mac only hosts tokenizer and sampling bookkeeping
    w, ok = fits_gpu(model, range(L), gpu, ctx, streams, True, True)
    t = tdev(model, range(L), gpu, ctx, True) + 30e-6
    out.append(("eGPU-resident", 0, L, t, 1 / t, ok, f"VRAM {w / GiB:.1f} GiB"))
    # layer pipeline: eGPU takes the first k layers (+ embedding), the Mac the rest (+ head); one hop each way
    best = None
    bestp = None
    for k in range(1, L):
        w, ok = fits_gpu(model, range(k), gpu, ctx, streams, False, True)
        wm, okm = fits_gpu(model, range(k, L), MAC, ctx, streams, True, False)
        if not ok or wm > mac_budget:
            continue
        tg, tm = tdev(model, range(k), gpu, ctx), tdev(model, range(k, L), MAC, ctx, True)
        t = tg + tm + TO_MAC + TO_GPU
        c = ("pipeline", L - k, k, t, 1 / t, True, f"{k} layers on the eGPU, VRAM {w / GiB:.1f} GiB, Mac {wm / GiB:.1f} GiB; stages {tg * 1e3:.1f}+{tm * 1e3:.1f} ms")
        if best is None or t < best[3]:
            best = c
        pp = 1 / max(tg, tm)
        if bestp is None or pp > bestp[0]:
            bestp = (pp, c)
    if best:
        out.append(best)
        out.append(("pipeline, 2+ streams", bestp[1][1], bestp[1][2], 1 / bestp[0], bestp[0], True, bestp[1][6]))
    if model.layers[0].experts:
        # design A: eGPU = everything except routed experts of the layers it cannot hold; those experts live on the Mac
        dense_w = sum(l.resident for l in model.layers) + model.embed + model.head
        best = None
        for r in range(0, L + 1):
            need = dense_w + sum(model.layers[i].experts for i in range(r)) + sum(model.layers[i].kv_per_tok * ctx * streams + model.layers[i].state * streams for i in range(L)) + model.scratch
            if need > gpu.mem or (L - r) * model.layers[0].experts > mac_budget:
                continue
            t = tdev(model, range(L), gpu, ctx, True)
            for i in range(r, L):  # layer i sends x, ids; the Mac streams the top-k experts (3 launches); results come back
                t += TO_MAC + TO_GPU + model.layers[i].experts_read / (MAC.peak * MAC.eff) - model.layers[i].experts_read / (gpu.peak * gpu.eff)
            if best is None or t < best[3]:
                best = ("expert split (A)", L - r, r, t, 1 / t, True, f"VRAM {need / GiB:.1f} GiB, {r} expert layers resident, {40 - r if L == 40 else L - r} crossings")
        if best:
            out.append(best)
        # reverse: the Mac keeps attention/dense/KV; the eGPU holds routed experts for as many layers as fit
        mac_dense = sum(l.resident for l in model.layers) + model.embed + model.head + sum(l.kv_per_tok * ctx * streams + l.state * streams for l in model.layers)
        best = None
        for r in range(1, L + 1):
            need = r * model.layers[0].experts + model.scratch
            rest = (L - r) * model.layers[0].experts
            if need > gpu.mem or mac_dense + rest > mac_budget:
                continue
            t = tdev(model, range(L), MAC, ctx, True) - sum(model.layers[i].experts_read for i in range(r)) / (MAC.peak * MAC.eff)
            for i in range(r):
                t += 45e-6 + TO_GPU + TO_MAC - 15e-6 + model.layers[i].experts_read / (gpu.peak * gpu.eff) + 4 * gpu.launch
            if best is None or t < best[3]:
                best = ("reverse (Mac attention, eGPU experts)", r, r, t, 1 / t, True, f"VRAM {need / GiB:.1f} GiB, {r} expert layers on the eGPU, {2 * r} crossings")
        if best:
            out.append(best)
    return out


def show(model, ctxs=(8192, 131072), mac_budget=MAC.mem, label="", streams=1):
    print(f"\n### {model.name}: weights {model.weights() / GiB:.2f} GiB; Mac working-set budget {mac_budget / GiB:.0f} GiB{label}")
    for g in PROFILES:
        for ctx in ctxs:
            rows = plan(model, g, ctx, mac_budget, streams)
            print(f"{g.name}, ctx {ctx // 1024}k, streams {streams}:")
            for name, ngpu_layers_or_mac, on_gpu, t, tps, ok, note in rows:
                if not ok:
                    print(f"  {name:38s} does not fit ({note})")
                    continue
                print(f"  {name:38s} {tps:6.0f} tok/s ({t * 1e3:5.1f} ms)  {note}")


if __name__ == "__main__":
    a3b, d4 = qwen36_a3b(), qwen35_4b()
    # calibration against this Mac's measurements
    for m, meas in ((a3b, 91.6), (d4, 66.0)):
        t = tdev(m, range(len(m.layers)), MAC, 512, True)
        print(f"calibration {m.name}: model {1 / t:.0f} tok/s, measured {meas} tok/s")
    t = tdev(d4, range(32), PROFILES[0], 512, True)
    print(f"calibration 4B on Vega 56: model {1 / t:.0f} tok/s, measured 52-57 tok/s (recipe)")
    # reverse split pays when the experts' read time saved on the eGPU beats its two crossings (45 us Mac->eGPU, 90 us eGPU->Mac)
    for g in PROFILES:
        x = (45e-6 + TO_GPU + TO_MAC + 4 * g.launch) / (1 / (MAC.peak * MAC.eff) - 1 / (g.peak * g.eff))
        if x < 0:
            print(f"reverse break-even on {g.name}: never, its decode bandwidth is below the Mac's")
        else:
            print(f"reverse break-even on {g.name}: routed bytes read per layer per token >= {x / MiB:.0f} MiB (Qwen3.6-A3B reads 13.5 MiB)")
    show(a3b)
    show(a3b, ctxs=(32768,), mac_budget=16 * GiB, label=" (24 GiB Mac)")
    show(d4, ctxs=(8192,))
