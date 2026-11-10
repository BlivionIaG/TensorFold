"""What an external GPU must do to beat the Mac alone serving N sequences of Qwen3.6-35B-A3B (docs/proposals/metal-tb4-split-moe.md).

Pure arithmetic on top of placement_model.py. The Mac alone is a measurement (mlx_lm 0.32.2 batch_generate, different prompts,
M4 Pro, tok/s over N sequences): a round of N rows costs t(N) = a + b N; a is what every round reads once (attention and
DeltaNet mixers, shared experts, routers, embedding, head, state, launches), b the routed experts one more row brings (8 of 256
a layer, almost no overlap between different rows). Everything about the card is a model [INFERENCE]: it runs the same
bytes at r times the Mac's speed, from the planner's per-token model (placement_model.tdev).
run: python3 batched_split_model.py
"""
from placement_model import GiB, MAC, MiB, PROFILES, Gpu, fits_gpu, qwen36_a3b, tdev

# measured on the Mac (results in card_streams.txt): N -> decode tok/s, the mean of the runs
MEASURED = {1: 88.8, 2: 135.8, 4: 176.4, 8: 206.9}


def fit(measured):
    """Least squares t(N) = a + b N over the measured rounds (seconds)."""
    pts = [(n, n / tps) for n, tps in measured.items()]
    k = len(pts)
    sx = sum(n for n, _ in pts)
    sy = sum(t for _, t in pts)
    sxx = sum(n * n for n, _ in pts)
    sxy = sum(n * t for n, t in pts)
    b = (k * sxy - sx * sy) / (k * sxx - sx * sx)
    return (sy - b * sx) / k, b


A, B = fit(MEASURED)


def mac(n):
    return A + B * n


def resident(n, r):
    """The whole model on the card: it batches like the Mac, r times as fast. Only tokens cross."""
    return n / (mac(n) / r + 100e-6)


def split_batched(n, r, f_max):
    """Two microbatches of n/2 rows, the card taking a fraction f of the layers: each side's round is its share of a + b n/2.
    Throughput is n over twice the slower side's round. Needs stages that batch rows of different sequences (not implemented)."""
    n2 = max(n // 2, 1)
    best = (0, 0)
    for i in range(1, 100):
        f = min(i / 100, f_max)
        tc = f * mac(n2) / r + 160e-6 + 15e-6
        tm = (1 - f) * mac(n2)
        t = max(tc, tm)
        tps = (n if n > 1 else 1) / (2 * t) if n > 1 else 1 / (tc + tm)
        if tps > best[0]:
            best = (tps, f)
    return best


def split_single_row(n, r, f_max):
    """What is implemented: every step is one row of one sequence, so the slower side's one-row round bounds the whole."""
    best = (0, 0)
    for i in range(1, 100):
        f = min(i / 100, f_max)
        tc = f * mac(1) / r + 160e-6 + 15e-6
        tm = (1 - f) * mac(1)
        tps = 1 / max(tc, tm) if n > 1 else 1 / (tc + tm)
        if tps > best[0]:
            best = (tps, f)
    return best


def speed_ratio(g: Gpu, m):
    """r: how much faster than the Mac the card decodes the whole model for one row (placement_model's per-token model)."""
    return tdev(m, range(len(m.layers)), MAC, 512, True) / tdev(m, range(len(m.layers)), g, 512, True)


def f_max(g: Gpu, m, ctx, streams):
    """Largest fraction of the layers the card can hold with the KV and state of `streams` sequences, the embedding and workspace."""
    per_layer = sum(l.resident + l.experts for l in m.layers) / len(m.layers)
    kv = sum(l.kv_per_tok * ctx * streams + l.state * streams for l in m.layers)
    room = g.mem - m.scratch - m.embed - kv
    return max(0.0, min(1.0, room / (per_layer * len(m.layers))))


def break_even_r(n):
    """Resident needs r > 1. A split of single-row steps cannot beat the Mac's batched round at all once N rows share its weights;
    a split of batched stages beats the Mac at r above this (card taking the best fraction, free crossings)."""
    lo, hi = 0.01, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if split_batched(n, mid, 1.0)[0] > n / mac(n):
            hi = mid
        else:
            lo = mid
    return hi


def validate_4b():
    """The same model against the card runs of the 4B (card_streams.txt): the Mac alone is TensorFold Metal's batched decode
    (70.1, 116.5, 154.7, 199.0 tok/s), the card r = 14.1 ms / 19.1 ms (its unsplit position times); the card ran cuts 8-24 of 32."""
    pts = {1: 70.5, 2: 116.3, 3: 154.7, 4: 199.0}
    a, b = fit(pts)
    r = 14.1 / 19.1
    print(f"validation on the 4B: Mac alone a = {a * 1e3:.2f} ms, b = {b * 1e3:.2f} ms; r = {r:.2f}")
    for n in (1, 2, 3, 4):
        best = 0.0
        for i in range(1, 100):
            f = i / 100
            tc, tm = f * (a + b) / r + 175e-6, (1 - f) * (a + b)  # single-row steps, the implemented split
            best = max(best, 1 / max(tc, tm) if n > 1 else 1 / (tc + tm))
        print(f"  N={n}: Mac alone {pts[n]:.0f} | single-row split model {best:.0f} tok/s | measured on the card: {({1: 53, 2: 130, 3: 120, 4: 120})[n]} (best cut, 4B)")


if __name__ == "__main__":
    validate_4b()
    print()
    m = qwen36_a3b()
    print(f"Mac alone fit: a = {A * 1e3:.2f} ms a round, b = {B * 1e3:.2f} ms a row (routed experts: 8 x 40 x 1.6875 MiB = 540 MiB -> {540 * MiB / B / 1e9:.0f} GB/s)")
    for n in (1, 2, 4, 8):
        print(f"  N={n}: measured {MEASURED[n]:.1f} tok/s, fit {n / mac(n):.1f}")
    print("\nbreak-even speed of the card (r = card's whole-model one-row decode speed over the Mac's, 88.8 tok/s):")
    print("  resident (the card holds the model): r > 1")
    for n in (2, 4, 8):
        print(f"  split, batched stages, N={n}: r > {break_even_r(n):.2f} (needs the card to hold the fraction it is given; implemented single-row split: never beats {1 / mac(1):.0f}+ tok/s)")
    for g in PROFILES:
        r = speed_ratio(g, m)
        print(f"\n{g.name}: r = {r:.2f} ({1 / tdev(m, range(40), g, 512, True):.0f} tok/s one row, model)")
        for n in (1, 2, 4, 8):
            fm = f_max(g, m, 8192, n)
            line = f"  N={n}: Mac alone {n / mac(n):5.0f}"
            if fm >= 1.0:
                line += f" | resident {resident(n, r):5.0f} ({resident(n, r) / (n / mac(n)):.2f}x)"
            else:
                line += f" | resident does not fit (the card holds {fm * 40:.0f} of 40 layers)"
            if n > 1:
                sb, fb = split_batched(n, r, min(fm, 1.0))
                ss, fs = split_single_row(n, r, min(fm, 1.0))
                line += f" | split batched {sb:5.0f} ({sb / (n / mac(n)):.2f}x, card {fb * 40:.0f} layers) | split single-row (implemented) {ss:5.0f} ({ss / (n / mac(n)):.2f}x)"
            print(line)
