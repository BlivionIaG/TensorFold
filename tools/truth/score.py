#!/usr/bin/env python3
"""Score candidate logits against truth logits: KL(truth || cand), top-1, max |logit error|, perplexity.

Usage: score.py <truth.npy> <candidate.npy> [more...] [--ids ids.npy]
Candidates are f16/bf16/f32 npy (bf16 as uint16 in a `.bf16.npy` file). Next-token targets for
perplexity come from `<truth>.ids.npy` (written by truth.py) unless --ids is given.
"""
import argparse
from pathlib import Path

import numpy as np

CHUNK = 128  # rows per float64 block


def load(path: str) -> np.ndarray:
    a = np.load(path, mmap_mode="r")
    if path.endswith(".bf16.npy") or a.dtype == np.uint16:
        return (np.asarray(a).astype(np.uint32) << 16).view(np.float32)
    return a


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def score(truth: np.ndarray, cand: np.ndarray, targets: np.ndarray | None) -> dict:
    rows = truth.shape[0]
    kl, agree, err = np.empty(rows), 0, 0.0
    nll_t, nll_c = [], []
    for a in range(0, rows, CHUNK):
        t, c = truth[a:a + CHUNK], cand[a:a + CHUNK]
        lt, lc = log_softmax(t), log_softmax(c)
        kl[a:a + len(t)] = (np.exp(lt) * (lt - lc)).sum(-1)
        agree += int((t.argmax(-1) == c.argmax(-1)).sum())
        err = max(err, float(np.abs(t.astype(np.float64) - c.astype(np.float64)).max()))
        if targets is not None:
            tg = targets[a:a + len(t)]
            keep = tg >= 0
            r = np.arange(len(t))[keep]
            nll_t.append(-lt[r, tg[keep]]), nll_c.append(-lc[r, tg[keep]])
    out = dict(rows=rows, kl_mean=kl.mean(), kl_max=kl.max(), top1=100.0 * agree / rows, max_abs=err)
    if targets is not None:
        out["ppl"] = float(np.exp(np.concatenate(nll_c).mean()))
        out["ppl_truth"] = float(np.exp(np.concatenate(nll_t).mean()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("truth"), ap.add_argument("candidates", nargs="+"), ap.add_argument("--ids")
    a = ap.parse_args()
    truth = load(a.truth)
    ids_path = Path(a.ids or a.truth.removesuffix(".npy") + ".ids.npy")
    targets = None
    if ids_path.exists():
        ids = np.load(ids_path).reshape(-1)
        targets = np.append(ids[1:], -1)  # row i predicts token i+1; the last row has no target
    else:
        print(f"no ids at {ids_path}: perplexity skipped")
    for path in a.candidates:
        cand = load(path)
        if cand.shape[0] != truth.shape[0] or cand.shape[1] < truth.shape[1]:
            raise SystemExit(f"{path}: shape {cand.shape} does not match truth {truth.shape}")
        if cand.shape[1] > truth.shape[1]:
            print(f"{path}: padded vocab {cand.shape[1]} sliced to {truth.shape[1]}")
            cand = cand[:, :truth.shape[1]]
        r = score(truth, cand, targets)
        line = (f"{path}: rows={r['rows']} KL mean={r['kl_mean']:.3e} max={r['kl_max']:.3e} "
                f"top1={r['top1']:.2f}% max|dlogit|={r['max_abs']:.4g}")
        if "ppl" in r:
            line += f" ppl truth={r['ppl_truth']:.4f} cand={r['ppl']:.4f}"
        print(line)


if __name__ == "__main__":
    main()
