"""Decode-path and prompt-path logits of each tree against the HF fp32 reference, per sequence and pooled.
usage: score.py LABEL..."""
import sys, os
import numpy as np
A = os.path.expanduser("$REFS"); W = os.path.expanduser("$WORK/logs"); N = os.path.expanduser("$REFS")
SEQS = {"chat_full": 27, "code_full": 15, "story_full": 15, "chat_long": 27, "code_long": 15}
V = 131072
def ref_of(seq):
    p = f"{A}/ref_{seq}.npy" if seq.endswith("long") else f"{N}/ref_{seq}.npy"
    return np.load(p, mmap_mode="r") if os.path.exists(p) else None
def stats(r, h):
    best = r.argmax(1); second = np.argsort(r, axis=1)[:, -2]
    i = np.arange(len(r))
    marg = h[i, best] - h[i, second]
    err = np.abs(r - h)
    return dict(rows=len(r), agree=int((best == h.argmax(1)).sum()), mean=float(err.mean()), max=float(err.max()),
                worst_margin=float(marg.min()), ref_margin=float((r[i, best] - r[i, second]).min()), err=err)
for label in sys.argv[1:]:
    for kind in ("dec", "pro"):
        for tree in ("final", "rounds"):
            pooled = []; line = []
            for seq, p in SEQS.items():
                f = (f"{W}/{kind}_{label}_{tree}_{seq}.bin" if tree == "final" else f"{A}/{kind}_{label}_{tree}_{seq}.bin"); ref = ref_of(seq)
                if ref is None or not os.path.exists(f): continue
                h = np.fromfile(f, dtype=np.float32).reshape(-1, V).astype(np.float64)
                if kind == "pro": h = h[p:]
                r = np.asarray(ref[p:p + len(h)], dtype=np.float64)
                if len(r) != len(h): line.append(f"{seq}: rows {len(h)} vs ref {len(r)}"); continue
                s = stats(r, h)
                # a long sequence holds the full one's first rows: the pool takes each row once
                if not (seq.endswith("full") and os.path.exists(f.replace(seq, seq[:-4] + "long")) and ref_of(seq[:-4] + "long") is not None):
                    pooled.append((r, h))
                line.append(f"{seq} {s['agree']}/{s['rows']} mean {s['mean']:.4f} max {s['max']:.3f} worst margin {s['worst_margin']:.3f} (ref {s['ref_margin']:.3f})")
            if not pooled: continue
            r = np.concatenate([x for x, _ in pooled]); h = np.concatenate([y for _, y in pooled]); s = stats(r, h)
            print(f"{label} {kind} {tree:6s} ALL {s['agree']}/{s['rows']} top-1, mean |err| {s['mean']:.4f}, max {s['max']:.3f}, worst margin {s['worst_margin']:.3f}")
            for l in line: print("    " + l)
