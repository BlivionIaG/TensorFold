#!/usr/bin/env python3
"""cosine.py OURS.npy MLX.npy: per-row cosine, argmax agreement and mean |diff| of two [rows, vocab] logits files (uv run --with numpy)."""
import sys
import numpy as np
a = np.load(sys.argv[1]).astype(np.float64)
b = np.load(sys.argv[2]).astype(np.float64)
n = min(a.shape[0], b.shape[0])
v = min(a.shape[1], b.shape[1])
a, b = a[:n, :v], b[:n, :v]
cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
print("rows %d  cosine min %.6f mean %.6f  argmax equal %d/%d  mean|diff| %.4f" % (n, cos.min(), cos.mean(), int((a.argmax(1) == b.argmax(1)).sum()), n, np.abs(a - b).mean()))
sys.exit(0 if cos.min() > 0.999 and (a.argmax(1) == b.argmax(1)).all() else 1)
