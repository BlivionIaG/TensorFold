"""Mac-alone batched decode and prefill with mlx_lm, the baseline for the split (docs/proposals/metal-tb4-split-moe.md).

run: /tmp/mlxenv/bin/python mlx_batched_decode.py MODEL_DIR [N,N,...] [PROMPT_REPEAT] [TOKENS]
     (mlx 0.32.4, mlx_lm 0.32.2; `uv venv /tmp/mlxenv && uv pip install mlx-lm`)
N sequences go through mlx_lm's batch_generate with greedy sampling: prefill, then TOKENS decode steps with all N rows in one
round. The sequences are DIFFERENT prompts of about nine tokens (each repeated PROMPT_REPEAT times; set SAME=1 for N copies of
the card runs' prompt): a mixture-of-experts model reads the union of the rows' experts, so identical prompts, which pick
the same experts every step, would flatter batching. Prints the library's own prompt and generation rates and the wall time
of the whole call, best of two passes after a warm-up.
"""
import os
import sys
import time

import mlx.core as mx
from mlx_lm import batch_generate, load

path = sys.argv[1]
counts = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1,2,4").split(",")]
repeat = int(sys.argv[3]) if len(sys.argv) > 3 else 1
tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 40
model, tok = load(path)
TEXTS = [
    "The three primary colors are red, yellow and",
    "In 1969 the first humans walked on the surface of",
    "A binary search tree keeps its keys in sorted order so that",
    "The capital of France is Paris and the capital of",
    "To make a good loaf of bread you need flour, water,",
    "Quantum computers use qubits, which unlike classical bits can",
    "The quick brown fox jumps over the lazy dog while the",
    "Once upon a time in a small village by the sea there",
]
same = os.environ.get("SAME") == "1"
prompts = [tok.encode(TEXTS[0 if same else i % len(TEXTS)]) * repeat for i in range(max(counts))]
ids = prompts[0]
print(f"{path.rstrip('/').split('/')[-1]}: prompts of {len(ids)} tokens ({'the same' if same else 'different'}), {tokens} generated, mlx {mx.__version__}")


def run(n):
    t0 = time.perf_counter()
    r = batch_generate(model, tok, prompts[:n], max_tokens=tokens, return_token_ids=True)
    return r, time.perf_counter() - t0


run(1)  # warm the kernels
for n in counts:
    best = None
    for _ in range(2):
        r, wall = run(n)
        s = r.stats
        if best is None or s.generation_tps > best[0].generation_tps:
            best = (s, wall, r)
    s, wall, r = best
    distinct = len({tuple(t) for t in r.token_ids})
    print(f"mlx_lm {n} sequences: prefill {s.prompt_tokens} tokens at {s.prompt_tps:.1f} tok/s; decode {s.generation_tokens} tokens "
          f"= {s.generation_tps:.1f} tok/s ({s.generation_tps / n:.1f} a stream); wall {wall * 1e3:.0f} ms; "
          f"{distinct} distinct continuations; peak {s.peak_memory:.1f} GB")
