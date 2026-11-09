#!/usr/bin/env python3
"""The mlx_lm references the gfx900 gate compares with: greedy tokens of two chat runs and float32 logits of a 87-token sequence.

    uvx --from mlx-lm python tools/vega-macos/mlx_ref.py [MODEL_DIR] [OUT_DIR]

OUT_DIR (default ~/.cache/tensorfold-vega/ref) gets ref.json (prompts, token ids, text, TensorFold's token sha), ids.npy (int32 ids, the
thinking prompt and its first 64 greedy tokens) and mlx_logits.npy (float32 [87, vocab], one forward over those ids).
"""
import hashlib, json, os, sys
import numpy as np
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

model_dir = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/models/Qwen3.5-4B-MLX-4bit")
out_dir = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "~/.cache/tensorfold-vega/ref")
os.makedirs(out_dir, exist_ok=True)
m, t = load(model_dir)
Q = "What is the capital of France? Answer in one short sentence."
out = {}
for name, think, n in (("nothink", False, 24), ("think96", True, 96)):
    p = t.apply_chat_template([{"role": "user", "content": Q}], add_generation_prompt=True, enable_thinking=think, tokenize=False)
    ids = t.encode(p)
    toks = []
    for r in stream_generate(m, t, ids, max_tokens=n, sampler=make_sampler(temp=0.0)):
        toks.append(int(r.token))
        if r.token in (248046, 248044):
            break
    sha = hashlib.sha256(",".join(map(str, toks)).encode()).hexdigest()[:12]
    out[name] = {"prompt": p, "prompt_ids": ids, "tokens": toks, "text": t.decode(toks), "sha": sha}
    print(name, len(toks), sha, toks, flush=True)
json.dump(out, open(os.path.join(out_dir, "ref.json"), "w"))
ids = out["think96"]["prompt_ids"] + out["think96"]["tokens"][:64]
np.save(os.path.join(out_dir, "ids.npy"), np.array(ids, dtype=np.int32))
lg = m(mx.array(ids)[None]).astype(mx.float32)
np.save(os.path.join(out_dir, "mlx_logits.npy"), np.array(lg[0]))
print("ids", len(ids), "logits", lg.shape)
