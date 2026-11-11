#!/usr/bin/env python3
"""mlx_lm greedy references for the MoE offload gate: several chat prompts, up to N tokens each, token ids and TensorFold's sha of them.

    /tmp/mlxenv/bin/python tools/vega-macos/moe_ref.py [MODEL_DIR] [OUT.json] [N]

(mlx 0.32.4, mlx_lm 0.32.2: `uv venv /tmp/mlxenv && uv pip install --python /tmp/mlxenv/bin/python mlx-lm`). `zig build tf-moe-gate` (tools/vega-macos/moe_gate.zig) compares a running server with OUT.json.
"""
import hashlib, json, os, sys
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

model_dir = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/models/Qwen3.6-35B-A3B-4bit")
out_path = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "~/.cache/tensorfold-vega/ref-moe.json")
n = int(sys.argv[3]) if len(sys.argv) > 3 else 96
PROMPTS = [
    "What is the capital of France? Answer in one short sentence.",
    "Write a Python function that returns the n-th Fibonacci number, with a short explanation.",
    "Explain in three sentences why the sky is blue.",
    "Translate to German: 'The harbour town kept its ledgers in a stone tower.' Then list five German nouns from the sentence.",
    "List the planets of the solar system in order from the sun, one per line, with one fact each.",
    "Summarize the plot of Romeo and Juliet in about 80 words.",
    # two long prompts: one prompt span of ~460 rows, and one of ~1100 rows (a second span past the 1024-row chunk), 40 tokens each
    "The harbour town of Merrowick kept its ledgers in a stone tower. " * 28 + "Summarize this in two sentences.",
    "The harbour town of Merrowick kept its ledgers in a stone tower. " * 66 + "Summarize this in two sentences.",
]
m, t = load(model_dir)
out = []
for q in PROMPTS:
    p = t.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, enable_thinking=False, tokenize=False)
    ids = t.encode(p)
    toks = []
    limit = 40 if len(q) > 400 else n
    for r in stream_generate(m, t, ids, max_tokens=limit, sampler=make_sampler(temp=0.0)):
        toks.append(int(r.token))
        if r.token in (248046, 248044):
            break
    sha = hashlib.sha256(",".join(map(str, toks)).encode()).hexdigest()[:12]
    out.append({"question": q, "prompt_ids": ids, "tokens": toks, "text": t.decode(toks), "sha": sha})
    print(len(toks), sha, repr(t.decode(toks)[:80]), flush=True)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
json.dump(out, open(out_path, "w"))
