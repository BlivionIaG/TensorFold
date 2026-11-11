"""Capture the real router decisions of Qwen3.6-35B-A3B (MLX affine 4-bit) with mlx_lm, one npz a prompt. No eGPU.

run: /tmp/mlxenv/bin/python routing_trace.py [MODEL_DIR] [OUT_DIR] [MAX_TOKENS]
     (mlx 0.32.4 / mlx_lm 0.32.2; `uv venv /tmp/mlxenv && uv pip install mlx mlx_lm`)
Checkpoint: mlx-community/Qwen3.6-35B-A3B-4bit @38740b847e4cb78f352aba30aa41c76e08e6eb46, not requantized or converted.

What is recorded, per prompt, for every token the model processes (prefill tokens and decode tokens) and every one of the
40 layers: the router's top-8 expert ids and the renormalized weights (the same arithmetic as `Qwen3NextSparseMoeBlock`:
fp32-precise softmax over 256, argpartition top-8, divide by the sum of the eight). Decode runs the real sampler
(temperature 0.7, top-k 20, top-p 0.95; greedy for the code and math prompts) through the real chat template, so the trace is
what a user's tokens would route to, including the model's own thinking text for half of the prompts.

Lookahead candidates ("preds"): the routing a prefetcher could guess before a layer's own router has run, by applying a
later layer's post_attention_layernorm + router to an earlier hidden state. Per layer t (the target), five guesses, uint8 ids
(255 where the source layer does not exist):
  0 = t's router on the residual entering layer t   (before t's mixer; available the moment layer t-1 finished)
  1 = t's router on the residual entering layer t-1 (one whole layer ahead)
  2 = t's router on the residual entering layer t-2
  3 = t's router on t-1's residual after its mixer  (before t-1's MoE)
  4 = t's router on t-2's residual after its mixer
Prefill tokens carry no guesses (255); only decode tokens do.
"""
import json
import os
import sys
import time

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models import qwen3_5
from mlx_lm.sample_utils import make_sampler

MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/models/Qwen3.6-35B-A3B-4bit")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces")
MAX_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 288
TOPK, NL = 8, 40
HERE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(HERE, "..", "metal-tb4-split-moe.md")

PROMPTS = {
    "chat": [
        "I just moved to a new city and I don't know anyone. How do I make friends as an adult? Be practical.",
        "My sourdough starter smells like nail polish remover. Is it dead? What do I do?",
        "Can you explain the difference between a Roth IRA and a traditional IRA like I'm 25 and just got my first job?",
        "What should I pack for a five day trip to Lisbon in March? I like walking and food, I dislike crowds.",
        "Give me a polite but firm message to send my landlord who hasn't fixed the heating for two weeks.",
        "Why is the sky blue but sunsets are red? My kid asked and I gave up halfway through.",
        "What are good questions to ask at the end of a job interview for a data engineering role?",
        "Recommend a weekly workout plan for someone with one 20 minute slot a day and no equipment.",
    ],
    "code": [
        "Write a Python function that merges overlapping intervals, with type hints and a few pytest tests.",
        "Implement an LRU cache in Rust with O(1) get and put, without unsafe code. Explain the design briefly.",
        "Here is a bug: `for i in range(len(xs)): if xs[i] % 2 == 0: xs.remove(xs[i])` raises IndexError sometimes. Explain and fix it in Python.",
        "Write a SQL query that returns, for each customer, their second most recent order and the days between it and the most recent one. Postgres.",
        "Write a TypeScript debounce function with proper generics and a cancel method, and show a usage example.",
        "Explain what this bash does and rewrite it safely: `for f in $(ls *.log); do rm $f; done`",
        "Write a C function that reverses a singly linked list in place, then one that detects a cycle with Floyd's algorithm.",
        "Implement binary search in Go for the first index where a predicate is true, and explain the off-by-one pitfalls.",
    ],
    "math": [
        "Prove that the square root of 2 is irrational, then prove that the sum of two irrationals can be rational.",
        "Compute the integral of x^2 * e^(-x) from 0 to infinity, showing each step of integration by parts.",
        "A bag has 5 red and 7 blue marbles. Two are drawn without replacement. What is the probability they are different colours? Show the work.",
        "Solve the system: 3x + 2y - z = 7, x - y + 2z = 3, 2x + 3y + z = 12. Use elimination and verify.",
        "How many positive integers less than 1000 are divisible by 3 or 5 but not by 15? Explain the counting.",
        "Find the eigenvalues and eigenvectors of the matrix [[4, 1], [2, 3]] and diagonalize it.",
        "Show that the sum of the first n odd numbers is n^2 by induction, then give a geometric picture.",
        "A train leaves at 9:00 at 80 km/h and another at 9:30 at 100 km/h on the same track in the same direction. When does the second catch up and how far have they gone?",
    ],
    "long": [
        "Write a 600 word short story about a lighthouse keeper who discovers the light has been signalling someone for decades.",
        "Write a detailed essay on how the printing press changed European politics and religion between 1450 and 1650.",
        "Explain how a modern CPU executes an instruction, from fetch to retirement, including out-of-order execution and branch prediction.",
        "Write a thorough guide to starting a vegetable garden on a small balcony, with a month by month calendar.",
        "Describe the causes, course and consequences of the 2008 financial crisis for an undergraduate economics class.",
        "Write a letter from an astronaut on a long mission to their younger self, reflecting on what mattered and what did not.",
        "Explain how mixture-of-experts language models route tokens, why they are cheaper to run than dense models, and what goes wrong in training.",
        "Write a product requirements document for a note-taking app aimed at field biologists with intermittent connectivity.",
    ],
}
GREEDY = {"code", "math"}


def build_jobs():
    jobs = []
    for cat, ps in PROMPTS.items():
        for i, p in enumerate(ps):
            # half of each category with the model's thinking, half answering directly
            jobs.append(dict(name=f"{cat}{i}", cat=cat, text=p, think=(i % 2 == 0), max_tokens=MAX_TOKENS))
    with open(DOC) as f:
        doc = f.read()
    for i, (lo, hi) in enumerate([(0, 6500), (9000, 15500)]):
        jobs.append(dict(name=f"doc{i}", cat="doc", text="Summarize the following design note in six bullet points, "
                         "then list the three biggest risks it names.\n\n" + doc[lo:hi], think=False, max_tokens=MAX_TOKENS))
    return jobs


def softmax_top(gates, k=TOPK):
    probs = mx.softmax(gates, axis=-1, precise=True)
    ids = mx.argpartition(probs, kth=-k, axis=-1)[..., -k:]
    w = mx.take_along_axis(probs, ids, axis=-1)
    return ids, w / w.sum(axis=-1, keepdims=True)


class Recorder:
    def __init__(self, layers):
        self.layers = layers
        self.on = False
        self.want_preds = False
        self.ids = [None] * NL
        self.w = [None] * NL
        self.xin = [None] * NL
        self.hmid = [None] * NL

    def router_ids(self, t, src):
        lay = self.layers[t]
        g = lay.mlp.gate(lay.post_attention_layernorm(src))
        return softmax_top(g)[0]

    def collect(self):
        """ids (S,L,8), w (S,L,8), preds (S,L,5,8) as mx arrays (preds None unless want_preds)."""
        ids = mx.stack(self.ids, axis=2)[0]
        w = mx.stack(self.w, axis=2)[0]
        preds = None
        if self.want_preds:
            S = ids.shape[0]
            none = mx.full((S, TOPK), 255, dtype=mx.uint32)
            per_layer = []
            for t in range(NL):
                cols = [self.router_ids(t, self.xin[t])[0].astype(mx.uint32)]
                for d in (1, 2):
                    cols.append(self.router_ids(t, self.xin[t - d])[0].astype(mx.uint32) if t - d >= 0 else none)
                for d in (1, 2):
                    cols.append(self.router_ids(t, self.hmid[t - d])[0].astype(mx.uint32) if t - d >= 0 else none)
                per_layer.append(mx.stack(cols, axis=1))
            preds = mx.stack(per_layer, axis=1)
        return ids.astype(mx.uint32), w.astype(mx.float16), preds


def patch(model):
    layers = model.model.layers
    rec = Recorder(layers)
    for i, lay in enumerate(layers):
        lay._rec_idx = i

    def call(self, x, mask=None, cache=None):
        i = self._rec_idx
        r = (self.linear_attn if self.is_linear else self.self_attn)(self.input_layernorm(x), mask, cache)
        h = x + r
        hn = self.post_attention_layernorm(h)
        if rec.on:
            ids, w = softmax_top(self.mlp.gate(hn))
            rec.ids[i], rec.w[i] = ids, w
            if rec.want_preds:
                rec.xin[i], rec.hmid[i] = x, h
        return h + self.mlp(hn)

    qwen3_5.DecoderLayer.__call__ = call
    return rec


def run_job(model, tok, rec, job):
    msgs = [{"role": "user", "content": job["text"]}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=job["think"])
    ptoks = tok.encode(prompt)
    sampler = make_sampler(temp=0.0) if job["cat"] in GREEDY else make_sampler(temp=0.7, top_p=0.95, top_k=20)
    cache = model.make_cache()
    rows_ids, rows_w, rows_pred, kinds, gen = [], [], [], [], []

    # prefill in one call (prompts are at most about 2.5k tokens); the last position's logits give the first token
    rec.on, rec.want_preds = True, False
    t0 = time.perf_counter()
    logits = model(mx.array(ptoks)[None], cache=cache)[:, -1, :]
    nxt = sampler(logits - mx.logsumexp(logits, axis=-1, keepdims=True))
    ids, w, _ = rec.collect()
    mx.eval(nxt, ids, w)
    t_pre = time.perf_counter() - t0
    rows_ids.append(np.array(ids, dtype=np.uint8)); rows_w.append(np.array(w))
    rows_pred.append(np.full((len(ptoks), NL, 5, TOPK), 255, dtype=np.uint8)); kinds += [0] * len(ptoks)
    gen.append(int(nxt.item()))

    rec.want_preds = True
    eos = set(tok.eos_token_ids) if hasattr(tok, "eos_token_ids") else {tok.eos_token_id}
    t0 = time.perf_counter()
    while len(gen) < job["max_tokens"] and gen[-1] not in eos:
        logits = model(mx.array([[gen[-1]]]), cache=cache)[:, -1, :]
        nxt = sampler(logits - mx.logsumexp(logits, axis=-1, keepdims=True))
        ids, w, preds = rec.collect()
        mx.eval(nxt, ids, w, preds)
        # this step routed gen[-1]'s position; row order is the processing order of tokens
        rows_ids.append(np.array(ids, dtype=np.uint8)); rows_w.append(np.array(w)); rows_pred.append(np.array(preds, dtype=np.uint8))
        kinds.append(1)
        gen.append(int(nxt.item()))
    t_dec = time.perf_counter() - t0
    rec.on = False
    ids = np.concatenate(rows_ids); w = np.concatenate(rows_w); preds = np.concatenate(rows_pred)
    kinds = np.array(kinds, dtype=np.uint8)
    # the last generated token was sampled but never processed: no routing for it
    text = tok.decode(gen)
    return dict(ids=ids, w=w, preds=preds, kind=kinds, prompt_tokens=np.array(ptoks, dtype=np.int32),
                gen_tokens=np.array(gen, dtype=np.int32)), text, t_pre, t_dec


def main():
    os.makedirs(OUT, exist_ok=True)
    model, tok = load(MODEL)
    rec = patch(model)
    jobs = build_jobs()
    meta = []
    only = os.environ.get("ONLY")
    for job in jobs:
        if only and job["name"] not in only.split(","):
            continue
        d, text, t_pre, t_dec = run_job(model, tok, rec, job)
        np.savez_compressed(os.path.join(OUT, job["name"] + ".npz"), **d)
        n_dec = int((d["kind"] == 1).sum())
        meta.append(dict(name=job["name"], cat=job["cat"], think=job["think"], prompt_tokens=int(len(d["prompt_tokens"])),
                         decode_tokens=n_dec, prefill_s=round(t_pre, 2), decode_s=round(t_dec, 2), sample=text[:160]))
        print(f"{job['name']:8s} prompt {len(d['prompt_tokens']):5d}  decode {n_dec:4d}  prefill {t_pre:5.1f}s  "
              f"decode {t_dec:5.1f}s ({n_dec / max(t_dec, 1e-9):.0f} tok/s with capture)", flush=True)
        with open(os.path.join(OUT, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
