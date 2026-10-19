"""CPU fp32 reference for Nemotron-H: the MLX 4-bit checkpoint dequantized exactly into the HF model, teacher-forced.

usage: nem_ref.py tokenize <model dir> <out dir>            writes <out dir>/<name>.ids.bin (u32) per prompt
       nem_ref.py logits <model dir> <ids.bin> <out.npy>    fp32 logits [T, V] of the HF model on CPU, with stats
"""
import json, sys, time, os
import numpy as np

MODEL_DIR = None

PROMPTS = {
    "code": "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "chat": [{"role": "user", "content": "Explain in two sentences why the sky is blue."}],
    "story": "Once upon a time, in a small village by the sea, there lived",
}


def tokenize(model_dir, out_dir):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    os.makedirs(out_dir, exist_ok=True)
    for name, p in PROMPTS.items():
        if isinstance(p, list):
            ids = tok.apply_chat_template(p, add_generation_prompt=True, tokenize=True)
            if not isinstance(ids, list):
                ids = ids["input_ids"]
        else:
            ids = tok(p)["input_ids"]
        np.asarray(ids, dtype=np.uint32).tofile(os.path.join(out_dir, f"{name}.ids.bin"))
        print(name, len(ids), ids[:12])


def dequant(w, s, b, bits=4, group=64):
    """MLX affine: w uint32 [n, k/8] nibbles little-end first; out[n, k] = s[n, k//g] * q + b[n, k//g] in fp32."""
    import torch
    n, kw = w.shape
    per = 32 // bits
    shifts = torch.arange(per, dtype=torch.int32) * bits
    q = ((w.view(torch.int32).unsqueeze(-1) >> shifts) & ((1 << bits) - 1)).reshape(n, kw * per).to(torch.float32)
    k = kw * per
    s32 = s.to(torch.float32).repeat_interleave(group, dim=-1)[:, :k]
    b32 = b.to(torch.float32).repeat_interleave(group, dim=-1)[:, :k]
    return q * s32 + b32


def load_state(model_dir):
    import torch
    from safetensors.torch import load_file
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    files = sorted(set(idx.values()))
    raw = {}
    for f in files:
        raw.update(load_file(os.path.join(model_dir, f)))
    out = {}
    done = set()
    for k in list(raw):
        if k in done:
            continue
        if k.endswith(".scales") or k.endswith(".biases"):
            continue
        if k.endswith(".weight") and k[:-7] + ".scales" in raw:
            stem = k[:-7]
            w, s, b = raw[k], raw[stem + ".scales"], raw[stem + ".biases"]
            if w.dim() == 3:
                full = torch.stack([dequant(w[e], s[e], b[e]) for e in range(w.shape[0])])
            else:
                full = dequant(w, s, b)
            done.update({k, stem + ".scales", stem + ".biases"})
            out[stem + ".weight"] = full
        else:
            v = raw[k].to(torch.float32)
            # MLX keeps the depthwise conv taps as (C, K, 1); the HF Conv1d wants (C, 1, K)
            out[k] = v.permute(0, 2, 1).contiguous() if k.endswith("conv1d.weight") else v
    # MLX names -> HF names
    hf = {}
    for k, v in out.items():
        kk = k.replace("backbone.", "model.", 1).replace(".switch_mlp.fc1.weight", ".experts.up_proj").replace(".switch_mlp.fc2.weight", ".experts.down_proj")
        hf[kk] = v
    return hf


def logits(model_dir, ids_path, out_path):
    import torch
    from transformers import AutoConfig
    from transformers.models.nemotron_h import modeling_nemotron_h as mh
    torch.set_num_threads(min(64, os.cpu_count()))
    cfg = AutoConfig.from_pretrained(model_dir)
    cfg.use_cache = False
    t0 = time.time()
    state = load_state(model_dir)
    print(f"dequantized {len(state)} tensors in {time.time()-t0:.0f}s", flush=True)
    with torch.device("meta"):
        model = mh.NemotronHForCausalLM(cfg)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    print("missing", [m for m in missing][:20], "unexpected", unexpected[:20])
    model = model.float().eval()
    # upstream TensorFold's dt clamp rule: time_step_limit from the config, else (0, inf); HF uses (time_step_min, inf)
    lim = tuple(cfg.time_step_limit) if getattr(cfg, "time_step_limit", None) else (0.0, float("inf"))
    stats = {"fc1_max": 0.0, "shared_up_max": 0.0, "dt_below_1e-3": 0, "dt_total": 0, "fc1_over_255.9": 0, "fc1_count": 0}
    per_layer = {}
    for name, mod in model.named_modules():
        if isinstance(mod, mh.NemotronHMamba2Mixer):
            mod.time_step_limit = lim
            def dt_hook(m, inp, out, mod=mod):
                dt = torch.nn.functional.softplus(out[..., -mod.num_heads:] + mod.dt_bias)
                stats["dt_below_1e-3"] += int((dt < 1e-3).sum())
                stats["dt_total"] += dt.numel()
            mod.in_proj.register_forward_hook(dt_hook)
        if isinstance(mod, mh.NemotronHExperts):
            def experts_forward(hidden_states, top_k_index, top_k_weights, self=mod, name=name):
                final = torch.zeros_like(hidden_states)
                mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
                for e in mask.sum(dim=(-1, -2)).nonzero().squeeze(-1).tolist():
                    pos, tok = torch.where(mask[e])
                    up = torch.nn.functional.linear(hidden_states[tok], self.up_proj[e])
                    m = float(up.abs().max())
                    stats["fc1_max"] = max(stats["fc1_max"], m)
                    per_layer[name] = max(per_layer.get(name, 0.0), m)
                    stats["fc1_over_255.9"] += int((up.abs() > 255.9).sum())
                    stats["fc1_count"] += up.numel()
                    h = self.act_fn(up)
                    h = torch.nn.functional.linear(h, self.down_proj[e]) * top_k_weights[tok, pos, None]
                    final.index_add_(0, tok, h)
                return final
            mod.forward = experts_forward
        if isinstance(mod, mh.NemotronHMLP) and "shared" in name:
            def up_hook(m, inp, out):
                stats["shared_up_max"] = max(stats["shared_up_max"], float(out.abs().max()))
            mod.up_proj.register_forward_hook(up_hook)
    ids = np.fromfile(ids_path, dtype=np.uint32).astype(np.int64)
    print("tokens", len(ids), "dt limit", lim, flush=True)
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=torch.tensor(ids)[None], use_cache=False).logits[0].float().numpy()
    print(f"forward {time.time()-t0:.0f}s, logits {out.shape}, finite {np.isfinite(out).all()}", flush=True)
    np.save(out_path, out)
    print("stats", json.dumps(stats))
    worst = sorted(per_layer.items(), key=lambda kv: -kv[1])[:5]
    print("fc1 |up| max by layer (top 5):", worst)
    print("greedy", out.argmax(-1).tolist())


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "tokenize":
        tokenize(sys.argv[2], sys.argv[3])
    elif cmd == "logits":
        logits(sys.argv[2], sys.argv[3], sys.argv[4])
