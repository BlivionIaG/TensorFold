#!/usr/bin/env python3
"""Cross-check truth.py against the transformers modeling code on CPU (architecture oracle, not an engine).

  hf_check.py raw <hf_model_dir> <ids.npy> [--dtype f64]   # real raw checkpoint, e.g. Qwen3.5-0.8B
  hf_check.py tiny-moe <workdir>                            # random tiny Qwen3.5-MoE exported in MLX layout
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.modules.update(fla=None, causal_conv1d=None)  # force transformers' pure torch fallbacks (no GPU kernels)
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).parent))
import truth


def run_truth(path, ids, dt):
    st = truth.Store(Path(path), dt)
    out = np.empty((len(ids), st.cfg["vocab_size"]), np.float32)
    with torch.inference_mode():
        truth.forward(st, ids, out, log=open("/dev/null", "w"))
    return out


def compare(a, b):
    d = np.abs(a - b)
    print(f"max|dlogit|={d.max():.3e} mean={d.mean():.3e} top1 agree={(a.argmax(-1) == b.argmax(-1)).mean() * 100:.1f}%")


def tiny_moe(work: Path):
    from transformers import Qwen3_5MoeConfig, Qwen3_5MoeForCausalLM
    torch.manual_seed(0)
    text = dict(hidden_size=128, head_dim=32, num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=4,
                full_attention_interval=4, linear_conv_kernel_dim=4, linear_key_head_dim=16, linear_num_key_heads=2,
                linear_num_value_heads=4, linear_value_head_dim=16, moe_intermediate_size=48, shared_expert_intermediate_size=48,
                num_experts=8, num_experts_per_tok=2, vocab_size=300, rms_norm_eps=1e-6, hidden_act="silu",
                layer_types=["linear_attention"] * 3 + ["full_attention"], tie_word_embeddings=False,
                rope_parameters=dict(rope_type="default", rope_theta=10000.0, partial_rotary_factor=0.25,
                                     mrope_section=[1, 1, 2], mrope_interleaved=True))
    cfg = Qwen3_5MoeConfig(text_config=text).text_config
    cfg._experts_implementation = "eager"  # grouped_mm has no float64
    m = Qwen3_5MoeForCausalLM(cfg).double().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n or "A_log" in n or "dt_bias" in n:
                p.copy_(torch.randn_like(p) * 0.3 + (0.0 if "dt_bias" not in n else 0.5))
            elif p.dim() >= 2:
                p.copy_(torch.randn_like(p) * 0.15)
    sd, out = m.state_dict(), {}
    for n, p in sd.items():
        n = n.removeprefix("model.")
        p = p.contiguous()
        if n == "lm_head.weight":
            out["language_model.lm_head.weight"] = p
            continue
        n = "language_model.model." + n.removeprefix("language_model.")
        if n.endswith("experts.gate_up_proj"):
            b = n.removesuffix("experts.gate_up_proj") + "switch_mlp."
            g, u = p.chunk(2, 1)
            out[b + "gate_proj.weight"], out[b + "up_proj.weight"] = g.contiguous(), u.contiguous()
        elif n.endswith("experts.down_proj"):
            out[n.removesuffix("experts.down_proj") + "switch_mlp.down_proj.weight"] = p
        elif n.endswith("conv1d.weight"):
            out[n] = p.transpose(1, 2).contiguous()  # MLX layout (C, K, 1)
        elif n.endswith("norm.weight") and "linear_attn" not in n or n.endswith(("layernorm.weight", "q_norm.weight", "k_norm.weight")):
            out[n] = p + 1.0  # MLX bakes the (1 + w) in
        else:
            out[n] = p
    work.mkdir(parents=True, exist_ok=True)
    save_file(out, work / "model.safetensors")
    (work / "config.json").write_text(json.dumps({"text_config": text}))
    ids = np.random.RandomState(1).randint(0, 300, 97)
    with torch.no_grad():
        ref = m(torch.from_numpy(ids)[None]).logits[0].float().numpy()
    print("tiny MoE, truth f64 vs transformers f64:")
    compare(ref, run_truth(work, ids, torch.float64))


def raw(model, ids_path, dtype):
    from transformers import AutoModelForCausalLM
    ids = np.load(ids_path).reshape(-1)
    dt = torch.float64 if dtype == "f64" else torch.float32
    m = AutoModelForCausalLM.from_pretrained(model, dtype=dt).eval()
    with torch.no_grad():
        ref = m(torch.from_numpy(ids)[None]).logits[0].float().numpy()
    print(f"{model} {dtype}, truth vs transformers:")
    compare(ref, run_truth(model, ids, dt))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("raw", "tiny-moe")), ap.add_argument("path"), ap.add_argument("ids", nargs="?")
    ap.add_argument("--dtype", default="f64")
    a = ap.parse_args()
    tiny_moe(Path(a.path)) if a.mode == "tiny-moe" else raw(a.path, a.ids, a.dtype)
