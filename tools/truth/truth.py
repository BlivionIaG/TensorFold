#!/usr/bin/env python3
"""Independent CPU truth forward for Qwen3.5 / Qwen3.6 (dense and MoE), MLX affine or raw HF weights.

Plain torch, no engine code. Weights are dequantized one tensor at a time, the recurrence is the
sequential gated delta rule (not the chunked form), everything runs in float64 unless --dtype f32.
Usage: truth.py <model_dir> <ids.npy> <out.npy> [--dtype f64|f32]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


class Store:
    """Tensor access by module name; dequantizes `<name>.weight/.scales/.biases` on demand."""

    def __init__(self, root: Path, dt: torch.dtype):
        self.root, self.dt = root, dt
        cfg = json.loads((root / "config.json").read_text())
        self.cfg = cfg.get("text_config", cfg)
        self.group = int(cfg.get("quantization", {}).get("group_size", 64))
        index = root / "model.safetensors.index.json"
        files = sorted(root.glob("*.safetensors"))
        if index.exists():
            self.where = json.loads(index.read_text())["weight_map"]
        else:
            self.where = {k: f.name for f in files for k in safe_open(f, "pt").keys()}
        self.files: dict[str, object] = {}
        self.cache: dict[str, torch.Tensor] = {}  # tensors read this layer (stacked experts are read once)
        mlx = "language_model.model.embed_tokens.weight" in self.where
        self.pre = "language_model.model." if mlx else "model.language_model."
        self.head = "language_model.lm_head" if mlx else "lm_head"
        self.shift = 0.0 if mlx else 1.0  # MLX checkpoints bake the (1 + w) of the RMSNorms in

    def has(self, name: str) -> bool:
        return name in self.where

    def raw(self, name: str) -> torch.Tensor:
        if name not in self.cache:
            f = self.files.get(self.where[name])
            if f is None:
                f = self.files[self.where[name]] = safe_open(self.root / self.where[name], "pt")
            self.cache[name] = f.get_tensor(name)
        return self.cache[name]

    def dense(self, name: str, rows=None) -> torch.Tensor:
        """Weight `name` (optionally first-dim rows) as a dequantized tensor of the working dtype."""
        w = self.raw(name + ".weight")
        if not self.has(name + ".scales"):
            return (w if rows is None else w[rows]).to(self.dt)
        s, b = self.raw(name + ".scales"), self.raw(name + ".biases")
        if rows is not None:
            w, s, b = w[rows], s[rows], b[rows]
        return self.dequant(w, s, b)

    def dequant(self, w, s, b) -> torch.Tensor:
        n_in = s.shape[-1] * self.group
        bits = w.shape[-1] * 32 // n_in
        by = w.contiguous().view(torch.uint8)  # little-endian words: value 0 sits in the low bits
        if bits == 8:
            q = by
        elif bits == 4:
            q = torch.stack((by & 15, by >> 4), -1).flatten(-2)
        else:
            q = unpack(w, bits, n_in)
        q = q.reshape(*w.shape[:-1], s.shape[-1], self.group).to(self.dt)
        out = q * s.to(self.dt)[..., None] + b.to(self.dt)[..., None]
        return out.reshape(*w.shape[:-1], n_in)

    def norm(self, name: str) -> torch.Tensor:  # RMSNorm scale, already as the multiplier
        return self.raw(name).to(self.dt) + self.shift


def unpack(w: torch.Tensor, bits: int, n: int) -> torch.Tensor:
    """Values of `bits` bits packed little-endian across uint32 words (MLX's layout for 2, 3, 5 and 6 bits)."""
    words = w.to(torch.int64) & 0xFFFFFFFF
    at = torch.arange(n) * bits
    word, shift = at // 32, at % 32
    low = words[..., word] >> shift
    high = words[..., (word + 1).clamp(max=words.shape[-1] - 1)] << (32 - shift)
    return (low | high) & ((1 << bits) - 1)


def rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope_tables(positions, rot, theta, dt):
    inv = theta ** (-torch.arange(0, rot, 2, dtype=torch.float64) / rot)  # f64 trig for any dtype
    ang = positions.double()[:, None] * inv[None]
    emb = torch.cat((ang, ang), -1)
    return emb.cos().to(dt), emb.sin().to(dt)


def apply_rope(x, cos, sin):  # x (heads, T, hd): rotate the first `rot` dims, rotate_half layout
    rot = cos.shape[-1]
    a, b = x[..., :rot], x[..., rot:]
    half = torch.cat((-a[..., rot // 2:], a[..., :rot // 2]), -1)
    return torch.cat((a * cos + half * sin, b), -1)


def full_attention(st: Store, base: str, x, cos, sin, c):
    T = x.shape[0]
    nh, nkv, hd, eps = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["rms_norm_eps"]
    qg = F.linear(x, st.dense(base + "q_proj")).view(T, nh, 2 * hd)
    q, gate = qg[..., :hd], qg[..., hd:].reshape(T, nh * hd)  # per-head [q | gate]
    q = rms(q, st.norm(base + "q_norm.weight"), eps).transpose(0, 1)
    k = rms(F.linear(x, st.dense(base + "k_proj")).view(T, nkv, hd), st.norm(base + "k_norm.weight"), eps)
    v = F.linear(x, st.dense(base + "v_proj")).view(T, nkv, hd).transpose(0, 1)
    q, k = apply_rope(q, cos, sin), apply_rope(k.transpose(0, 1), cos, sin)
    k, v = k.repeat_interleave(nh // nkv, 0), v.repeat_interleave(nh // nkv, 0)
    out = torch.empty(nh, T, hd, dtype=x.dtype)
    for a in range(0, T, 512):  # causal attention in query blocks
        b = min(a + 512, T)
        s = q[:, a:b] @ k[:, :b].transpose(1, 2) * hd ** -0.5
        mask = torch.arange(b)[None] > torch.arange(a, b)[:, None]
        out[:, a:b] = torch.softmax(s.masked_fill(mask, float("-inf")), -1) @ v[:, :b]
    out = out.transpose(0, 1).reshape(T, nh * hd) * torch.sigmoid(gate)
    return F.linear(out, st.dense(base + "o_proj"))


def linear_attention(st: Store, base: str, x, c):
    T = x.shape[0]
    hk, hv, dk, dv = c["linear_num_key_heads"], c["linear_num_value_heads"], c["linear_key_head_dim"], c["linear_value_head_dim"]
    eps, K = c["rms_norm_eps"], c["linear_conv_kernel_dim"]
    mixed = F.linear(x, st.dense(base + "in_proj_qkv"))
    z = F.linear(x, st.dense(base + "in_proj_z")).view(T, hv, dv)
    beta = torch.sigmoid(F.linear(x, st.dense(base + "in_proj_b")))
    a = F.linear(x, st.dense(base + "in_proj_a"))
    cw = st.raw(base + "conv1d.weight").to(x.dtype)
    cw = cw[:, :, 0] if cw.shape[-1] == 1 else cw[:, 0, :]  # (channels, K), tap 0 = oldest
    pad = F.pad(mixed.t(), (K - 1, 0))  # (channels, T + K - 1), causal
    mixed = F.silu(sum(cw[:, j:j + 1] * pad[:, j:j + T] for j in range(K)).t())
    kd = hk * dk
    q, k, v = mixed[:, :kd].view(T, hk, dk), mixed[:, kd:2 * kd].view(T, hk, dk), mixed[:, 2 * kd:].view(T, hv, dv)
    q, k = q * torch.rsqrt(q.pow(2).sum(-1, keepdim=True) + 1e-6), k * torch.rsqrt(k.pow(2).sum(-1, keepdim=True) + 1e-6)
    q = q.repeat_interleave(hv // hk, 1) * dk ** -0.5  # each key head serves hv/hk consecutive value heads
    k = k.repeat_interleave(hv // hk, 1)
    A = st.raw(base + "A_log").to(x.dtype).exp()
    decay = torch.exp(-A * F.softplus(a + st.raw(base + "dt_bias").to(x.dtype)))  # (T, hv)
    S = torch.zeros(hv, dk, dv, dtype=x.dtype)
    out = torch.empty(T, hv, dv, dtype=x.dtype)
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)  # these tiny per-token ops run fastest on one thread
    for t in range(T):  # S <- decay S; S += k (beta (v - S^T k))^T; o = S^T q
        S.mul_(decay[t][:, None, None])
        d = (v[t] - torch.bmm(k[t][:, None], S)[:, 0]) * beta[t][:, None]
        S.baddbmm_(k[t][:, :, None], d[:, None])
        out[t] = torch.bmm(q[t][:, None], S)[:, 0]
    torch.set_num_threads(n_threads)
    out = rms(out, st.raw(base + "norm.weight").to(x.dtype), eps) * F.silu(z)  # gated norm, plain weight
    return F.linear(out.reshape(T, hv * dv), st.dense(base + "out_proj"))


def swiglu(st, name, x, rows=None):
    g, u, d = (st.dense(f"{name}.{p}_proj", rows) for p in ("gate", "up", "down"))
    return F.linear(F.silu(F.linear(x, g)) * F.linear(x, u), d)


def moe(st: Store, base: str, x, c):
    probs = torch.softmax(F.linear(x, st.dense(base + "gate")), -1)  # softmax over all experts, then top-k
    w, idx = probs.topk(c["num_experts_per_tok"], -1)
    w = w / w.sum(-1, keepdim=True)
    out = torch.zeros_like(x)
    for e in idx.unique().tolist():
        tok, slot = (idx == e).nonzero(as_tuple=True)
        y = swiglu(st, base + "switch_mlp", x[tok], e)
        out.index_add_(0, tok, y * w[tok, slot][:, None])
    shared = swiglu(st, base + "shared_expert", x)
    return out + torch.sigmoid(F.linear(x, st.dense(base + "shared_expert_gate"))) * shared


def forward(st: Store, ids: np.ndarray, out: np.ndarray, log=sys.stderr):
    c, dt = st.cfg, st.dt
    T, eps = len(ids), c["rms_norm_eps"]
    rp = c.get("rope_parameters") or {}
    rot = int(c["head_dim"] * rp.get("partial_rotary_factor", c.get("partial_rotary_factor", 1.0)))
    cos, sin = rope_tables(torch.arange(T), rot, float(rp.get("rope_theta", c.get("rope_theta", 1e7))), dt)
    kinds = c.get("layer_types") or ["full_attention" if (i + 1) % c["full_attention_interval"] == 0 else "linear_attention"
                                     for i in range(c["num_hidden_layers"])]
    h = st.dense(st.pre + "embed_tokens", torch.from_numpy(ids.astype(np.int64)))
    for i, kind in enumerate(kinds):
        t0, base = time.time(), f"{st.pre}layers.{i}."
        st.cache.clear()
        x = rms(h, st.norm(base + "input_layernorm.weight"), eps)
        if kind == "full_attention":
            h = h + full_attention(st, base + "self_attn.", x, cos, sin, c)
        else:
            h = h + linear_attention(st, base + "linear_attn.", x, c)
        x = rms(h, st.norm(base + "post_attention_layernorm.weight"), eps)
        h = h + (moe(st, base + "mlp.", x, c) if c.get("num_experts") else swiglu(st, base + "mlp", x))
        print(f"layer {i:2d} {kind[:6]} {time.time() - t0:6.1f}s", file=log, flush=True)
    h = rms(h, st.norm(st.pre + "norm.weight"), eps)
    head = st.head if st.has(st.head + ".weight") else st.pre + "embed_tokens"
    V = out.shape[1]
    for a in range(0, V, 16384):  # vocab head in chunks, one dequantized slice at a time
        out[:, a:a + 16384] = F.linear(h, st.dense(head, slice(a, min(a + 16384, V)))).float().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model"), ap.add_argument("ids"), ap.add_argument("out")
    ap.add_argument("--dtype", choices=("f64", "f32"), default="f64")
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    ids = np.load(a.ids).reshape(-1)
    st = Store(Path(a.model), torch.float64 if a.dtype == "f64" else torch.float32)
    out = np.empty((len(ids), st.cfg["vocab_size"]), np.float32)
    t0 = time.time()
    with torch.inference_mode():
        forward(st, ids, out)
    np.save(a.out, out)
    np.save(str(a.out).removesuffix(".npy") + ".ids.npy", ids.astype(np.int64))  # next-token targets for score.py
    print(f"{len(ids)} tokens {a.dtype} {time.time() - t0:.1f}s -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
