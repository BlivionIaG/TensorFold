"""Prefill attention as a 64-row Flash Attention 2 tile on ROCm Triton.

Dao-AILab/flash-attention is the upstream AMD kernel. This is that schedule
against the Triton beside the ROCm wheel, so serving does not install ``aiter``.
gfx1100 uses a bf16 WMMA dot. gfx1030 uses fp16. ``causal`` calls the tile
only on the lengths where that dot beat the HIP tile. One new token stays the
HIP split-key walk.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # a wheel without Triton keeps the HIP tile
    triton = None
    tl = None

# 64 query rows is the long-prefill tile. 16 rows reloads the cache four times as often.
BM = 64
BN = 64

_off = False


if tl is not None:

    @triton.jit
    def _attend(Q, K, V, OUT, q_pos0, qlen, span, scale,
                stride_qb, stride_qh, stride_qs, stride_qd,
                stride_kb, stride_kh, stride_ks, stride_kd,
                stride_vb, stride_vh, stride_vs, stride_vd,
                H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, DOT: tl.constexpr):
        batch = tl.program_id(0)
        head = tl.program_id(1)
        block = tl.program_id(2)
        hk = head // (H // HK)
        rows = block * BM + tl.arange(0, BM)
        ok = rows < qlen
        pos = q_pos0 + rows
        offs_d = tl.arange(0, D)
        q = tl.load(Q + batch * stride_qb + head * stride_qh + rows[:, None] * stride_qs
                    + offs_d[None, :] * stride_qd, mask=ok[:, None], other=0.0)
        m_i = tl.full((BM,), float("-inf"), tl.float32)
        l_i = tl.zeros((BM,), tl.float32)
        acc = tl.zeros((BM, D), tl.float32)
        q0 = block * BM
        visible = tl.minimum(span, q_pos0 + q0 + tl.minimum(BM, qlen - q0))
        n_tiles = (visible + BN - 1) // BN
        tile = 0
        while tile < n_tiles:
            keys = tile * BN + tl.arange(0, BN)
            in_span = keys < span
            k = tl.load(K + batch * stride_kb + hk * stride_kh + keys[:, None] * stride_ks
                        + offs_d[None, :] * stride_kd, mask=in_span[:, None], other=0.0)
            v = tl.load(V + batch * stride_vb + hk * stride_vh + keys[:, None] * stride_vs
                        + offs_d[None, :] * stride_vd, mask=in_span[:, None], other=0.0)
            # DOT 1 is the gfx1100 bf16 WMMA product. gfx1030 stays on the fp16 dot.
            if DOT == 1:
                score = tl.dot(q.to(tl.bfloat16), tl.trans(k.to(tl.bfloat16))).to(tl.float32) * scale
            else:
                score = tl.dot(q.to(tl.float16), tl.trans(k.to(tl.float16))).to(tl.float32) * scale
            valid = ok[:, None] & in_span[None, :] & (keys[None, :] <= pos[:, None])
            score = tl.where(valid, score, float("-inf"))
            tile_m = tl.max(score, 1)
            active = tile_m != float("-inf")
            next_m = tl.where(active, tl.maximum(m_i, tile_m), m_i)
            alpha = tl.where(active, tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - next_m)), 1.0)
            prob = tl.where(valid & active[:, None], tl.exp(score - next_m[:, None]), 0.0)
            if DOT == 1:
                acc = acc * alpha[:, None] + tl.dot(prob.to(tl.bfloat16), v.to(tl.bfloat16)).to(tl.float32)
            else:
                acc = acc * alpha[:, None] + tl.dot(prob.to(tl.float16), v.to(tl.float16)).to(tl.float32)
            l_i = l_i * alpha + tl.sum(prob, 1)
            m_i = next_m
            tile += 1
        out = tl.where(l_i[:, None] == 0.0, 0.0, acc / l_i[:, None])
        tl.store(OUT + batch * stride_qb + head * stride_qh + rows[:, None] * stride_qs
                 + offs_d[None, :] * stride_qd, out, mask=ok[:, None])


def _rdna() -> bool:
    """gfx1030 and gfx11. gfx12 is a different WMMA ABI and is not this tile."""

    from tensorfold.rocm.build import gfx_name

    name = gfx_name()
    return name.startswith("gfx103") or name.startswith("gfx11")


def prefill(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, q_pos0: int, *,
            force: bool = False, dot: str | None = None) -> torch.Tensor | None:
    """The 64-row tile. ``None`` means the caller keeps the HIP tile.

    ``force`` is the direct test. ``TENSORFOLD_ATTN=hip`` keeps the HIP tile, and
    ``TENSORFOLD_ATTN=triton`` reports a launch failure instead of falling back.
    """

    global _off
    mode = os.environ.get("TENSORFOLD_ATTN", "auto")
    if not force and (mode == "hip" or _off or triton is None or not _rdna()):
        return None
    if triton is None:
        raise RuntimeError("Triton attention needs the Triton package next to this ROCm PyTorch")
    if q.shape[2] <= 1 or q.shape[-1] < 16 or q.shape[-1] > 256 or q.shape[-1] % 16 != 0:
        return None
    if q.shape[1] % k.shape[1] != 0 or k.shape != v.shape or not q.is_cuda:
        return None
    out = torch.empty_like(q)
    heads, kv_heads, dim = q.shape[1], k.shape[1], q.shape[3]
    if dot is None:
        dot = "bf16" if os.environ.get("TENSORFOLD_ATTN_DOT") == "bf16" else "fp16"
    dot_id = 1 if dot == "bf16" else 0
    grid = (q.shape[0], heads, triton.cdiv(q.shape[2], BM))
    try:
        _attend[grid](q, k, v, out, int(q_pos0), int(q.shape[2]), int(k.shape[2]), float(scale),
                      q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                      k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                      v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                      H=heads, HK=kv_heads, D=dim, BM=BM, BN=BN, DOT=dot_id,
                      num_warps=8 if dim >= 128 else 4, num_stages=1)
    except Exception as exc:
        if force or mode == "triton":
            raise
        _off = True
        print(f"[tensorfold] Triton attention is off ({exc}); prefill stays on the HIP tile", flush=True)
        return None
    return out
