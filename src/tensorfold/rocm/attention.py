"""HIP causal attention. The cache stays in the activation dtype and the kv heads are not repeated."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch

_KIND = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.rocm.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_rocm_attn",
                sources=[str(here / "attention.cpp"), str(here / "attention.hip")],
                extra_include_paths=[str(here)], verbose=False)


@lru_cache(maxsize=1)
def _gfx() -> str:
    from tensorfold.rocm.build import gfx_name

    return gfx_name()


def triton_prefill(gfx: str, batch: int, qlen: int) -> bool:
    """Where the Triton tile beat the HIP tile.

    gfx1100 uses the bf16 WMMA tile. It is faster from 128 queries through 32k,
    and from 64 queries when the batch is 8 or more. Shorter rows stay on HIP.
    gfx1030 has no WMMA, so it keeps the fp16 tile: from 1024 queries, or from
    384 when the batch is 8 or more. The chunked matmul is not a choice.
    """

    if gfx.startswith("gfx103"):
        return qlen >= 1024 or (batch >= 8 and qlen >= 384)
    if gfx.startswith("gfx110"):
        return qlen >= (64 if batch >= 8 else 128)
    return False


def causal(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, q_pos0: int) -> torch.Tensor:
    """``q`` is (batch, heads, qlen, d) fp32. ``k`` and ``v`` are (batch, kv heads, span, d)."""

    q = q.contiguous()
    # One query stays the HIP split over keys. Longer rows take Triton only where it was faster.
    mode = os.environ.get("TENSORFOLD_ATTN", "auto")
    gfx = _gfx()
    if q.shape[2] > 1 and mode != "hip" and (mode == "triton" or triton_prefill(gfx, q.shape[0], q.shape[2])):
        from tensorfold.rocm.attention_triton import prefill

        dot = os.environ.get("TENSORFOLD_ATTN_DOT")
        if dot not in ("bf16", "fp16"):
            dot = "bf16" if gfx.startswith("gfx110") else "fp16"
        tiled = prefill(q, k, v, scale, q_pos0, dot=dot)
        if tiled is not None:
            return tiled
    out = torch.empty_like(q)
    if k.dtype not in _KIND or k.dtype != v.dtype:
        raise ValueError("k and v must be fp16, bf16, or fp32, and they must match")
    _ext().causal(q, k, v, out, float(scale), int(q_pos0))
    return out
