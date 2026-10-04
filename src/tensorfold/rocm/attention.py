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


def triton_prefill(gfx: str) -> bool:
    """True where every prefill row takes the Triton tile; gfx11's WMMA product lets masked values move the last bit."""

    return gfx.startswith("gfx103")


def causal(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, q_pos0: int, *,
           prefill: bool = False) -> torch.Tensor:
    """``q`` is (batch, heads, qlen, d) fp32. ``k`` and ``v`` are (batch, kv heads, span, d)."""

    q = q.contiguous()
    # Prefill rows take one kernel whatever their span, so a resumed prompt repeats a fresh one's bits.
    mode = os.environ.get("TENSORFOLD_ATTN", "auto")
    gfx = _gfx()
    if prefill and mode != "hip" and (mode == "triton" or triton_prefill(gfx)):
        from tensorfold.rocm.attention_triton import prefill as tiled_prefill

        dot = os.environ.get("TENSORFOLD_ATTN_DOT")
        if dot not in ("bf16", "fp16"):
            dot = "bf16" if gfx.startswith("gfx110") else "fp16"
        tiled = tiled_prefill(q, k, v, scale, q_pos0, dot=dot)
        if tiled is not None:
            return tiled
    out = torch.empty_like(q)
    if k.dtype not in _KIND or k.dtype != v.dtype:
        raise ValueError("k and v must be fp16, bf16, or fp32, and they must match")
    _ext().causal(q, k, v, out, float(scale), int(q_pos0), bool(prefill))
    return out


def causal_at(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, pos: torch.Tensor) -> torch.Tensor:
    """One query at device position ``pos`` (int32, one element) over the whole cache: graph-capturable decode."""

    if k.dtype not in _KIND or k.dtype != v.dtype:
        raise ValueError("k and v must be fp16, bf16, or fp32, and they must match")
    q = q.contiguous()
    out = torch.empty_like(q)
    _ext().causal_at(q, k, v, out, float(scale), pos)
    return out
