"""HIP causal attention. The cache stays in the activation dtype and the kv heads are not repeated."""

from __future__ import annotations

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


def causal(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, q_pos0: int) -> torch.Tensor:
    """``q`` is (batch, heads, qlen, d) fp32. ``k`` and ``v`` are (batch, kv heads, span, d)."""

    q = q.contiguous()
    out = torch.empty_like(q)
    if k.dtype not in _KIND or k.dtype != v.dtype:
        raise ValueError("k and v must be fp16, bf16, or fp32, and they must match")
    _ext().causal(q, k, v, out, float(scale), int(q_pos0))
    return out
