"""Short-row RMSNorm, length-1 causal conv, and length-1 RoPE. Prefill keeps the PyTorch ops."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.rocm.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_rocm_act",
                sources=[str(here / "act.cpp"), str(here / "act.hip")],
                extra_include_paths=[str(here)], verbose=False)


def rms(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """``x`` is (rows, width) fp32, fp16 or bf16; ``y`` has its dtype, computed in fp32. ``weight`` is fp32."""

    y = torch.empty_like(x)
    _ext().rms(x, weight if weight is not None else torch.empty(0, device=x.device), y, float(eps))
    return y


def conv_decode(x: torch.Tensor, weight: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """Length-1 depthwise conv. ``state`` is updated in place."""

    y = torch.empty_like(x)
    _ext().conv_decode(x, weight, state, y)
    return y


def rope_decode(x: torch.Tensor, pos: int, rotary: int, theta: float) -> torch.Tensor:
    """Rotate the first ``rotary`` columns of one position."""

    y = torch.empty_like(x)
    _ext().rope_decode(x, y, int(pos), int(rotary), float(theta))
    return y
