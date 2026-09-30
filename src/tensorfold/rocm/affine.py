"""Packed MLX affine projections. The kernel reads the integer words and applies each group's scale and bias; it does not write a BF16 copy of the weight."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

BITS = (2, 3, 4, 5, 6, 8)
GROUPS = (32, 64, 128)


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.rocm.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_rocm_affine",
                sources=[str(here / "affine.cpp"), str(here / "affine_gemv.hip"), str(here / "affine_wmma.hip"),
                         str(here / "affine_dot2.hip")],
                extra_include_paths=[str(here)], verbose=False)


def matmul(x: torch.Tensor, words: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor, *, bits: int,
           group: int, schedule: str = "auto", f32: bool = False) -> torch.Tensor:
    """``x`` (M, K) times packed words (N, K * bits / 32). ``x`` is BF16, or FP16 on RDNA2.

    ``schedule`` is ``auto``, ``gemv`` or ``wmma``. FP16 ``auto`` and ``gemv`` are ``v_dot2_f32_f16``.
    """

    from tensorfold.rocm.build import WMMA, gfx_name

    if bits not in BITS or group not in GROUPS:
        raise ValueError("RDNA affine weights require 2/3/4/5/6/8 bits and groups of 32/64/128")
    which = {"auto": 0, "gemv": 1, "wmma": 2}[schedule]
    if x.ndim != 2 or x.dtype not in (torch.bfloat16, torch.float16) or not x.is_cuda or not x.is_contiguous():
        raise ValueError("affine inputs must be a contiguous BF16 or FP16 matrix on the device")
    if x.dtype == torch.float16 and gfx_name() in WMMA:
        raise ValueError("FP16 activations are the RDNA2 schedule")
    m, k = x.shape
    if k % group != 0 or (k * bits) % 32 != 0:
        raise ValueError("K must be whole groups and whole packed words")
    if words.ndim != 2 or words.dtype != torch.int32 or words.shape[1] != k * bits // 32:
        raise ValueError("packed words must be int32 of shape (N, K * bits / 32)")
    n = words.shape[0]
    groups = k // group
    scale = scale.to(torch.float32).contiguous()
    bias = bias.to(torch.float32).contiguous()
    if scale.shape != (n, groups) or bias.shape != scale.shape:
        raise ValueError("scale and bias must be (N, K / group)")
    if not all(t.is_cuda and t.device == x.device for t in (words, scale, bias)):
        raise ValueError("affine operands must share the input's device")
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    _ext().affine(x, words.contiguous(), scale, bias, out, bits, group, which)
    return out if f32 else out.to(x.dtype)
