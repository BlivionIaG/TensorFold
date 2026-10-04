"""Packed MLX affine projections. The kernel reads the integer words and applies each group's scale and bias."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

BITS = (2, 3, 4, 5, 6, 8)
GROUPS = (32, 64, 128)


SCALE_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def _tables(scale: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale and bias as stored when they are fp32, bf16 or fp16 of one type; the kernel widens them exactly."""

    if scale.dtype != bias.dtype or scale.dtype not in SCALE_DTYPES:
        scale, bias = scale.to(torch.float32), bias.to(torch.float32)
    return scale.contiguous(), bias.contiguous()


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.rocm.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_rocm_affine",
                sources=[str(here / name) for name in ("affine.cpp", "affine_launch.hip", "affine_gemv.hip",
                                                       "affine_dot2.hip", "affine_tiles.hip")],
                extra_include_paths=[str(here)], verbose=False)


def matmul(x: torch.Tensor, words: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor, *, bits: int,
           group: int, schedule: str = "auto", f32: bool = False, dot2_split: bool | None = None) -> torch.Tensor:
    """``x`` (M, K) BF16, or FP16 on RDNA2, times packed words (N, K * bits / 32); ``schedule`` auto or gemv."""

    from tensorfold.rocm.build import BF16_DOT2, gfx_name

    if bits not in BITS or group not in GROUPS:
        raise ValueError("RDNA affine weights require 2/3/4/5/6/8 bits and groups of 32/64/128")
    which = {"auto": 0, "gemv": 1}[schedule]
    if x.ndim != 2 or x.dtype not in (torch.bfloat16, torch.float16) or not x.is_cuda or not x.is_contiguous():
        raise ValueError("affine inputs must be a contiguous BF16 or FP16 matrix on the device")
    if x.dtype == torch.float16 and gfx_name() in BF16_DOT2:
        raise ValueError("FP16 activations are the RDNA2 schedule")
    m, k = x.shape
    if k % group != 0 or (k * bits) % 32 != 0:
        raise ValueError("K must be whole groups and whole packed words")
    if words.ndim != 2 or words.dtype != torch.int32 or words.shape[1] != k * bits // 32:
        raise ValueError("packed words must be int32 of shape (N, K * bits / 32)")
    n = words.shape[0]
    groups = k // group
    scale, bias = _tables(scale, bias)
    if scale.shape != (n, groups) or bias.shape != scale.shape:
        raise ValueError("scale and bias must be (N, K / group)")
    if not all(t.is_cuda and t.device == x.device for t in (words, scale, bias)):
        raise ValueError("affine operands must share the input's device")
    # None follows the column grid. False is one launch. True splits K on group boundaries.
    split_mode = 0 if dot2_split is None else (2 if dot2_split else 1)
    # The RDNA2 decode tile rounds its fp16 output itself: the same bits as the fp32 result cast after.
    half = (not f32 and x.dtype == torch.float16 and m <= 8 and which == 0 and split_mode != 2
            and gfx_name() not in BF16_DOT2)
    out = torch.empty((m, n), dtype=torch.float16 if half else torch.float32, device=x.device)
    _ext().affine(x, words.contiguous(), scale, bias, out, bits, group, which, split_mode)
    return out if f32 or half else out.to(x.dtype)


def matmul_routed(x: torch.Tensor, words: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor, items: torch.Tensor,
                  members: torch.Tensor, *, pairs: int, x_div: int, rows: int, bits: int, group: int) -> torch.Tensor:
    """Every item (expert, first, count) of a plan in one launch: (pairs, N) fp32 by pair id."""

    if bits not in BITS or group not in GROUPS:
        raise ValueError("RDNA affine weights require 2/3/4/5/6/8 bits and groups of 32/64/128")
    scale, bias = _tables(scale, bias)
    out = torch.empty((pairs, words.shape[1]), dtype=torch.float32, device=x.device)
    _ext().affine_routed(x.contiguous(), words, scale, bias, out, items, members, x_div, rows, bits, group)
    return out
