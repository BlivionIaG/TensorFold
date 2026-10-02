"""Packed MLX affine projections. The kernel reads the integer words and applies each group's scale and bias."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

BITS = (2, 3, 4, 5, 6, 8)
GROUPS = (32, 64, 128)


def _fp32(tensor: torch.Tensor) -> torch.Tensor:
    """fp32 and contiguous already stays put, so a decode step does not recast the scales."""

    if tensor.dtype == torch.float32 and tensor.is_contiguous():
        return tensor
    return tensor.to(dtype=torch.float32).contiguous()


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.rocm.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_rocm_affine",
                sources=[str(here / "affine.cpp"), str(here / "affine_gemv.hip"), str(here / "affine_wmma.hip"),
                         str(here / "affine_dot2.hip")],
                extra_include_paths=[str(here)], verbose=False)


def matmul(x: torch.Tensor, words: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor, *, bits: int,
           group: int, schedule: str = "auto", f32: bool = False, dot2_split: bool | None = None) -> torch.Tensor:
    """``x`` (M, K) times packed words (N, K * bits / 32). ``x`` is BF16, or FP16 on RDNA2.

    ``schedule`` is ``auto``, ``gemv``, ``wmma`` or ``decode`` (the 8-bit column stream, up to 16 rows, whose
    sum order differs from WMMA). FP16 ``auto`` and ``gemv`` are ``v_dot2_f32_f16``.
    """

    from tensorfold.rocm.build import WMMA, gfx_name

    if bits not in BITS or group not in GROUPS:
        raise ValueError("RDNA affine weights require 2/3/4/5/6/8 bits and groups of 32/64/128")
    which = {"auto": 0, "gemv": 1, "wmma": 2, "decode": 3}[schedule]
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
    scale, bias = _fp32(scale), _fp32(bias)
    if scale.shape != (n, groups) or bias.shape != scale.shape:
        raise ValueError("scale and bias must be (N, K / group)")
    if not all(t.is_cuda and t.device == x.device for t in (words, scale, bias)):
        raise ValueError("affine operands must share the input's device")
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    # None follows the column grid. False is one launch. True splits K on group boundaries.
    split_mode = 0 if dot2_split is None else (2 if dot2_split else 1)
    _ext().affine(x, words.contiguous(), scale, bias, out, bits, group, which, split_mode)
    return out if f32 else out.to(x.dtype)


def _as_affine(words, scale, bias, k, bits, group):
    if words.dtype != torch.int32 or words.ndim != 2 or words.shape[1] != k * bits // 32:
        raise ValueError("packed words must be int32 of shape (N, K * bits / 32)")
    n = words.shape[0]
    groups = k // group
    scale, bias = _fp32(scale), _fp32(bias)
    if scale.shape != (n, groups) or bias.shape != scale.shape:
        raise ValueError("scale and bias must be (N, K / group)")
    return words.contiguous(), scale, bias, n


def matmul_pair(x: torch.Tensor, words_a: torch.Tensor, scale_a: torch.Tensor, bias_a: torch.Tensor,
                words_b: torch.Tensor, scale_b: torch.Tensor, bias_b: torch.Tensor, *, bits: int, group: int,
                f32: bool = False):
    """Two packed products that share ``x``. Each side matches a solo WMMA launch. 8-bit only."""

    from tensorfold.rocm.build import WMMA, gfx_name

    if x.dtype != torch.bfloat16 or gfx_name() not in WMMA or bits != 8:
        raise ValueError("the paired matmul is the BF16 WMMA 8-bit schedule")
    if x.ndim != 2 or not x.is_cuda or not x.is_contiguous():
        raise ValueError("affine inputs must be a contiguous BF16 matrix on the device")
    m, k = x.shape
    if k % group != 0:
        raise ValueError("K must be whole groups")
    wa, sa, ba, na = _as_affine(words_a, scale_a, bias_a, k, bits, group)
    wb, sb, bb, nb = _as_affine(words_b, scale_b, bias_b, k, bits, group)
    if na != nb:
        raise ValueError("a paired matmul needs both sides to share N")
    out_a = torch.empty((m, na), dtype=torch.float32, device=x.device)
    out_b = torch.empty((m, nb), dtype=torch.float32, device=x.device)
    _ext().affine_pair(x, wa, sa, ba, out_a, wb, sb, bb, out_b, bits, group)
    if f32:
        return out_a, out_b
    return out_a.to(x.dtype), out_b.to(x.dtype)


def matmul_group(x: torch.Tensor, packeds: tuple, *, bits: int, group: int, f32: bool = False):
    """Up to four packed products that share ``x``. Each side matches a solo WMMA launch."""

    from tensorfold.rocm.build import WMMA, gfx_name

    if not 1 <= len(packeds) <= 4:
        raise ValueError("a grouped matmul takes 1 to 4 weights")
    if x.dtype != torch.bfloat16 or gfx_name() not in WMMA or bits != 8:
        raise ValueError("the grouped matmul is the BF16 WMMA 8-bit schedule")
    if x.ndim != 2 or not x.is_cuda or not x.is_contiguous():
        raise ValueError("affine inputs must be a contiguous BF16 matrix on the device")
    m, k = x.shape
    if k % group != 0:
        raise ValueError("K must be whole groups")
    words, scale, bias, outs = [], [], [], []
    for packed in packeds:
        w, s, b, n = _as_affine(*packed, k, bits, group)
        words.append(w)
        scale.append(s)
        bias.append(b)
        outs.append(torch.empty((m, n), dtype=torch.float32, device=x.device))
    _ext().affine_group(x, words, scale, bias, outs, bits, group)
    if f32:
        return tuple(outs)
    return tuple(y.to(x.dtype) for y in outs)
