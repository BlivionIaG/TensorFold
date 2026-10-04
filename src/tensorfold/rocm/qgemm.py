"""W4A16 GPTQ projections on RDNA2 (gfx1030): int4 weights, fp16 activations, fp16 out."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.rocm.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_rocm_qgemm",
                sources=[str(here / "qgemm.cpp"), str(here / "qgemm_rdna2.hip"),
                         str(here / "qgemm_rdna2_prefill.hip"), str(here / "qgemm_moe_rdna2.hip")],
                extra_include_paths=[str(here)], verbose=False)


def matmul(x: torch.Tensor, qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor,
           g_idx: torch.Tensor | None = None, *, use_v2_format: bool = True,
           prefill: bool = False) -> torch.Tensor:
    """``x`` (M, K) fp16 times a GPTQ int4 ``qweight`` (K / 8, N); gfx1030 only."""

    from tensorfold.rocm.build import gfx_name

    if gfx_name() != "gfx1030":
        raise ValueError("the W4A16 GPTQ path is the RDNA2 (gfx1030) fp16 schedule")
    if x.ndim != 2 or x.dtype != torch.float16 or not x.is_cuda or not x.is_contiguous():
        raise ValueError("W4A16 inputs must be a contiguous FP16 matrix on the device")
    if qweight.dtype != torch.int32 or qweight.ndim != 2:
        raise ValueError("qweight must be int32 of shape (K / 8, N)")
    m, k = x.shape
    if qweight.shape[0] * 8 != k:
        raise ValueError("qweight's first dim must be K / 8")
    if qzeros.dtype != torch.int32 or scales.dtype != torch.float16:
        raise ValueError("qzeros must be int32 and scales fp16")
    if g_idx is not None and (g_idx.dtype != torch.int32 or g_idx.numel() != k):
        raise ValueError("g_idx must be int32 of length K, or None")
    empty = g_idx if g_idx is not None else torch.empty(0, dtype=torch.int32, device=x.device)
    return _ext().gptq_matmul(x, qweight.contiguous(), qzeros.contiguous(), scales.contiguous(),
                              empty, bool(use_v2_format), bool(prefill))


def moe(x: torch.Tensor, qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor,
        items: torch.Tensor, members: torch.Tensor, rows: int, slots: int, *, epi: int,
        block_m: int = 4, use_v2_format: bool = True, limit: float = 0.0) -> torch.Tensor:
    """One layer's W4A16 experts over a routing plan; ``epi`` 0 down, 1 relu^2, 2 SwiGLU."""

    from tensorfold.rocm.build import gfx_name

    if gfx_name() != "gfx1030":
        raise ValueError("the W4A16 GPTQ path is the RDNA2 (gfx1030) fp16 schedule")
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous():
        raise ValueError("MoE activations must be a contiguous BF16 matrix on the device")
    if epi not in (0, 1, 2):
        raise ValueError("epi is 0 (fp32), 1 (relu^2) or 2 (SwiGLU)")
    if block_m not in (1, 2, 4, 8):
        raise ValueError("block_m is 1, 2, 4 or 8")
    return _ext().moe_gptq(x, qweight.contiguous(), qzeros.contiguous(), scales.contiguous(),
                           items.contiguous(), members.contiguous(), int(rows), int(slots), int(epi),
                           int(block_m), bool(use_v2_format), float(limit))
