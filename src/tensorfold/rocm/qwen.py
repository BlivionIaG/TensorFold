"""Qwen3.5 / Qwen3.6 text model on RDNA: the projection dispatch, and the model's load and tp entry points."""

from __future__ import annotations

import torch

from tensorfold.rocm import qwen_math
from tensorfold.rocm.checkpoint import load, load_mtp_head  # noqa: F401 - the package's entry points
from tensorfold.rocm.forward import greedy
from tensorfold.rocm.model import FullLayer, LinearLayer, MTPHead, TextModel  # noqa: F401
from tensorfold.rocm.qwen_math import Dense, GptqPacked, Packed, Spec  # noqa: F401 - Spec for callers
from tensorfold.rocm.slicing import slice_for_tp  # noqa: F401

_RDNA2 = {f"gfx103{i}" for i in range(7)}


def activation_dtype(gfx: str) -> torch.dtype:
    """FP16 on RDNA2 (the FP16 dot2), BF16 on gfx11 / gfx12 (the BF16 dot2)."""

    from tensorfold.rocm.build import BF16_DOT2

    if gfx in _RDNA2:
        return torch.float16
    if gfx in BF16_DOT2:
        return torch.bfloat16
    raise RuntimeError(f"no activation dtype for {gfx}")


class Engine:
    """The forward's projections: ``schedule`` picks the affine kernel, ``rccl`` the tp ring."""

    def __init__(self, model: TextModel, schedule: str = "auto", dtype: torch.dtype | None = None, rccl=None):
        self.model = model
        self.schedule = schedule
        self.dtype = dtype
        self.rccl = rccl
        self.projections = 0

    def linear(self, flat: torch.Tensor, packed: Packed) -> torch.Tensor:
        from tensorfold.rocm import affine as affine_mod

        if self.dtype is None:
            from tensorfold.rocm.build import gfx_name

            self.dtype = activation_dtype(gfx_name())
        flat = flat.reshape(-1, flat.shape[-1]).contiguous()
        if isinstance(packed, GptqPacked):
            return self._gptq(flat, packed)
        if isinstance(packed, Dense):
            return torch.nn.functional.linear(flat.float(), packed.weight).to(self.dtype)
        flat = flat.to(dtype=self.dtype)
        self._note(flat, packed)
        words, scale, bias = packed.words, packed.scale, packed.bias
        schedule = self.schedule
        # A tensor-parallel share along K stays fp32 until the ranks are summed.
        kwargs = {"bits": packed.bits, "group": packed.group, "schedule": schedule, "f32": packed.partial}
        span = qwen_math.SPAN
        if flat.shape[0] <= span:
            return affine_mod.matmul(flat, words, scale, bias, **kwargs)
        # One output buffer. Keeping every chunk and then concatenating doubles a long prefill.
        out = torch.empty(flat.shape[0], words.shape[0], dtype=torch.float32 if packed.partial else flat.dtype,
                          device=flat.device)
        for start in range(0, flat.shape[0], span):
            stop = min(start + span, flat.shape[0])
            out[start:stop] = affine_mod.matmul(flat[start:stop], words, scale, bias, **kwargs)
        return out

    def _note(self, flat: torch.Tensor, packed: Packed) -> None:
        words = packed.words
        k = flat.shape[-1]
        expect = k * packed.bits // 32
        if words.dtype != torch.int32 or words.ndim != 2 or words.shape[1] != expect:
            raise RuntimeError("a projection handed the matmul a weight that is not packed int32 words")
        self.projections += 1

    def _gptq(self, flat: torch.Tensor, packed: GptqPacked) -> torch.Tensor:
        """A W4A16 GPTQ projection: the RDNA2 fp16 dot, tiled by the prefill row counts."""

        from tensorfold.rocm import qgemm

        if packed.g_idx is not None:
            raise ValueError("the RDNA W4A16 path does not carry an act-order permutation yet")
        x = flat.to(dtype=torch.float16).contiguous()
        qweight, qzeros, scales = packed.qweight, packed.qzeros, packed.scales
        span = qwen_math.SPAN
        if x.shape[0] <= span:
            return qgemm.matmul(x, qweight, qzeros, scales, use_v2_format=packed.v2, prefill=x.shape[0] > 16)
        out = torch.empty(x.shape[0], qweight.shape[1], dtype=torch.float16, device=x.device)
        for start in range(0, x.shape[0], span):
            stop = min(start + span, x.shape[0])
            out[start:stop] = qgemm.matmul(x[start:stop], qweight, qzeros, scales, use_v2_format=packed.v2,
                                           prefill=True)
        return out

    def generate(self, prompts: list[list[int]], n_new: int, after_token=None) -> list[list[int]]:
        device = self.model.embed.words.device
        if self.dtype is None:
            from tensorfold.rocm.build import gfx_name

            self.dtype = activation_dtype(gfx_name())
        hooks = {}
        if self.rccl is not None and self.rccl.world > 1:
            from functools import partial

            from tensorfold.rocm.qwen_tp import all_reduce_local, vocab_gather

            hooks = {"reduce": partial(all_reduce_local, self.rccl), "gather": partial(vocab_gather, self.rccl)}
        with torch.inference_mode():
            return greedy(self.model, prompts, n_new, self.linear, device, after_token=after_token,
                          cache_dtype=self.dtype, **hooks)


