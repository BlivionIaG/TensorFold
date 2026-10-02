"""The Qwen3.5 ROCm MTP drafter skeleton (Phase 1: state only).

Phase 2 wires the head's forward on top of ``qwen_math._attention_span`` so the same per-layer KV
machinery the main model uses. Same shape as ``tensorfold.families.qwen4_exp.cuda.mtp``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MTPState:
    k: torch.Tensor = None
    v: torch.Tensor = None
    pos: int = 0
    capacity: int = 0

    def reset(self) -> None:
        if self.k is not None:
            self.k.zero_()
            self.v.zero_()
        self.pos = 0


__all__ = ["MTPState"]