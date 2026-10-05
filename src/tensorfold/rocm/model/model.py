"""The Qwen text model's layers on RDNA: packed projections, norms, and the optional experts and MTP head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from tensorfold.rocm.model.qwen_math import Packed, Spec

if TYPE_CHECKING:
    from tensorfold.rocm.model.moe import Routed


@dataclass
class LinearLayer:
    input_norm: torch.Tensor
    post_norm: torch.Tensor
    qkv: Packed
    z: Packed
    a: Packed
    b: Packed
    conv: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    gnorm: torch.Tensor
    out: Packed
    gate: Packed
    up: Packed
    down: Packed
    moe: "Routed | None" = None


@dataclass
class FullLayer:
    input_norm: torch.Tensor
    post_norm: torch.Tensor
    q: Packed
    k: Packed
    v: Packed
    o: Packed
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    gate: Packed
    up: Packed
    down: Packed
    moe: "Routed | None" = None


@dataclass
class TextModel:
    spec: Spec
    embed: Packed
    layers: list
    final_norm: torch.Tensor
    head: Packed | None = None
    mtp: "MTPHead | None" = None

    def output_head(self) -> Packed:
        return self.embed if self.head is None else self.head


@dataclass
class MTPHead:
    """An MTP head: the Flash Next shape, or the Qwen3 shape with gated attention and an MLP."""

    fc_e_norm: torch.Tensor
    fc_h_norm: torch.Tensor
    fc_e: Packed
    fc_h: Packed
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    q: Packed
    k: Packed
    v: Packed
    o: Packed
    final_norm: torch.Tensor
    head: Packed | None = None             # None ties with main ``model.output_head()``
    input_norm: torch.Tensor | None = None
    post_norm: torch.Tensor | None = None
    gate: Packed | None = None
    up: Packed | None = None
    down: Packed | None = None
    moe: "Routed | None" = None
    gated: bool = False
