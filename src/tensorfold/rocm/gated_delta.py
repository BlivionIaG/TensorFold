"""HIP gated delta for the Qwen3.5 linear layers. The Python reference owns the same lane reduction."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.rocm.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_rocm_gdn",
                sources=[str(here / "gated_delta.cpp"), str(here / "gated_delta.hip")],
                extra_include_paths=[str(here)], verbose=False)


def recurrence(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gate: torch.Tensor, beta: torch.Tensor,
               state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the wave kernel. ``state`` is updated in place and returned."""

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    gate = gate.contiguous()
    beta = beta.contiguous()
    state = state.contiguous()
    y = torch.empty_like(v)
    _ext().gdn(q, k, v, gate, beta, state, y)
    # q, k, v, gate and beta can die when this returns. The kernel has to be finished first.
    torch.cuda.synchronize()
    return y, state
