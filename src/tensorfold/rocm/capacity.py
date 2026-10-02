"""ROCm startup estimates and a per-rank capacity decision.

The dataclasses and tensor-geometry helpers in :mod:`tensorfold.cuda.capacity` are pure-Python
(safetensors headers + struct + json), so they are reused here. The ROCm-specific overrides:

- ``floor`` returns the gfx target the family's kernels need (RDNA 2 / RDNA 3 / RDNA 3.5 / RDNA 4).
  Anything else (RDNA 1, unlisted gfx, CDNA Instinct on consumer drivers) is refused by the build.
- ``admit`` does the per-rank startup decision; the ``gather`` callable, supplied by the family's
  ROCm app, abstracts RCCL on tp > 1 and the local list on tp = 1, so this module is the same code path
  for both single-process and multi-rank starts.

The capacity message names "ROCm rank"; nothing in the body is GPU-vendor-specific beyond the build check.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Callable

from tensorfold.cuda.capacity import (GIB, Geometry, Plan, SIZES, Weights, choose, config, estimate_weights,
                                      headers, tables_note)


def gfx() -> str:
    """The visible device's gfx target; refuses hardware the kernels have no schedule for."""

    from tensorfold.rocm.build import gfx_name

    return gfx_name()


def admit(model_dir: str | Path, requested: int | None, explicit: bool | None, torch,
          geometry: Geometry | Callable, transform: Callable, *, rank: int = 0, world: int = 1,
          gather: Callable | None = None, draft_dir: Path | None = None,
          draft_geometry: Geometry | Callable | None = None, startup_copies: int = 0,
          extra_files: tuple[Path, ...] = (), files: list[Path] | None = None,
          draft_transform: Callable | None = None,
          draft_weights: Callable[[Path], Weights] | None = None) -> dict:
    """Per-rank: refuse, print and admit. The ``gather`` callable lists every rank's ``status``."""

    from tensorfold.cuda.capacity import (itemsize as _itemsize, make_plan, page_room,
                                          available_bytes, host_stream_bytes, unified)
    import math

    gfx()
    error = None
    plan = None
    try:
        text = config(model_dir)
        geometry = geometry(text) if callable(geometry) else geometry
        weights = estimate_weights(model_dir, transform, rank=rank, files=files)
        host_staging = weights.staging
        if extra_files:
            more = estimate_weights(model_dir, transform, files=list(extra_files))
            host_staging = max(host_staging, more.staging)
            weights = Weights(weights.resident + more.resident, max(weights.staging, more.staging),
                              weights.mapped + more.mapped)
        weights = Weights(weights.resident, weights.staging + startup_copies * weights.resident, weights.mapped)
        if draft_dir is not None:
            draft = (draft_weights(draft_dir) if draft_weights is not None
                     else estimate_weights(
                         draft_dir,
                         draft_transform or (lambda name, info: (math.prod(info["shape"])
                                                                  * max(4, _itemsize(info, name)),
                                                                  0))))
            host_staging = max(host_staging, draft.staging)
            weights = Weights(weights.resident + draft.resident,
                              max(weights.staging - draft.resident, draft.staging),
                              weights.mapped)
            if draft_geometry is not None:
                draft_geometry = (draft_geometry(config(draft_dir)) if callable(draft_geometry)
                                  else draft_geometry)
                main = geometry
                geometry = Geometry(lambda slots: main.bytes_at(slots) + draft_geometry.bytes_at(slots),
                                    main.reserve, main.minimum_slots)
        if not unified(torch):
            host_free = host_stream_bytes()
            if host_free is not None and host_staging > host_free:
                raise ValueError(f"host staging needs an estimated {host_staging / GIB:.2f} GiB, "
                                 f"but only {host_free / GIB:.2f} GiB is available after its reserve; "
                                 "free host memory or use a checkpoint with smaller loading buffers")
        plan = make_plan(int(text.get("max_position_embeddings") or 0), requested,
                         requested is not None if explicit is None else explicit,
                         available_bytes(torch), weights, geometry, room=page_room(torch))
    except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
        error = f"{type(exc).__name__}: {exc}"
    status = [1 if error else 0,
              *(plan.settings + [plan.fitting, plan.largest] if plan else [0, -1, 0, 0, 0])]
    both = gather(status) if world > 1 and gather is not None else [status]
    if any(row[0] for row in both):
        raise ValueError("ROCm startup memory geometry could not be established on every rank: " +
                         (error or "another rank could not read its checkpoint; check both folders/configs"))
    window = choose(plan, [row[1:] for row in both])
    receipt = {**plan.receipt(window), "largest_window": min(row[5] for row in both)}
    print(f"[tensorfold] ROCm rank {rank} startup estimate {receipt['total_bytes_estimate'] / GIB:.2f} GiB "
          f"within {plan.budget / GIB:.2f} GiB; native {plan.native}, allocated prompt/reply window {window}, "
          f"cache slots {receipt['cache_slots']}", flush=True)
    note = tables_note(plan)
    if note:
        print(f"[tensorfold] {note}", flush=True)
    return receipt


__all__ = ["GIB", "Geometry", "Plan", "SIZES", "Weights", "admit", "gfx", "choose", "config", "estimate_weights",
           "headers", "tables_note"]