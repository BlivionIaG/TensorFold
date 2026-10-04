"""Startup memory plan on ROCm: the context window one request can use and the prompt cache's bytes, as on CUDA."""

from __future__ import annotations

import torch

from tensorfold.cuda.capacity import GIB, reserve_bytes


def state_bytes(spec, tokens: int, act_bytes: int) -> int:
    """One request's cache at ``tokens``: keys and values of the attention layers plus the linear layers' state."""

    full = sum(spec.full(index) for index in range(spec.n_layers))
    linear = spec.n_layers - full
    keys = full * 2 * spec.kv_heads * spec.head_dim * act_bytes * tokens
    conv = (spec.key_width * 2 + spec.value_width) * (spec.conv - 1) * 4
    recurrent = spec.value_heads * spec.value_dim * spec.key_dim * 4
    return int(keys + linear * (conv + recurrent))


def plan(engine, native: int, requested: int | None, explicit: bool, byte_budget: int | None,
         rccl) -> tuple[int, int | None]:
    """(window, prompt-cache bytes) for this rank's memory, the ranks' least; an explicit window that does not fit
    is refused. The window leaves room for one request and one kept copy of its prompt, as on Macs."""

    if not torch.cuda.is_available():
        return requested or native, byte_budget
    spec, act = engine.model.spec, torch.finfo(engine._dtype()).bits // 8
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    resident = torch.cuda.memory_allocated()
    engine.warm()                                         # one prefill span: kernels built, workspace measured
    workspace = torch.cuda.max_memory_allocated() - resident
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    room = free - reserve_bytes(total) - workspace
    per_token = max(1, state_bytes(spec, 1, act) - state_bytes(spec, 0, act))
    fixed = state_bytes(spec, 0, act)
    fitting = max(0, (room - 2 * fixed) // (2 * per_token))
    target = requested or native
    window = min(target, fitting) if target else fitting
    cache = byte_budget if byte_budget is not None else max(0, room - state_bytes(spec, window, act))
    agreed = torch.tensor([window, cache], dtype=torch.int64, device="cuda")
    if rccl is not None:
        rccl.all_reduce(agreed, agreed, op="min")
    window, cache = (int(value) for value in agreed.tolist())
    if explicit and requested and requested > window:
        raise ValueError(f"--context {requested} does not fit this GPU's memory: one request and a kept copy of its "
                         f"prompt fit {window} tokens beside the weights; lower --context or add ranks (--tp)")
    if window <= 0:
        raise ValueError("the weights leave no room for a request on this GPU; add ranks (--tp)")
    print(f"[tensorfold] ROCm rank {engine.rank}: weights {resident / GIB:.2f} GiB, prefill workspace "
          f"{workspace / GIB:.2f} GiB, context window {window:,} tokens, prompt cache "
          f"{(cache if byte_budget is None else byte_budget) / GIB:.2f} GiB, reserve {reserve_bytes(total) / GIB:.2f} "
          f"GiB of {total / GIB:.2f}", flush=True)
    return window, cache
