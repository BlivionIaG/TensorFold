"""A lone stream's window forward replayed from a captured HIP graph, one per window length, with the eager bits."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import torch

from tensorfold.rocm.model.window import Window, window_forward


@dataclass
class _Entry:
    """One window length's buffers: token ids and row positions on the device, and the captured graph's outputs."""

    device: torch.Tensor                 # (2, rows) int64: the token ids, then the rows' positions
    host: torch.Tensor                   # the same, pinned, so one copy sets both without blocking
    warm: bool = False
    graph: torch.cuda.CUDAGraph | None = None
    hidden: torch.Tensor | None = None
    states: dict = field(default_factory=dict)

    @property
    def ids(self) -> torch.Tensor:
        return self.device[0:1]

    @property
    def slots(self) -> torch.Tensor:
        return self.device[1]


class WindowGraphs:
    """Graphs bound to one stream's caches; a different stream, or its end, drops them."""

    def __init__(self, engine) -> None:
        self.e = engine
        self.sid: int | None = None
        self.entries: dict[int, _Entry] = {}
        self.pool = None

    def reset(self) -> None:
        self.sid, self.entries, self.pool = None, {}, None

    def forward(self, sid: int, window: Window, reduce=None) -> torch.Tensor:
        """The window's hidden rows: eager the first time a length is seen, captured the second, replayed after."""

        e = self.e
        if sid != self.sid:
            self.reset()
            self.sid = sid
        rows = len(window.tokens)
        entry = self.entries.get(rows)
        if entry is None:
            device = e._device()
            entry = self.entries[rows] = _Entry(torch.zeros((2, rows), dtype=torch.int64, device=device),
                                                torch.zeros((2, rows), dtype=torch.int64, pin_memory=True))
        if rows == 1:                                         # two fills cost less than a copy
            entry.device[0].fill_(window.tokens[0])
            entry.device[1].fill_(window.pos)
        else:
            host = entry.host.numpy()
            host[0] = window.tokens
            host[1] = range(window.pos, window.pos + rows)
            entry.device.copy_(entry.host, non_blocking=True)  # the last round's copy is done: its logits were read
        window.slots = entry.slots
        run = lambda: window_forward(e.model, [window], e.kernels.linear, e._dtype(), ids=entry.ids,
                                     reduce=reduce)
        with torch.inference_mode():
            if entry.graph is not None:
                entry.graph.replay()
                window.states = entry.states
                return entry.hidden
            if not entry.warm:
                entry.warm = True
                return run()
            graph = torch.cuda.CUDAGraph()
            if self.pool is None:
                self.pool = torch.cuda.graph_pool_handle()
            failed = None
            try:
                # thread_local: a request thread syncing the GPU meanwhile does not break the lane worker's capture.
                with torch.cuda.graph(graph, pool=self.pool, capture_error_mode="thread_local"):
                    entry.hidden = run()
                    entry.states = window.states
            except Exception as exc:  # noqa: BLE001 - any capture failure: this engine stays eager
                failed = exc
            if self._any_rank(failed is not None):    # the ranks replay together or not at all
                print(f"[tensorfold] window graph capture failed, decoding eagerly: {failed}", file=sys.stderr)
                e.graphs = False
                self.reset()
                torch.cuda.synchronize()
                return run()
            entry.graph = graph
            graph.replay()
            window.states = entry.states
            return entry.hidden


    def _any_rank(self, flag: bool) -> bool:
        """True on every rank when ``flag`` is true on one (one rank: ``flag``)."""

        e = self.e
        if e.tp <= 1:
            return flag
        value = torch.tensor([int(flag)], dtype=torch.int32, device=e._device())
        e.rccl.all_reduce(value, value, op="max")
        return bool(value.item())


__all__ = ["WindowGraphs"]
