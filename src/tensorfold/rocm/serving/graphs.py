"""A lone stream's window forward replayed from a captured HIP graph, one per window length, with the eager bits."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import torch

from tensorfold.rocm.model.window import Window, window_forward


@dataclass
class _Entry:
    """One window length's buffers: token ids and row positions on the device, and the captured graph's outputs."""

    ids: torch.Tensor
    slots: torch.Tensor
    warm: bool = False
    graph: torch.cuda.CUDAGraph | None = None
    hidden: torch.Tensor | None = None
    states: dict = field(default_factory=dict)


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
            entry = self.entries[rows] = _Entry(torch.zeros((1, rows), dtype=torch.long, device=device),
                                                torch.zeros(rows, dtype=torch.int64, device=device))
        entry.ids.copy_(torch.tensor([window.tokens], dtype=torch.long))
        entry.slots.copy_(torch.arange(window.pos, window.pos + rows, dtype=torch.int64))
        window.slots = entry.slots
        run = lambda: window_forward(e.model, [window], e.kernels.linear, e._dtype(), ids=entry.ids,  # noqa: E731
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
                with torch.cuda.graph(graph, pool=self.pool):
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
