"""ROCm lane rounds: a prompt prefills a step a round, then every live stream's window is verified in one forward."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from tensorfold.cuda.streams import Stream, accept, next_fill
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.engine.grammar import GrammarError
from tensorfold.rocm.model.forward import _project
from tensorfold.rocm.model.window import Window, commit, window_forward

STEP = 1024          # prompt rows a prefill step takes while other streams decode


@dataclass
class Lane:
    """A stream's model state: its caches, the slot its next token goes to, and the hidden row that chose it."""

    caches: list | None
    pos: int
    total: int
    depth: int
    hidden: torch.Tensor | None = None
    stops: tuple = ()


class Lanes:
    """The ``Scheduler``'s decoder over a ``QwenEngine``: ``admit``, ``round``, ``finish``, ``drop``, ``live``."""

    def __init__(self, engine) -> None:
        self.e = engine
        self.streams: dict[int, Stream] = {}         # decoding
        self.filling: list[Stream] = []              # admitted, prompts still prefilling (oldest first)
        self.next_id = 0

    def live(self) -> int:
        return len(self.streams) + len(self.filling)

    def admit(self, s: Stream) -> None:
        """Queue a request on the longest kept prefix of its prompt; rounds prefill the rest."""

        e = self.e
        window = e.context_window
        if window is not None and len(s.prompt) >= window:
            raise ValueError(f"prompt of {len(s.prompt)} tokens exceeds the {window}-token window")
        if window is not None:
            s.count = max(1, min(s.count, window - len(s.prompt)))
        depth = e.mtp_depth if s.draft and e.mtp is not None and not e.no_drafts else 0
        hit = e.cache.longest(s.prompt) if s.draft else None
        s.cached = len(hit[0]) if hit is not None else 0
        from tensorfold.rocm.serving.engine import clone_caches

        held = clone_caches(hit[1]) if hit is not None else None
        stops = tuple(e._cuts(s.prompt, s.cached)) if s.draft else ()
        s.st = Lane(held, s.cached, len(s.prompt) + s.count + depth + 1, depth, stops=stops)
        s.sid = self.next_id
        self.next_id += 1
        self.filling.append(s)

    def _fill(self) -> list[Stream]:
        """Prefill the oldest queued prompt a step: to its next kept state, or STEP rows while others decode."""

        s = next_fill(self.filling)
        lane, n = s.st, len(s.prompt)
        stop = next((p for p in lane.stops if p > lane.pos), n)
        if s.background or any(not x.done for x in self.streams.values()):
            stop = min(stop, lane.pos + STEP)
        t0 = time.perf_counter()
        try:
            hidden, lane.caches = self.e._span(list(s.prompt[lane.pos:stop]), lane.caches, lane.pos, lane.total)
            lane.pos = stop
            if stop in lane.stops:
                self.e._remember(s.prompt[:stop], lane.caches)
            if stop < n:
                return []
            first = self._first(s, hidden)
        except Exception as exc:                      # noqa: BLE001  (this request fails, the others go on)
            self.filling = [x for x in self.filling if x is not s]
            s.error, s.done = exc, True
            return [s]
        finally:
            s.prefill_s += time.perf_counter() - t0
        lane.hidden = hidden[:, -1:]
        s.context = list(s.prompt)
        s.started = time.perf_counter()
        self.filling = [x for x in self.filling if x is not s]
        self.streams[s.sid] = s
        if s.constraint is not None:
            try:
                s.constraint.advance([first])
            except GrammarError as exc:
                s.error, s.done = exc, True
                return [s]
        s.take([first], self._ends(s))
        return [s] if s.done else []

    def _first(self, s: Stream, hidden: torch.Tensor) -> int:
        logits = _project(hidden[:, -1], self.e.model.output_head(), self.e.kernels.linear)
        return self.e._draw_rows(logits, [len(s.prompt)], s.sampling, s.constraint, None)[0]

    def _propose(self, s: Stream) -> list[int]:
        """The stream's MTP drafts for its next window (none for a serial stream), within its remaining tokens."""

        lane = s.st
        room = s.count - len(s.out)
        depth = min(lane.depth, room - 1)
        if depth <= 0:
            return []
        e = self.e
        dtype = e._dtype()
        cache = e.mtp.fresh_cache(batch=1, total=lane.pos + depth + 1, device=e._device(), dtype=dtype)
        # A greedy request drafts greedily; verification keeps the request's own sampling.
        return e.mtp.draft_chain(lane.hidden, s.out[-1], lane.pos, depth, cache,
                                 sampling=s.sampling or Sampling(seed=0, temperature=0.0), dtype=dtype)

    @torch.no_grad()
    def round(self) -> list[Stream]:
        """A prefill step for the next queued prompt, then one forward over every decoding stream's window."""

        done = self._fill() if self.filling else []
        live = [s for s in self.streams.values() if not s.done]
        if not live:
            return done
        grammars = {}
        for s in live:
            s.drafts = self._propose(s)
            if s.constraint is not None:
                try:
                    window = s.constraint.window([s.out[-1]] + s.drafts, list(range(-1, len(s.drafts))))
                except GrammarError as exc:           # this request ends with its error, the others go on
                    s.error, s.done = exc, True
                    continue
                s.drafts, grammars[s.sid] = window.tokens[1:], window
        done += [s for s in live if s.done]
        live = [s for s in live if not s.done]
        if not live:
            return done
        e = self.e
        windows = [Window([s.out[-1]] + s.drafts, s.st.caches, s.st.pos) for s in live]
        with torch.inference_mode():
            hidden = window_forward(e.model, windows, e.kernels.linear, e._dtype())
            logits = _project(hidden[0], e.model.output_head(), e.kernels.linear)
        start = 0
        for s, w in zip(live, windows):
            rows = len(w.tokens)
            positions = [s.st.pos + 1 + i for i in range(rows)]
            try:
                sampled = e._draw_rows(logits[start:start + rows], positions, s.sampling, s.constraint,
                                       grammars.get(s.sid))
            except GrammarError as exc:
                s.error, s.done = exc, True
                start += rows
                continue
            path, end = accept(w.tokens, list(range(-1, rows - 1)), sampled, s.count - len(s.out), self._ends(s))
            commit(e.model, w, len(path))
            s.st.pos += len(path)
            s.st.hidden = hidden[:, start + path[-1]:start + path[-1] + 1]
            new = [w.tokens[r] for r in path[1:]] + [end]
            s.committed.extend(w.tokens[r] for r in path)
            s.counted(rows)
            if s.constraint is not None:
                try:
                    s.constraint.advance(new)
                except GrammarError as exc:
                    s.error, s.done = exc, True
                    start += rows
                    continue
            s.take(new, self._ends(s))
            start += rows
        return done + [s for s in live if s.done]

    def _ends(self, s: Stream) -> tuple[int, ...]:
        return self.e.eos if s.stop_eos else ()

    def finish(self, done: list[Stream]) -> None:
        for s in done:
            self.streams.pop(s.sid, None)

    def drop(self) -> list[Stream]:
        """After an error in a round: forget the live streams and the queued prompts."""

        live = [s for s in self.streams.values() if not s.done] + self.filling
        self.streams, self.filling = {}, []
        return live


__all__ = ["Lane", "Lanes", "STEP"]
