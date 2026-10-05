"""ROCm lane rounds: a prompt prefills a step a round, then every live stream's window is verified in one forward."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import partial

import torch

from tensorfold.cuda.streams import Stream, accept, next_fill
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.engine.grammar import GrammarError
from tensorfold.rocm.model.forward import _project
from tensorfold.rocm.model.window import Window, commit, window_forward
from tensorfold.rocm.serving.graphs import WindowGraphs

CONFIDENCE = 0.3     # a draft chain ends after a draft the MTP head gives less than this
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
    """The ``Scheduler``'s decoder over a ``QwenEngine``; under tp, rank 0 decides and the other ranks follow."""

    def __init__(self, engine) -> None:
        self.e = engine
        self.streams: dict[int, Stream] = {}         # decoding
        self.filling: list[Stream] = []              # admitted, prompts still prefilling (oldest first)
        self.admitted: list[Stream] = []             # admitted since the last round (tp: sent with the next)
        self.next_id = 0
        self.graphs = WindowGraphs(engine)           # a lone stream's windows replay a captured graph
        self.asleep = True                           # tp: the other ranks wait on the store until a round
        self.wakes = 0

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
        stops = tuple(e._cuts(s.prompt, s.cached)) if s.draft else ()
        self._open(s, depth, stops, hit[1] if hit is not None else None)
        self.admitted.append(s)

    def _open(self, s: Stream, depth: int, stops: tuple, held) -> None:
        from tensorfold.rocm.serving.engine import clone_caches

        s.st = Lane(clone_caches(held) if held is not None else None, s.cached,
                    len(s.prompt) + s.count + depth + 1, depth, stops=stops)
        s.sid = self.next_id
        self.next_id += 1
        self.filling.append(s)

    # --- what every rank runs, in the same order ---------------------------------------------------------------

    def _span(self, s: Stream, stop: int):
        lane = s.st
        hidden, lane.caches = self.e._span(list(s.prompt[lane.pos:stop]), lane.caches, lane.pos, lane.total)
        lane.pos = stop
        if stop in lane.stops:
            self.e._remember(s.prompt[:stop], lane.caches)
        return hidden

    def _logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Every row's full-vocabulary logits (tp: the ranks' slices joined)."""

        e = self.e
        local = _project(hidden, e.model.output_head(), e.kernels.linear)
        if e.tp == 1:
            return local
        from tensorfold.rocm.model.qwen_tp import vocab_gather

        return vocab_gather(e.rccl, local.reshape(-1, local.shape[-1]))

    def _draft(self, s: Stream, depth: int, share) -> list[int]:
        e, lane = self.e, s.st
        dtype = e._dtype()
        cache = e.mtp.fresh_cache(batch=1, total=lane.pos + depth + 1, device=e._device(), dtype=dtype)
        # A greedy request drafts greedily; verification keeps the request's own sampling.
        return e.mtp.draft_chain(lane.hidden, s.out[-1], lane.pos, depth, cache,
                                 sampling=s.sampling or Sampling(seed=0, temperature=0.0), dtype=dtype,
                                 confidence=CONFIDENCE, share=share)

    def _verify(self, live: list[Stream]) -> tuple[list[Window], torch.Tensor, torch.Tensor]:
        e = self.e
        windows = [Window([s.out[-1]] + s.drafts, s.st.caches, s.st.pos) for s in live]
        if len(live) == 1 and e.graphs and e.tp == 1:
            hidden = self.graphs.forward(live[0].sid, windows[0])
        else:
            hidden = window_forward(e.model, windows, e.kernels.linear, e._dtype(), reduce=self._reduce())
        return windows, hidden, self._logits(hidden[0])

    def _reduce(self):
        """tp shares summed as one row sums alone: one add at two ranks, rank order past two."""

        e = self.e
        if e.tp == 1:
            return None
        from tensorfold.rocm.model.qwen_tp import all_reduce_local, ordered_sum

        return partial(ordered_sum if e.tp > 2 else all_reduce_local, e.rccl)

    def _keep(self, s: Stream, w: Window, hidden: torch.Tensor, start: int, rows: int) -> None:
        commit(self.e.model, w, rows)
        s.st.pos += rows
        s.st.hidden = hidden[:, start + rows - 1:start + rows].clone()   # a replay rewrites hidden

    # --- rank 0 ------------------------------------------------------------------------------------------------

    @torch.inference_mode()
    def round(self) -> list[Stream]:
        """A prefill step for the next queued prompt, then one forward over every decoding stream's window."""

        e = self.e
        tp = e.tp > 1
        if tp and self.asleep:
            e.rccl.store.set(f"tf_lanes/{self.wakes}", "go")
            self.wakes += 1
            self.asleep = False
        fill = next_fill(self.filling) if self.filling else None
        stop = 0
        if fill is not None:
            lane = fill.st
            stop = next((p for p in lane.stops if p > lane.pos), len(fill.prompt))
            if fill.background or any(not x.done for x in self.streams.values()):
                stop = min(stop, lane.pos + STEP)
        if tp:
            self._send_admitted(fill, stop)
        self.admitted = []
        done, first = self._fill(fill, stop) if fill is not None else ([], -1)
        live = [s for s in self.streams.values() if not s.done]
        depths = [(s, max(0, min(s.st.depth, s.count - len(s.out) - 1))) for s in live]
        if tp:
            self._share([first, len(depths)] + [v for s, d in depths for v in (s.sid, d)])
        grammars = {}
        share = self._share if tp else None
        for s, depth in depths:
            s.drafts = self._draft(s, depth, share) if depth > 0 else []
            if s.constraint is not None:
                try:
                    window = s.constraint.window([s.out[-1]] + s.drafts, list(range(-1, len(s.drafts))))
                except GrammarError as exc:           # this request ends with its error, the others go on
                    s.error, s.done = exc, True
                    continue
                s.drafts, grammars[s.sid] = window.tokens[1:], window
        done += [s for s in live if s.done]
        live = [s for s in live if not s.done]
        if tp:
            self._share([len(live)] + [v for s in live for v in (s.sid, len(s.drafts), *s.drafts)])
        if not live:
            if tp:
                self._close_round([])
            return done
        windows, hidden, logits = self._verify(live)
        report, start = [], 0
        for s, w in zip(live, windows):
            rows = len(w.tokens)
            positions = [s.st.pos + 1 + i for i in range(rows)]
            try:
                sampled = e._draw_rows(logits[start:start + rows], positions, s.sampling, s.constraint,
                                       grammars.get(s.sid))
                path, end = accept(w.tokens, list(range(-1, rows - 1)), sampled, s.count - len(s.out),
                                   self._ends(s))
                new = [w.tokens[r] for r in path[1:]] + [end]
                if s.constraint is not None:
                    s.constraint.advance(new)
            except GrammarError as exc:
                s.error, s.done = exc, True
                report.append((s.sid, 0, []))
                start += rows
                continue
            self._keep(s, w, hidden, start, len(path))
            s.committed.extend(w.tokens[r] for r in path)
            s.counted(rows)
            s.take(new, self._ends(s))
            report.append((s.sid, len(path), new))
            start += rows
        if tp:
            self._close_round(report)
        return done + [s for s in live if s.done]

    def _fill(self, s: Stream, stop: int) -> tuple[list[Stream], int]:
        """Prefill ``s`` to ``stop``; at its prompt's end, sample the first token and start decoding it."""

        t0 = time.perf_counter()
        try:
            hidden = self._span(s, stop)
            if stop < len(s.prompt):
                return [], -1
            logits = self._logits(hidden[:, -1])
            first = self.e._draw_rows(logits, [len(s.prompt)], s.sampling, s.constraint, None)[0]
        except Exception as exc:                      # noqa: BLE001  (this request fails, the others go on)
            self.filling = [x for x in self.filling if x is not s]
            s.error, s.done = exc, True
            return [s], -1
        finally:
            s.prefill_s += time.perf_counter() - t0
        self._started(s, hidden)
        if s.constraint is not None:
            try:
                s.constraint.advance([first])
            except GrammarError as exc:
                s.error, s.done = exc, True
                return [s], first
        s.take([first], self._ends(s))
        return ([s] if s.done else []), first

    def _started(self, s: Stream, hidden: torch.Tensor) -> None:
        s.st.hidden = hidden[:, -1:]
        s.context = list(s.prompt)
        s.started = time.perf_counter()
        self.filling = [x for x in self.filling if x is not s]
        self.streams[s.sid] = s

    def _send_admitted(self, fill: Stream | None, stop: int) -> None:
        values = [len(self.admitted)]
        for s in self.admitted:
            values += [s.sid, int(s.draft), s.st.depth, s.count, s.cached, len(s.st.stops), *s.st.stops,
                       len(s.prompt), *s.prompt]
        values += [fill.sid if fill is not None else -1, stop]
        self._share(values)

    def _close_round(self, report: list[tuple[int, int, list[int]]]) -> None:
        """Every stream's kept rows and new tokens, and whether the other ranks may sleep after this round."""

        values = [len(report)]
        for sid, rows, new in report:
            s = self.streams[sid]
            values += [sid, rows, int(s.done), len(new), *new]
        idle = not any(not s.done for s in self.streams.values()) and not self.filling
        self._share(values + [int(idle)])
        self.asleep = idle

    def _share(self, values: list[int]) -> list[int]:
        """Rank 0's ``values`` on every rank: their count, then the values, over RCCL."""

        e = self.e
        device = e._device()
        count = torch.tensor([len(values) if e.rank == 0 else 0], dtype=torch.int64, device=device)
        e.rccl.broadcast(count, count, root=0)
        n = int(count.item())
        if n == 0:
            return []
        buf = (torch.tensor(values, dtype=torch.int64, device=device) if e.rank == 0
               else torch.empty(n, dtype=torch.int64, device=device))
        e.rccl.broadcast(buf, buf, root=0)
        return [int(v) for v in buf.tolist()]

    def close(self) -> None:
        """Rank 0: the other ranks stop following."""

        if self.e.tp > 1 and self.e.rank == 0:
            self.e.rccl.store.set(f"tf_lanes/{self.wakes}", "stop")
            self.wakes += 1

    # --- the other ranks ---------------------------------------------------------------------------------------

    def follow(self) -> None:
        """Ranks above 0: mirror rank 0's rounds, sleeping on the store between them, until rank 0 stops."""

        e = self.e
        while True:
            key = f"tf_lanes/{self.wakes}"
            try:
                e.rccl.store.wait([key])
            except Exception as exc:  # noqa: BLE001 - the store's wait timeout: rank 0 is idle
                text = str(exc).lower()
                if "timeout" in text:
                    continue
                if "recv" in text or "connection" in text or "broken pipe" in text:
                    return                                  # rank 0 closed the store: the server stopped
                raise
            self.wakes += 1
            if e.rccl.store.get(key).decode() == "stop":
                return
            with torch.inference_mode():
                while not self._mirror():
                    pass

    def _mirror(self) -> bool:
        """One of rank 0's rounds; True when rank 0 says the ranks may sleep."""

        e = self.e
        values = self._share([])
        at = 1
        for _ in range(values[0]):
            sid, draft, depth, count, cached, n_stops = values[at:at + 6]
            stops = tuple(values[at + 6:at + 6 + n_stops])
            at += 6 + n_stops
            n_prompt = values[at]
            prompt = values[at + 1:at + 1 + n_prompt]
            at += 1 + n_prompt
            s = Stream(prompt, count, None, draft=bool(draft))
            s.cached = cached
            hit = e.cache.named(prompt, cached) if cached else None
            self._open(s, depth, stops, hit[1] if hit is not None else None)
            s.sid = sid
            self.next_id = sid + 1
        fill_sid, stop = values[at], values[at + 1]
        filled = None
        if fill_sid >= 0:
            s = next(x for x in self.filling if x.sid == fill_sid)
            hidden = self._span(s, stop)
            if stop == len(s.prompt):
                self._logits(hidden[:, -1])
                filled = (s, hidden)
        plan = self._share([])
        if filled is not None:
            s, hidden = filled
            self._started(s, hidden)
            s.out.append(plan[0])
        for i in range(plan[1]):
            s = self.streams[plan[2 + 2 * i]]
            depth = plan[3 + 2 * i]
            s.drafts = self._draft(s, depth, self._share) if depth > 0 else []
        final = self._share([])
        live, at = [], 1
        for _ in range(final[0]):
            sid, k = final[at], final[at + 1]
            s = self.streams[sid]
            s.drafts = final[at + 2:at + 2 + k]
            live.append(s)
            at += 2 + k
        # A stream rank 0 ended outside a window (at its first token, or on a grammar error) ends here too.
        self.streams = {s.sid: s for s in live}
        report = []
        if live:
            windows, hidden, _ = self._verify(live)
            report = self._share([])
            at, start = 1, 0
            by_sid = {s.sid: (s, w) for s, w in zip(live, windows)}
            for s in live:
                sid, rows, finished, n_new = report[at:at + 4]
                new = report[at + 4:at + 4 + n_new]
                at += 4 + n_new
                s, w = by_sid[sid]
                if rows:
                    self._keep(s, w, hidden, start, rows)
                s.out.extend(new)
                if finished:
                    self.streams.pop(sid, None)
                start += len(w.tokens)
            return bool(report[at])
        report = self._share([])
        return bool(report[-1])

    def _ends(self, s: Stream) -> tuple[int, ...]:
        return self.e.eos if s.stop_eos else ()

    def finish(self, done: list[Stream]) -> None:
        for s in done:
            self.streams.pop(s.sid, None)
            if s.sid == self.graphs.sid:
                self.graphs.reset()

    def drop(self) -> list[Stream]:
        """After an error in a round: forget the live streams and the queued prompts."""

        live = [s for s in self.streams.values() if not s.done] + self.filling
        self.streams, self.filling = {}, []
        self.graphs.reset()
        return live


__all__ = ["Lane", "Lanes", "STEP"]
