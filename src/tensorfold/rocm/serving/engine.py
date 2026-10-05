"""ROCm Qwen engine for the torch server: lane rounds, prefix cache, MTP drafts, tp ranks."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose
from tensorfold.rocm.model.forward import _blank_caches, forward_hidden
from tensorfold.rocm.model.mtp import MTPEngine
from tensorfold.rocm.serving.prefix import PrefixCache, entry_end, trim_bytes
from tensorfold.rocm.model.qwen import Engine as Kernels
from tensorfold.rocm.model.qwen import activation_dtype, load, slice_for_tp
from tensorfold.rocm.model.qwen_tp import tp_forward_hidden

_DEFAULT_EOS = (151645,)
_DEFAULT_MTP_DEPTH = 3       # most MTP drafts a round, as the CUDA Qwen3.6 engine
_LONG_PROMPT = 8192            # past this many prompt tokens a request hands its freed memory back


def _resolve_p2p(gfx: str, p2p: bool | None) -> bool | None:
    """``--p2p`` / ``--no-p2p`` as given; on for a multi-die APU; otherwise ``None``: RCCL decides."""

    if p2p is not None:
        return bool(p2p)
    try:
        multi = bool(torch.cuda.get_device_properties(0).multi_gpu_capable)
        integrated = bool(torch.cuda.get_device_properties(0).is_integrated)
    except (AttributeError, AssertionError, RuntimeError):
        return None
    return True if multi and integrated else None


def draw(logits: torch.Tensor, sampling: Sampling | None, position: int) -> int:
    """One row's token: argmax for greedy, else the keyed draw at ``position`` over the top-k candidates."""

    row = logits.detach().float().reshape(-1)
    if sampling is None or float(sampling.temperature) <= 0.0:
        return int(torch.argmax(row).item())
    width = int(row.shape[0])
    k = int(sampling.top_k)
    if k:
        values, index = torch.topk(row, min(width, k + MARGIN), sorted=False)
        return choose(values.cpu().numpy(), index.cpu().numpy().astype(np.int64), position, sampling)
    return choose(row.cpu().numpy(), np.arange(width, dtype=np.int64), position, sampling)


def read_eos(model_dir: Path) -> tuple[int, ...]:
    """Every end id in config.json (top level and text_config) and generation_config.json."""

    found: list[int] = []

    def take(raw) -> None:
        for token in [raw] if isinstance(raw, int) else (raw or []):
            if int(token) not in found:
                found.append(int(token))

    root = Path(model_dir)
    for name in ("config.json", "generation_config.json"):
        path = root / name
        if path.is_file():
            config = json.loads(path.read_text())
            take(config.get("eos_token_id"))
            take(config.get("text_config", {}).get("eos_token_id"))
    return tuple(found) or _DEFAULT_EOS


def cache_bytes(caches: list[dict]) -> int:
    """Bytes of the tensors a prefix entry keeps."""

    total = 0
    for cache in caches:
        for key in ("k", "v", "conv", "state"):
            value = cache.get(key)
            if torch.is_tensor(value):
                total += int(value.numel()) * int(value.element_size())
    return total


def clone_caches(caches: list[dict]) -> list[dict]:
    """A detached copy. Key and value rows past ``len`` are the next reply's room, not the prefix."""

    cloned = []
    for cache in caches:
        if "k" in cache:
            used = int(cache["len"]) if "len" in cache else int(cache["k"].shape[2])
            cloned.append({
                "k": cache["k"][:, :, :used].detach().clone(),
                "v": cache["v"][:, :, :used].detach().clone(),
                "len": used,
            })
        else:
            conv, state = cache.get("conv"), cache.get("state")
            cloned.append({
                "conv": None if conv is None else conv.detach().clone(),
                "state": None if state is None else state.detach().clone(),
            })
    return cloned


def _grow(caches: list[dict], total: int, dtype: torch.dtype, device: torch.device) -> list[dict]:
    """Copy a prefix into buffers long enough for ``total`` positions."""

    grown = []
    for cache in caches:
        if "k" not in cache:
            grown.append(cache)
            continue
        batch, heads, _, dim = cache["k"].shape
        used = int(cache["len"]) if "len" in cache else int(cache["k"].shape[2])
        hold = max(total, used)
        key = torch.empty(batch, heads, hold, dim, dtype=dtype, device=device)
        value = torch.empty_like(key)
        if used:
            key[:, :, :used] = cache["k"][:, :, :used]
            value[:, :, :used] = cache["v"][:, :, :used]
        grown.append({"k": key, "v": value, "len": used})
    return grown


def _caches_equal(left: list[dict], right: list[dict]) -> bool:
    for one, two in zip(left, right, strict=True):
        if "k" in one:
            used = int(one["len"]) if "len" in one else int(one["k"].shape[2])
            other = int(two["len"]) if "len" in two else int(two["k"].shape[2])
            if used != other:
                return False
            if used and (not torch.equal(one["k"][:, :, :used], two["k"][:, :, :used])
                         or not torch.equal(one["v"][:, :, :used], two["v"][:, :, :used])):
                return False
        else:
            for key in ("conv", "state"):
                first, second = one.get(key), two.get(key)
                if first is None or second is None:
                    if first is not None or second is not None:
                        return False
                elif not torch.equal(first, second):
                    return False
    return True


class QwenEngine:
    """One ROCm model. Sampling is the shared keyed draw. Tool calls are the server's gate over ``generate``."""

    exact_sampling = True
    concurrent = False
    call_gate = None

    def __init__(self, model, kernels: Kernels, eos: Sequence[int], *, keep: int = 8,
                 context: int | None = None, byte_budget: int | None = None,
                 points: Callable[[Sequence[int]], list[int]] | None = None,
                 tp: int = 1, rank: int = 0, rccl: Any = None, no_drafts: bool = False,
                 mtp_depth: int = _DEFAULT_MTP_DEPTH, streams: int = 1):
        self.model = model
        self.kernels = kernels
        self.eos = tuple(int(token) for token in eos)
        self.cache = PrefixCache(max(0, int(keep)))
        self.context_window = context
        self.byte_budget = byte_budget
        self.points = points
        self.tp, self.rank, self.rccl, self.no_drafts = int(tp), int(rank), rccl, bool(no_drafts)
        head = getattr(model, "mtp", None)
        self.mtp = MTPEngine(model, head, linear=kernels.linear, rccl=rccl) if head is not None else None
        self.mtp_depth = int(mtp_depth)
        # A lone stream's windows replay a captured graph; TENSORFOLD_GRAPH=0 keeps them eager.
        self.graphs = os.environ.get("TENSORFOLD_GRAPH", "1") != "0"
        # Requests decode in lane rounds; ``streams`` > 1 takes that many at once (``--parallel``).
        self.streams = max(1, int(streams))
        self.concurrent = self.streams > 1
        self.scheduler = None
        self.lanes = None
        self._starting = threading.Lock()

    @classmethod
    def load(cls, model_dir: Path | str, *, schedule: str | None = None, keep: int = 8,
             context: int | None = None, context_explicit: bool = False, byte_budget: int | None = None,
             tp: int = 1, rank: int = 0, master: str = "", master_port: int = 29551,
             p2p: bool | None = None, no_drafts: bool = False,
             mtp_depth: int = _DEFAULT_MTP_DEPTH, streams: int = 1) -> QwenEngine:
        from tensorfold.rocm.kernels.build import gfx_name

        # TENSORFOLD_ROCM_SCHEDULE=wmma runs every projection on the gfx11 WMMA tiles (opt-in; auto is dot2).
        schedule = schedule or os.environ.get("TENSORFOLD_ROCM_SCHEDULE", "auto")
        if schedule not in ("auto", "gemv", "wmma"):
            raise ValueError(f"TENSORFOLD_ROCM_SCHEDULE is auto, gemv or wmma, not {schedule!r}")
        path = Path(model_dir)
        if tp == 1:
            if rank != 0:
                raise ValueError("rank must be 0 when tp=1")
            model = load(path)
            from tensorfold.rocm.serving.prefix import message_points

            engine = cls(model, Kernels(model, schedule=schedule), read_eos(path), keep=keep,
                         points=message_points(path), tp=1, rank=0, no_drafts=no_drafts, mtp_depth=mtp_depth,
                         streams=streams)
            return engine._planned(path, context, context_explicit, byte_budget)

        from tensorfold.rocm.serving.comm import RCCL

        # One GPU a rank: with every card visible, rank r takes card r; with one card per process, that card.
        torch.cuda.set_device(rank % torch.cuda.device_count())
        gfx = gfx_name()
        prefer_p2p = _resolve_p2p(gfx, p2p)
        rccl = RCCL(rank, tp, master, master_port, prefer_p2p=prefer_p2p)
        rccl.ready("startup")
        model = load(path)
        slice_for_tp(model, rank, tp)
        from tensorfold.rocm.serving.prefix import message_points

        engine = cls(model, Kernels(model, schedule=schedule), read_eos(path), keep=keep,
                     points=message_points(path), tp=tp, rank=rank, rccl=rccl, no_drafts=no_drafts,
                     mtp_depth=mtp_depth, streams=streams)
        return engine._planned(path, context, context_explicit, byte_budget)

    def _planned(self, path: Path, context: int | None, explicit: bool, byte_budget: int | None) -> QwenEngine:
        """Fit the context window and the prompt cache to this GPU's memory (every rank agrees)."""

        from tensorfold.rocm.serving import memory

        config = json.loads((path / "config.json").read_text())
        native = int((config.get("text_config") or config).get("max_position_embeddings") or 0)
        self.context_window, self.byte_budget = memory.plan(self, native, context, explicit, byte_budget, self.rccl)
        return self

    def warm(self) -> None:
        """One prefill span, nothing stored: every prefill kernel is built and its workspace allocated once."""

        from tensorfold.rocm.model.qwen_math import SPAN

        vocab = self.model.spec.vocab
        self._prefill([1 + index % (vocab - 1) for index in range(SPAN)], None, 0, SPAN + 1, store=False)

    def _dtype(self) -> torch.dtype:
        if self.kernels.dtype is None:
            from tensorfold.rocm.kernels.build import gfx_name

            self.kernels.dtype = activation_dtype(gfx_name())
        return self.kernels.dtype

    def _device(self) -> torch.device:
        return self.model.embed.words.device

    def _forward(self, tokens: Sequence[int], caches: list[dict] | None, pos0: int, *,
                 decode: bool = False) -> tuple[torch.Tensor, list[dict]]:
        """A prefill span, or with ``decode`` the one-token decode step's arithmetic, eager (a window row's reference)."""

        device, dtype = self._device(), self._dtype()
        if caches is None:
            caches = _blank_caches(self.model, 1, len(tokens), device, dtype)
        ids = torch.tensor([list(tokens)], dtype=torch.long, device=device)
        with torch.inference_mode():
            if self.tp > 1:
                return tp_forward_hidden(self.model, ids, caches, self.kernels.linear, pos0, self.rccl,
                                         act_dtype=dtype, exact_short=not decode)
            return forward_hidden(self.model, ids, caches, self.kernels.linear, pos0, dtype, exact_short=not decode)

    def _span(self, tokens: Sequence[int], caches: list[dict] | None, pos0: int, total: int,
              ) -> tuple[torch.Tensor, list[dict]]:
        """One forward of ``tokens`` starting at ``pos0``. Buffers grow to ``total`` positions."""

        device, dtype = self._device(), self._dtype()
        if caches is None:
            caches = _blank_caches(self.model, 1, total, device, dtype)
        else:
            caches = _grow(caches, total, dtype, device)
        return self._forward(tokens, caches, pos0)

    def _cuts(self, prompt: Sequence[int], cached: int) -> list[int]:
        """Positions past ``cached`` where this prefill keeps a state."""

        from tensorfold.cuda.markers import MIN_GAP

        stops = []
        if self.points is not None:
            stops.extend(int(point) for point in self.points(prompt) if cached < int(point) < len(prompt))
        stops.sort()
        end = entry_end(prompt)
        # A message start already next to the prompt end covers that boundary.
        near = bool(stops) and len(prompt) - stops[-1] < MIN_GAP
        if cached < end < len(prompt) and not near:
            stops.append(end)
        return sorted(set(stops))

    def _remember(self, ids: Sequence[int], caches: list[dict]) -> None:
        if self.cache.keep <= 0 or self.byte_budget == 0:
            return
        cloned = clone_caches(caches)
        self.cache.add(list(ids), cloned, cache_bytes(cloned))
        trim_bytes(self.cache, self.byte_budget)

    def _prefill(self, prompt: Sequence[int], caches: list[dict] | None, cached: int, total: int, *,
                 store: bool) -> tuple[torch.Tensor, list[dict]]:
        """Prefill ``prompt[cached:]``. Stored cuts split the forward so a resume repeats those launches."""

        cuts = self._cuts(prompt, cached) if store else []
        bounds = [cached, *cuts, len(prompt)]
        hidden = None
        for start, stop in zip(bounds, bounds[1:]):
            if start == stop:
                continue
            hidden, caches = self._span(list(prompt[start:stop]), caches, start, total)
            if store and stop in cuts:
                self._remember(prompt[:stop], caches)
        if hidden is None:
            raise RuntimeError("a prefill received no tokens")
        return hidden, caches

    def _draw_rows(self, logits: torch.Tensor, positions: list[int], sampling: Sampling | None, constraint,
                   window) -> list[int]:
        """Each row's token at its slot; a grammar masks the rows first (``window``: the rows' accepted prefixes)."""

        if constraint is not None:
            constraint.mask(logits, window)
        return [draw(logits[i:i + 1], sampling, position) for i, position in enumerate(positions)]

    def generate(self, prompt: list[int], max_tokens: int, sampling: Sampling | None,
                 on_tokens: Callable[[list[int]], bool | None], *, stop_eos: bool = True, draft: bool = True,
                 constraint=None, vision=None, background: bool = False) -> dict[str, int]:
        """One prompt; ``draft=False`` skips the prefix cache. Returns ``{'cached': reused length}``."""

        if vision is not None:
            raise ValueError("image inputs are not served on ROCm")
        if not prompt:
            raise ValueError("prompt is empty")
        if self.context_window is not None and len(prompt) >= self.context_window:
            raise ValueError(f"prompt of {len(prompt)} tokens exceeds the {self.context_window}-token window")
        room = int(max_tokens)
        if self.context_window is not None:
            room = min(room, self.context_window - len(prompt))
        if room < 1:
            raise ValueError("max_tokens must leave room for one token")
        with self._starting:                      # requests arriving together start one worker, not one each
            if self.scheduler is None:
                from tensorfold.cuda.scheduler import Scheduler
                from tensorfold.rocm.serving.lanes import Lanes

                self.lanes = Lanes(self)
                self.scheduler = Scheduler(self.lanes, max_streams=self.streams)

        def one_at_a_time(tokens: list[int]) -> bool:
            # A round's accepted drafts reach the caller one by one, so a stop inside them ends the reply there.
            return any(bool(on_tokens([token])) for token in tokens)

        stats = self.scheduler.submit(list(prompt), room, sampling, draft, one_at_a_time, stop_eos=stop_eos,
                                      constraint=constraint, background=background)
        if len(prompt) > _LONG_PROMPT:
            torch.cuda.empty_cache()              # a long prefill's freed blocks go back to the runtime's scratch
        return stats

    def close(self) -> None:
        """Stop the lane worker; under tp, rank 0 tells the other ranks there are no more rounds."""

        if self.scheduler is not None:
            self.scheduler.close()
            self.scheduler = None
        if self.tp > 1 and self.rank == 0:
            if self.lanes is None:
                from tensorfold.rocm.serving.lanes import Lanes

                self.lanes = Lanes(self)
            self.lanes.close()

    def follow(self) -> None:
        """Ranks above 0: run rank 0's lane rounds in step with it until rank 0 closes."""

        if self.rank == 0:
            raise RuntimeError("rank 0 serves requests; follow() is for the other ranks")
        from tensorfold.rocm.serving.lanes import Lanes

        self.lanes = Lanes(self)
        self.lanes.follow()

    def prefill_caches(self, prompt: Sequence[int]) -> list[dict]:
        """Caches after one forward of ``prompt``. A stored prefix of that length matches this."""

        _, caches = self._span(list(prompt), None, 0, len(prompt))
        return caches


def caches_equal(left: list[dict], right: list[dict]) -> bool:
    return _caches_equal(left, right)
