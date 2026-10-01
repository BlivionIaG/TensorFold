"""ROCm Qwen engine for the existing torch server.

``generate(prompt, max_tokens, sampling, on_tokens)`` returns ``{"cached": n}``. A prompt that
strictly extends a kept prefix prefills only the suffix. Entries are the state after a message
start and one token before the prompt ends, each produced by the same forward a fresh prefill of
those ids uses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose
from tensorfold.rocm.prefix import PrefixCache, entry_end, trim_bytes
from tensorfold.rocm.qwen import Engine as Kernels
from tensorfold.rocm.qwen import activation_dtype, load
from tensorfold.rocm.qwen_math import _blank_caches, _project, forward_hidden

# Qwen's end token when the checkpoint does not name one.
_DEFAULT_EOS = (151645,)


def read_eos(model_dir: Path) -> tuple[int, ...]:
    path = Path(model_dir) / "config.json"
    if not path.is_file():
        return _DEFAULT_EOS
    config = json.loads(path.read_text())
    text = config.get("text_config", config)
    raw = text.get("eos_token_id", config.get("eos_token_id"))
    if raw is None:
        return _DEFAULT_EOS
    if isinstance(raw, int):
        return (raw,)
    return tuple(int(token) for token in raw)


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
                 points: Callable[[Sequence[int]], list[int]] | None = None):
        self.model = model
        self.kernels = kernels
        self.eos = tuple(int(token) for token in eos)
        self.cache = PrefixCache(max(0, int(keep)))
        self.context_window = context
        self.byte_budget = byte_budget
        self.points = points

    @classmethod
    def load(cls, model_dir: Path | str, *, schedule: str = "auto", keep: int = 8,
             context: int | None = None, byte_budget: int | None = None) -> QwenEngine:
        path = Path(model_dir)
        model = load(path)
        from tensorfold.rocm.prefix import message_points

        return cls(model, Kernels(model, schedule=schedule), read_eos(path), keep=keep, context=context,
                   byte_budget=byte_budget, points=message_points(path))

    def _dtype(self) -> torch.dtype:
        if self.kernels.dtype is None:
            from tensorfold.rocm.build import gfx_name

            self.kernels.dtype = activation_dtype(gfx_name())
        return self.kernels.dtype

    def _device(self) -> torch.device:
        return self.model.embed.words.device

    def _forward(self, tokens: Sequence[int], caches: list[dict] | None, pos0: int) -> tuple[torch.Tensor, list[dict]]:
        device, dtype = self._device(), self._dtype()
        if caches is None:
            caches = _blank_caches(self.model, 1, len(tokens), device, dtype)
        ids = torch.tensor([list(tokens)], dtype=torch.long, device=device)
        with torch.inference_mode():
            # Prefill and the one-token step share the prefill conv and rope, so a split matches one forward.
            return forward_hidden(self.model, ids, caches, self.kernels.linear, pos0, dtype, exact_short=True)

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

    def _sample(self, hidden: torch.Tensor, sampling: Sampling | None, position: int, constraint) -> int:
        logits = _project(hidden[:, -1], self.model.output_head(), self.kernels.linear)
        if constraint is not None:
            constraint.mask(logits)
        row = logits.detach().float().reshape(-1)
        if sampling is None or float(sampling.temperature) <= 0.0:
            token = int(torch.argmax(row).item())
        else:
            width = int(row.shape[0])
            k = int(sampling.top_k)
            if k:
                count = min(width, k + MARGIN)
                values, index = torch.topk(row, count, sorted=False)
                token = choose(values.cpu().numpy(), index.cpu().numpy().astype(np.int64), position, sampling)
            else:
                token = choose(row.cpu().numpy(), np.arange(width, dtype=np.int64), position, sampling)
        if constraint is not None:
            constraint.advance([token])
        return token

    def generate(self, prompt: list[int], max_tokens: int, sampling: Sampling | None,
                 on_tokens: Callable[[list[int]], bool | None], *, stop_eos: bool = True, draft: bool = True,
                 constraint=None, vision=None, background: bool = False) -> dict[str, int]:
        """One prompt. ``draft=False`` ignores the prefix cache. ``stats['cached']`` is the reused length.

        ``constraint`` is the server's grammar. ``vision`` is refused. ``background`` is accepted and
        unused: this engine serves one request at a time, and the server orders those requests.
        """

        del background
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
        hit = self.cache.longest(prompt) if draft else None
        cached = len(hit[0]) if hit is not None else 0
        held = clone_caches(hit[1]) if hit is not None else None
        hidden, caches = self._prefill(prompt, held, cached, len(prompt) + room, store=draft)
        ends = set(self.eos)
        position = len(prompt)
        nxt = self._sample(hidden, sampling, position, constraint)
        for step in range(room):
            stop = bool(on_tokens([nxt])) if on_tokens is not None else False
            ended = stop_eos and nxt in ends
            grammar_done = constraint is not None and bool(getattr(constraint, "finished", False))
            if stop or ended or grammar_done or step + 1 == room:
                break
            hidden, caches = self._forward([nxt], caches, position)
            position += 1
            nxt = self._sample(hidden, sampling, position, constraint)
        return {"cached": cached}

    def prefill_caches(self, prompt: Sequence[int]) -> list[dict]:
        """Caches after one forward of ``prompt``. A stored prefix of that length matches this."""

        _, caches = self._span(list(prompt), None, 0, len(prompt))
        return caches


def caches_equal(left: list[dict], right: list[dict]) -> bool:
    return _caches_equal(left, right)
