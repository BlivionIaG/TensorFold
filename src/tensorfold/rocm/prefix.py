"""Host-side prompt cache and the stop rule for one ROCm request.

The cache is the CUDA ``PrefixCache``: a hit must leave at least one token to prefill,
and an entry that a later prompt resumed from is evicted last. ``entry_end`` is one
token before the prompt ends, because a following turn renders a different newline there.
"""

from __future__ import annotations

from typing import Callable, Sequence

from tensorfold.cuda.streams import PrefixCache

__all__ = ["PrefixCache", "decode_ids", "entry_end", "request_parents", "trim_bytes"]


def entry_end(prompt: Sequence[int]) -> int:
    """Where a prompt's cache entry ends: one token early."""

    return max(1, len(prompt) - 1)


def message_points(model_dir) -> Callable[[Sequence[int]], list[int]] | None:
    """Assistant-header cuts. A short chat keeps them: its think block is not a prefix of the next turn."""

    from pathlib import Path

    import numpy as np

    path = Path(model_dir)
    if not (path / "tokenizer.json").is_file():
        return None
    try:
        from tokenizers import Tokenizer

        from tensorfold.cuda.chat_template import ChatTemplate
        from tensorfold.cuda.markers import TemplateTokens
        from tensorfold.engine.prefill_plan import PrefillPlan, message_markers

        tokens = TemplateTokens(Tokenizer.from_file(str(path / "tokenizer.json")), ChatTemplate(path))
        openers, assistant = message_markers(tokens)
    except (OSError, ValueError, KeyError):
        return None
    if not (openers or assistant):
        return None
    gap = max(1, len(assistant))
    plan = PrefillPlan(openers=openers, assistant=assistant, min_chunk=gap, step=max(2048, gap))

    def points(ids: Sequence[int]) -> list[int]:
        found = plan.points(np.asarray(ids, dtype=np.int64))
        return [int(point) for point in found if 0 < int(point) < len(ids)]

    return points


def decode_ids(prompt: Sequence[int], step: Callable[[Sequence[int]], int], max_tokens: int, eos: Sequence[int],
               *, stop_eos: bool = True, on_tokens: Callable[[list[int]], bool | None] | None = None) -> list[int]:
    """Sample until ``max_tokens``, an end token, or ``on_tokens`` asking to stop.

    ``step`` sees the prompt plus tokens already produced and returns the next id. Requests of
    different lengths each call this with their own step, so a pad token never enters a state.
    """

    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    ends = set(int(token) for token in eos)
    out: list[int] = []
    produced = list(prompt)
    for _ in range(int(max_tokens)):
        token = int(step(produced))
        out.append(token)
        produced.append(token)
        if on_tokens is not None and on_tokens([token]):
            break
        if stop_eos and token in ends:
            break
    return out


def request_parents(lengths: Sequence[int]) -> list[list[int]]:
    """One chain per request, the parent list ``plan_host`` schedules. A short request is not padded."""

    parents: list[list[int]] = []
    for raw in lengths:
        count = int(raw)
        if count < 1:
            raise ValueError("a request produces at least one token")
        parents.append([-1, *range(count - 1)])
    return parents


def trim_bytes(cache: PrefixCache, budget: int | None) -> None:
    """Evict until snap byte counts fit ``budget``. None leaves the slot cap in charge."""

    if budget is None:
        return
    while cache.entries and sum(int(entry[2] or 0) for entry in cache.entries) > budget:
        cold = [entry for entry in cache.entries[:-1] if tuple(entry[0]) not in cache.hit]
        gone = cold[0] if cold else cache.entries[0]
        cache.entries = [entry for entry in cache.entries if entry is not gone]
        cache.hit &= {tuple(entry[0]) for entry in cache.entries}
