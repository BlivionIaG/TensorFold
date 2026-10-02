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
from tensorfold.rocm.qwen import activation_dtype, load, slice_for_tp
from tensorfold.rocm.qwen_math import _blank_caches, _project, forward_hidden
from tensorfold.rocm.qwen_tp import tp_forward_hidden, vocab_gather
from tensorfold.rocm.mtp import MTPEngine

_DEFAULT_EOS = (151645,)
_DEFAULT_MTP_DEPTH = 4


def _resolve_p2p(gfx: str, p2p: bool | None) -> bool:
    """ROCm P2P gating: opt-in only for RDNA 2/3/4 discrete; default-on for multi-mgpu-capable APUs.

    The user passes ``--p2p`` to opt in (or opt out). When unset, the default depends on whether the
    device is an integrated multi-mgpu APU. PCIe peer access is BIOS/ACS/driver-dependent; we don't
    autodetect — the CLI refuses with a name when ``--p2p`` is set and ``hipDeviceCanAccessPeer`` is 0.
    """
    if p2p is not None:
        return bool(p2p)
    try:
        multi = bool(torch.cuda.get_device_properties(0).multi_gpu_capable)
        integrated = bool(torch.cuda.get_device_properties(0).is_integrated)
        return multi and integrated
    except (AttributeError, AssertionError, RuntimeError):
        return False


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
                 points: Callable[[Sequence[int]], list[int]] | None = None,
                 tp: int = 1, rank: int = 0, rccl: Any = None, no_drafts: bool = False,
                 mtp_depth: int = _DEFAULT_MTP_DEPTH):
        self.model = model
        self.kernels = kernels
        self.eos = tuple(int(token) for token in eos)
        self.cache = PrefixCache(max(0, int(keep)))
        self.context_window = context
        self.byte_budget = byte_budget
        self.points = points
        self.tp, self.rank, self.rccl, self.no_drafts = int(tp), int(rank), rccl, bool(no_drafts)
        self._posted = 0          # requests rank 0 has published (tp > 1)
        head = getattr(model, "mtp", None)
        self.mtp = MTPEngine(model, head, linear=kernels.linear) if head is not None else None
        self.mtp_depth = int(mtp_depth)

    @classmethod
    def load(cls, model_dir: Path | str, *, schedule: str = "auto", keep: int = 8,
             context: int | None = None, byte_budget: int | None = None,
             tp: int = 1, rank: int = 0, master: str = "", master_port: int = 29551,
             p2p: bool | None = None, no_drafts: bool = False,
             mtp_depth: int = _DEFAULT_MTP_DEPTH) -> QwenEngine:
        from tensorfold.rocm.build import gfx_name

        path = Path(model_dir)
        if tp == 1:
            if rank != 0:
                raise ValueError("rank must be 0 when tp=1")
            model = load(path)
            from tensorfold.rocm.prefix import message_points

            return cls(model, Kernels(model, schedule=schedule), read_eos(path),
                       keep=keep, context=context, byte_budget=byte_budget,
                       points=message_points(path), tp=1, rank=0, no_drafts=no_drafts,
                       mtp_depth=mtp_depth)

        from tensorfold.rocm.comm import RCCL

        # One GPU a rank: with every card visible, rank r takes card r; with one card per process, that card.
        torch.cuda.set_device(rank % torch.cuda.device_count())
        gfx = gfx_name()
        prefer_p2p = _resolve_p2p(gfx, p2p)
        rccl = RCCL(rank, tp, master, master_port, prefer_p2p=prefer_p2p)
        rccl.ready("startup")
        model = load(path)
        slice_for_tp(model, rank, tp)
        from tensorfold.rocm.prefix import message_points

        return cls(model, Kernels(model, schedule=schedule), read_eos(path),
                   keep=keep, context=context, byte_budget=byte_budget,
                   points=message_points(path), tp=tp, rank=rank, rccl=rccl, no_drafts=no_drafts,
                   mtp_depth=mtp_depth)

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
            if self.tp > 1:
                return tp_forward_hidden(self.model, ids, caches, self.kernels.linear, pos0, self.rccl,
                                         act_dtype=dtype, exact_short=True)
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
        local = _project(hidden[:, -1], self.model.output_head(), self.kernels.linear)
        logits = vocab_gather(self.rccl, local) if self.tp > 1 else local     # every rank joins the gather
        if self.rank != 0:
            return self._share([0])[0]
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
        return self._share([token])[0]

    def _share(self, values: list[int]) -> list[int]:
        """Rank 0's ``values`` on every rank. Other ranks pass placeholders of the same length."""

        if self.tp <= 1:
            return values
        buffer = torch.tensor(values, dtype=torch.int64, device=self._device())
        self.rccl.broadcast(buffer, buffer, root=0)
        return [int(value) for value in buffer.tolist()]

    def generate(self, prompt: list[int], max_tokens: int, sampling: Sampling | None,
                 on_tokens: Callable[[list[int]], bool | None], *, stop_eos: bool = True, draft: bool = True,
                 constraint=None, vision=None, background: bool = False) -> dict[str, int]:
        """One prompt. ``draft=False`` ignores the prefix cache. ``stats['cached']`` is the reused length.

        ``constraint`` is the server's grammar. ``vision`` is refused. ``background`` is accepted and
        unused: this engine serves one request at a time, and the server orders those requests. With
        ``tp > 1`` this runs on rank 0, and every other rank runs the same request in :meth:`follow`.
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
        if self.tp > 1:
            self._post(list(prompt), room, draft)
        return self._run(list(prompt), room, sampling, on_tokens, stop_eos, draft, constraint)

    def _run(self, prompt: list[int], room: int, sampling: Sampling | None, on_tokens, stop_eos: bool,
             draft: bool, constraint) -> dict[str, int]:
        hit = self.cache.longest(prompt) if draft else None
        cached = len(hit[0]) if hit is not None else 0
        held = clone_caches(hit[1]) if hit is not None else None
        hidden, caches = self._prefill(prompt, held, cached, len(prompt) + room, store=draft)
        ends = set(self.eos)
        position = len(prompt)
        nxt = self._sample(hidden, sampling, position, constraint)
        depth = self.mtp_depth if self.mtp is not None and not self.no_drafts else 0
        for step in range(room):
            done = step + 1 == room
            if self.rank == 0 and not done:
                stop = bool(on_tokens([nxt])) if on_tokens is not None else False
                ended = stop_eos and nxt in ends
                grammar_done = constraint is not None and bool(getattr(constraint, "finished", False))
                done = stop or ended or grammar_done
            elif self.rank == 0 and on_tokens is not None:
                on_tokens([nxt])
            if self.tp > 1:
                done = bool(self._share([int(done)])[0])
            if done:
                break
            emitted, hidden, caches, position, nxt = self._decode_step(
                hidden, caches, position, nxt,
                sampling=sampling, constraint=constraint, depth=depth,
            )
            for tok in emitted[1:]:
                if self.rank == 0 and not done:
                    stop = bool(on_tokens([tok])) if on_tokens is not None else False
                    ended = stop_eos and tok in ends
                    grammar_done = constraint is not None and bool(getattr(constraint, "finished", False))
                    done = stop or ended or grammar_done
                    if self.tp > 1:
                        done = bool(self._share([int(done)])[0])
                    if done:
                        break
        return {"cached": cached}

    def _decode_step(self, hidden: torch.Tensor, caches: list[dict], position: int, last_token: int, *,
                     sampling: Sampling | None, constraint, depth: int) -> tuple[list[int], torch.Tensor, list[dict], int, int]:
        """One decode round. Entry: ``hidden`` at ``position - 1``, ``last_token`` at ``position - 1``.

        Returns ``(emitted, hidden, caches, position, nxt)``. ``emitted[0]`` is ``last_token`` so
        ``_run`` can skip the pre-step emit; later entries are the drafter's accepted tokens plus
        the verifier's final sample on full-chain acceptance. With ``depth <= 0`` or no MTP head
        installed this is one forward + one sample, the same shape as the serial path.
        """
        emitted = [last_token]
        if self.mtp is None or depth <= 0:
            hidden, caches = self._forward([last_token], caches, position)
            position += 1
            nxt = self._sample(hidden, sampling, position, constraint)
            return emitted + [nxt], hidden, caches, position, nxt

        dtype = self._dtype()
        hidden, caches = self._forward([last_token], caches, position)
        position += 1
        mtp_state = self.mtp.fresh_cache(batch=1,
                                         total=position + max(0, depth) + 1,
                                         device=self._device(), dtype=dtype)
        drafts = self.mtp.draft_chain(hidden[:, -1:], last_token, position - 1, depth, mtp_state,
                                      sampling=sampling, dtype=dtype)
        cur_hidden = hidden
        for i, d in enumerate(drafts):
            sample_key = position + i - 1
            nxt_main = self._sample(cur_hidden, sampling, sample_key, constraint)
            advance = nxt_main if nxt_main != d else d
            cur_hidden, caches = self._forward([advance], caches, position + i)
            if nxt_main != d:
                return emitted + [nxt_main], cur_hidden, caches, position + i + 1, nxt_main
            emitted.append(d)
        nxt = self._sample(cur_hidden, sampling, position + depth - 1, constraint)
        emitted.append(nxt)
        return emitted, cur_hidden, caches, position + depth, nxt

    def _post(self, prompt: list[int], room: int, draft: bool) -> None:
        """Rank 0 publishes a request on the rendezvous store. The previous one has been read by every rank."""

        store = self.rccl.store
        store.set(f"tf_request/{self._posted}", json.dumps([prompt, room, bool(draft)]))
        if self._posted:
            store.delete_key(f"tf_request/{self._posted - 1}")
        self._posted += 1

    def follow(self) -> None:
        """Ranks above 0: run each of rank 0's requests in step with it, until the process ends."""

        if self.rank == 0:
            raise RuntimeError("rank 0 serves requests; follow() is for the other ranks")
        while True:
            key = f"tf_request/{self._posted}"
            try:
                self.rccl.store.wait([key])
            except Exception as exc:  # noqa: BLE001 - the store's wait timeout: rank 0 is idle
                if "timeout" in str(exc).lower():
                    continue
                raise
            prompt, room, draft = json.loads(self.rccl.store.get(key))
            self._posted += 1
            self._run(prompt, room, None, None, False, draft, None)

    def prefill_caches(self, prompt: Sequence[int]) -> list[dict]:
        """Caches after one forward of ``prompt``. A stored prefix of that length matches this."""

        _, caches = self._span(list(prompt), None, 0, len(prompt))
        return caches


def caches_equal(left: list[dict], right: list[dict]) -> bool:
    return _caches_equal(left, right)
