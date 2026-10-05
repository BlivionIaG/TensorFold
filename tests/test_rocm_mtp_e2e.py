"""MTP draft-then-verify on a synthetic one-layer model: shapes, keyed deterministic drafts, a growing cache."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tensorfold.rocm.model.mtp import MTPEngine  # noqa: E402
from tensorfold.rocm.model.qwen import (  # noqa: E402
    Engine as Kernels,
)
from tensorfold.rocm.model.qwen import (
    FullLayer,
    MTPHead,
    Packed,
    Spec,
    TextModel,
    activation_dtype,
)
from tensorfold.rocm.model.qwen_math import _codes  # noqa: E402

BITS, GROUP = 8, 32

DEVICE = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")


def _act_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    from tensorfold.rocm.kernels.build import gfx_name

    return activation_dtype(gfx_name())


def _packed(n: int, k: int, g: torch.Generator, device: torch.device) -> Packed:
    words = torch.randint(-2**31, 2**31 - 1, (n, k * BITS // 32), generator=g, dtype=torch.int64).to(torch.int32)
    words = words.to(device)
    scale = (torch.rand((n, k // GROUP), generator=g) * 0.02 + 0.005).to(device)
    bias = (torch.randn((n, k // GROUP), generator=g) * 0.01).to(device)
    return Packed(words, scale, bias, BITS, GROUP)


def _vec(n: int, g: torch.Generator, device: torch.device) -> torch.Tensor:
    return (torch.randn(n, generator=g).mul(0.1)).to(device)


def _model_and_mtp(*, hidden: int, vocab: int, heads: int, kv_heads: int, head_dim: int,
                   rotary_dim: int, device: torch.device, g: torch.Generator) -> tuple[TextModel, MTPHead]:
    spec = Spec(hidden=hidden, intermediate=hidden * 2, n_layers=1, heads=heads, kv_heads=kv_heads,
                head_dim=head_dim, key_heads=kv_heads, value_heads=kv_heads, key_dim=head_dim // 2,
                value_dim=head_dim, conv=2, vocab=vocab, eps=1e-6, rope_theta=10000.0, rotary_dim=rotary_dim,
                full_every=1, bits=BITS, group=GROUP)

    def mlp():
        return (_packed(spec.intermediate, spec.hidden, g, device),
                _packed(spec.intermediate, spec.hidden, g, device),
                _packed(spec.hidden, spec.intermediate, g, device))

    full = FullLayer(_vec(spec.hidden, g, device), _vec(spec.hidden, g, device),
                     _packed(spec.heads * spec.head_dim * 2, spec.hidden, g, device),
                     _packed(spec.kv_heads * spec.head_dim, spec.hidden, g, device),
                     _packed(spec.kv_heads * spec.head_dim, spec.hidden, g, device),
                     _packed(spec.hidden, spec.heads * spec.head_dim, g, device),
                     _vec(spec.head_dim, g, device), _vec(spec.head_dim, g, device), *mlp())
    embed = _packed(spec.vocab, spec.hidden, g, device)
    head_proj = _packed(spec.vocab, spec.hidden, g, device)

    mtp = MTPHead(
        fc_e_norm=_vec(spec.hidden, g, device), fc_h_norm=_vec(spec.hidden, g, device),
        fc_e=_packed(spec.hidden, spec.hidden, g, device), fc_h=_packed(spec.hidden, spec.hidden, g, device),
        q_norm=_vec(spec.head_dim, g, device), k_norm=_vec(spec.head_dim, g, device),
        q=_packed(spec.heads * spec.head_dim, spec.hidden, g, device),
        k=_packed(spec.kv_heads * spec.head_dim, spec.hidden, g, device),
        v=_packed(spec.kv_heads * spec.head_dim, spec.hidden, g, device),
        o=_packed(spec.hidden, spec.heads * spec.head_dim, g, device),
        final_norm=_vec(spec.hidden, g, device), head=head_proj,
    )
    model = TextModel(spec, embed, [full], _vec(spec.hidden, g, device), head_proj)
    model.mtp = mtp
    return model, mtp


def _reference_logits(linear, hidden_row: torch.Tensor, head: Packed) -> torch.Tensor:
    n, groups = head.scale.shape
    codes = _codes(head.words, head.bits, groups * head.group).double().view(n, groups, head.group)
    weight = (codes * head.scale.double()[..., None] + head.bias.double()[..., None]).view(n, -1).to(hidden_row.dtype)
    return (hidden_row.double() @ weight.T.double()).float()


class _Sampling:
    temperature = 0.0
    top_k = 0
    top_p = 0.0
    seed = 11
    salt = ""


def test_mtp_engine_forward_shape():
    """One forward on a 1-row input: shape (1, 1, vocab) and cache length grows by 1."""

    if not torch.cuda.is_available():
        pytest.skip("no HIP device")
    device = DEVICE
    dtype = _act_dtype()
    g = torch.Generator().manual_seed(7)
    model, head = _model_and_mtp(hidden=64, vocab=64, heads=4, kv_heads=2, head_dim=16,
                                 rotary_dim=8, device=device, g=g)
    kernels = Kernels(model, schedule="auto", dtype=dtype)
    engine = MTPEngine(model, head, linear=kernels.linear)
    cache = engine.fresh_cache(batch=1, total=16, device=device, dtype=dtype)
    hidden = torch.randn(1, 1, 64, generator=g).to(device).to(dtype=dtype)
    tok = torch.tensor([3], dtype=torch.long, device=device)
    logits, residual = engine.forward(hidden, tok, position=0, cache=cache, dtype=dtype)
    assert logits.shape == (1, 1, 64), f"shape {logits.shape}"
    assert residual.shape == (1, 1, 64), f"residual shape {residual.shape}"
    assert cache["len"] == 1
    assert torch.isfinite(logits).all().item(), "logits contain NaN/inf"
    assert torch.isfinite(residual).all().item(), "residual contains NaN/inf"


def test_mtp_engine_draft_chain_is_deterministic():
    """Two chains with the same hidden and the same seed sample the same tokens in the same order."""

    if not torch.cuda.is_available():
        pytest.skip("no HIP device")
    device = DEVICE
    dtype = _act_dtype()
    g = torch.Generator().manual_seed(7)
    model, head = _model_and_mtp(hidden=64, vocab=128, heads=4, kv_heads=2, head_dim=16,
                                 rotary_dim=8, device=device, g=g)
    kernels = Kernels(model, schedule="auto", dtype=dtype)
    engine = MTPEngine(model, head, linear=kernels.linear)
    hidden = torch.randn(1, 1, 64, generator=g).to(device).to(dtype=dtype)
    cache_a = engine.fresh_cache(batch=1, total=16, device=device, dtype=dtype)
    cache_b = engine.fresh_cache(batch=1, total=16, device=device, dtype=dtype)
    a = engine.draft_chain(hidden, 5, 1, 4, cache_a, sampling=_Sampling(), dtype=dtype)
    b = engine.draft_chain(hidden, 5, 1, 4, cache_b, sampling=_Sampling(), dtype=dtype)
    assert a == b
    assert len(a) == 4
    assert all(0 <= t < 128 for t in a)


def test_mtp_engine_cache_grows_across_steps():
    """A depth=4 chain leaves cache len=4 and the K/V buffer large enough for the next chain."""

    if not torch.cuda.is_available():
        pytest.skip("no HIP device")
    device = DEVICE
    dtype = _act_dtype()
    g = torch.Generator().manual_seed(13)
    model, head = _model_and_mtp(hidden=64, vocab=64, heads=4, kv_heads=2, head_dim=16,
                                 rotary_dim=8, device=device, g=g)
    kernels = Kernels(model, schedule="auto", dtype=dtype)
    engine = MTPEngine(model, head, linear=kernels.linear)
    hidden = torch.randn(1, 1, 64, generator=g).to(device).to(dtype=dtype)
    cache = engine.fresh_cache(batch=1, total=16, device=device, dtype=dtype)
    engine.draft_chain(hidden, 2, 1, 4, cache, sampling=_Sampling(), dtype=dtype)
    assert cache["len"] == 4
    assert cache["k"].shape[2] >= 4
    assert cache["v"].shape[2] >= 4


def test_qwen_engine_generate_with_mtp_emits_tokens():
    """QwenEngine.generate runs to ``max_tokens`` both with and without MTP drafting."""

    from tensorfold.rocm.serving.engine import QwenEngine

    if not torch.cuda.is_available():
        pytest.skip("no HIP device")
    device = DEVICE
    dtype = _act_dtype()
    g = torch.Generator().manual_seed(42)
    model, head = _model_and_mtp(hidden=64, vocab=64, heads=4, kv_heads=2, head_dim=16,
                                 rotary_dim=8, device=device, g=g)
    kernels = Kernels(model, schedule="auto", dtype=dtype)
    eos = (0,)
    prompt = [1, 2, 3, 4]

    def _run(depth: int) -> list[int]:
        eng = QwenEngine(model, kernels, eos, tp=1, rank=0, rccl=None, no_drafts=False, mtp_depth=depth)
        emitted: list[int] = []

        def _on(tokens: list[int]) -> bool | None:
            emitted.extend(tokens)
            return None

        eng.generate(prompt, max_tokens=8, sampling=_Sampling(), on_tokens=_on, stop_eos=False)
        return emitted

    serial = _run(0)
    drafted = _run(2)
    assert len(serial) == 8, f"serial emitted {serial}"
    # MTP depth=2 may accept 0..2 drafts, so the run produces 8..24 tokens depending on accept rate.
    assert 8 <= len(drafted) <= 24, f"drafted emitted {drafted}"
    assert all(0 <= t < 64 for t in serial)
    assert all(0 <= t < 64 for t in drafted)