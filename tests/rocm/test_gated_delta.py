"""The HIP gated delta matches its lane reduction, and a row does not change with a longer batch or sequence."""

import pytest
torch = pytest.importorskip("torch")

if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
    pytest.skip("RDNA only", allow_module_level=True)

from tensorfold.rocm.gated_delta import recurrence  # noqa: E402
from tensorfold.rocm.qwen_math import _gate_beta, gated_delta, gated_delta_reference  # noqa: E402


def _args(batch, length, key_heads, key_dim, value_heads, value_dim, seed):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    shape = (batch, length, key_heads, key_dim)
    q = torch.randn(shape, generator=g, device="cuda")
    k = torch.randn(shape, generator=g, device="cuda")
    v = torch.randn(batch, length, value_heads, value_dim, generator=g, device="cuda")
    gate = torch.rand(batch, length, value_heads, generator=g, device="cuda") * 0.5 + 0.25
    beta = torch.rand(batch, length, value_heads, generator=g, device="cuda")
    state = torch.randn(batch, value_heads, value_dim, key_dim, generator=g, device="cuda")
    return q, k, v, gate, beta, state


@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("length", [1, 4, 32])
@pytest.mark.parametrize("key_dim,value_dim", [(16, 16), (128, 128)])
def test_kernel_matches_the_reference(batch, length, key_dim, value_dim):
    q, k, v, gate, beta, state = _args(batch, length, 2, key_dim, 2, value_dim, 5 + length + key_dim)
    got_y, got_state = recurrence(q, k, v, gate, beta, state.clone())
    ref_y, ref_state = gated_delta_reference(q, k, v, gate, beta, state)
    assert torch.equal(got_y, ref_y)
    assert torch.equal(got_state, ref_state)


def test_repeated_key_heads_match_the_reference():
    q, k, v, gate, beta, state = _args(2, 4, 2, 16, 4, 16, 19)
    got_y, got_state = recurrence(q, k, v, gate, beta, state.clone())
    ref_y, ref_state = gated_delta_reference(q, k, v, gate, beta, state)
    assert torch.equal(got_y, ref_y)
    assert torch.equal(got_state, ref_state)


def test_a_prefix_matches_a_shorter_run_and_the_state_splits():
    q, k, v, gate, beta, state = _args(2, 8, 2, 128, 2, 16, 23)
    full_y, full_state = recurrence(q, k, v, gate, beta, state.clone())
    head_y, head_state = recurrence(q[:, :4], k[:, :4], v[:, :4], gate[:, :4], beta[:, :4], state.clone())
    assert torch.equal(full_y[:, :4], head_y)
    tail_y, tail_state = recurrence(q[:, 4:], k[:, 4:], v[:, 4:], gate[:, 4:], beta[:, 4:], head_state)
    assert torch.equal(full_y[:, 4:], tail_y)
    assert torch.equal(full_state, tail_state)


def test_one_request_matches_inside_a_wider_batch():
    q, k, v, gate, beta, state = _args(4, 4, 2, 16, 2, 16, 29)
    wide_y, wide_state = recurrence(q, k, v, gate, beta, state.clone())
    one_y, one_state = recurrence(q[:1], k[:1], v[:1], gate[:1], beta[:1], state[:1].clone())
    assert torch.equal(wide_y[:1], one_y)
    assert torch.equal(wide_state[:1], one_state)


def test_the_public_entry_uses_the_same_reduction():
    q, k, v, gate, beta, state = _args(1, 4, 2, 16, 2, 16, 31)
    value_heads = v.shape[2]
    a = torch.randn(1, 4, value_heads, device="cuda")
    b = torch.randn_like(a)
    a_log = torch.randn(value_heads, device="cuda")
    dt_bias = torch.randn(value_heads, device="cuda")
    got_y, got_state = gated_delta(q, k, v, a, b, a_log, dt_bias, state.clone())
    built_gate, built_beta = _gate_beta(a, b, a_log, dt_bias)
    ref_y, ref_state = gated_delta_reference(q, k, v, built_gate, built_beta, state)
    assert torch.equal(got_y, ref_y)
    assert torch.equal(got_state, ref_state)
    assert not torch.equal(got_y, torch.zeros_like(got_y))
