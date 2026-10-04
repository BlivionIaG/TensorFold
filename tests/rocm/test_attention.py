"""HIP causal attention matches the fp32 spec on the activation dtype, including a short cache prefix."""

import torch
import pytest

if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
    pytest.skip("RDNA only", allow_module_level=True)

from tensorfold.rocm.attention import causal  # noqa: E402
from tensorfold.rocm.qwen_math import causal_attend  # noqa: E402


def _spec(q, k, v, scale, q_pos0):
    """Upcast the stored cache, then the chunked fp32 attention. Repeating heads copies values."""

    heads, kv_heads = q.shape[1], k.shape[1]
    kk, vv = k.float(), v.float()
    if heads != kv_heads:
        kk = kk.repeat_interleave(heads // kv_heads, dim=1)
        vv = vv.repeat_interleave(heads // kv_heads, dim=1)
    return causal_attend(q, kk, vv, scale, q_pos0)


def _case(batch, qlen, span, heads, kv_heads, dim, dtype, seed, q_pos0=0):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    q = torch.randn(batch, heads, qlen, dim, generator=g, device="cuda")
    k = torch.randn(batch, kv_heads, span, dim, generator=g, device="cuda", dtype=dtype)
    v = torch.randn(batch, kv_heads, span, dim, generator=g, device="cuda", dtype=dtype)
    scale = dim ** -0.5
    got = causal(q, k, v, scale, q_pos0)
    ref = _spec(q, k, v, scale, q_pos0)
    gap = (got - ref).abs().max().item()
    assert torch.allclose(got, ref, rtol=1e-3, atol=1e-3), gap


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("qlen,span", [(1, 1), (17, 17), (1, 17)])
def test_tiny_attention_matches_the_spec(dtype, qlen, span):
    q_pos0 = span - 1 if qlen == 1 and span > 1 else 0
    _case(1, qlen, span, 4, 2, 16, dtype, 11 + qlen + span, q_pos0)


def test_model_shaped_attention_matches_the_spec():
    _case(1, 4, 32, 8, 2, 256, torch.bfloat16, 40)
    _case(1, 4, 32, 8, 2, 256, torch.float16, 45)
    _case(1, 4, 32, 8, 2, 256, torch.float32, 46)
    # Decode scores one warp per key. The query sees the whole span.
    _case(1, 1, 32, 8, 2, 256, torch.bfloat16, 41, q_pos0=31)
    _case(8, 1, 32, 8, 2, 256, torch.float16, 42, q_pos0=31)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_prefill_offset_matches_the_spec(dtype):
    _case(1, 20, 48, 4, 2, 64, dtype, 43, q_pos0=8)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_prefill_matches_the_decode_walk(dtype, monkeypatch):
    """The HIP tile and the one-query walk agree on one cache. Both are the fp32 product."""

    monkeypatch.setenv("TENSORFOLD_ATTN", "hip")
    g = torch.Generator(device="cuda").manual_seed(60)
    heads, kv, dim = 16, 4, 256
    scale = dim ** -0.5
    q = torch.randn(2, heads, 32, dim, generator=g, device="cuda")
    k = torch.randn(2, kv, 48, dim, generator=g, device="cuda", dtype=dtype)
    v = torch.randn(2, kv, 48, dim, generator=g, device="cuda", dtype=dtype)
    pre = causal(q, k, v, scale, 0)
    for index in (0, 15, 31):
        one = causal(q[:, :, index:index + 1].contiguous(), k, v, scale, index)
        gap = (one - pre[:, :, index:index + 1]).abs().max().item()
        assert torch.allclose(one, pre[:, :, index:index + 1], rtol=1e-4, atol=1e-4), gap
    q2 = torch.randn(1, heads, 16, dim, generator=g, device="cuda")
    k2 = torch.randn(1, kv, 40, dim, generator=g, device="cuda", dtype=dtype)
    v2 = torch.randn(1, kv, 40, dim, generator=g, device="cuda", dtype=dtype)
    pre2 = causal(q2, k2, v2, scale, 8)
    for local in (0, 7, 15):
        one = causal(q2[:, :, local:local + 1].contiguous(), k2, v2, scale, 8 + local)
        gap = (one - pre2[:, :, local:local + 1]).abs().max().item()
        assert torch.allclose(one, pre2[:, :, local:local + 1], rtol=1e-4, atol=1e-4), gap


def test_triton_is_selected_only_where_it_was_faster():
    from tensorfold.rocm.attention import triton_prefill

    assert triton_prefill("gfx1030", 1, 512) is False
    assert triton_prefill("gfx1030", 1, 1024) is True
    assert triton_prefill("gfx1030", 1, 32768) is True
    assert triton_prefill("gfx1030", 8, 256) is False
    assert triton_prefill("gfx1030", 8, 384) is True
    assert triton_prefill("gfx1100", 1, 64) is False
    assert triton_prefill("gfx1100", 1, 128) is True
    assert triton_prefill("gfx1100", 1, 32768) is True
    assert triton_prefill("gfx1100", 8, 32) is False
    assert triton_prefill("gfx1100", 8, 64) is True
    assert triton_prefill("gfx1100", 8, 32768) is True
    assert triton_prefill("gfx1100", 2, 64) is False
    assert triton_prefill("gfx1100", 2, 128) is True
    assert triton_prefill("gfx1151", 1, 4096) is False


def test_triton_prefill_matches_the_spec_and_the_decode_walk():
    """gfx1030 and gfx11 prefill use the 64-row tile. It stays within the fp32 spec."""

    from tensorfold.rocm.attention_triton import prefill
    from tensorfold.rocm.build import gfx_name

    name = gfx_name()
    if not (name.startswith("gfx103") or name.startswith("gfx11")):
        pytest.skip("RDNA2/RDNA3 Triton prefill")
    g = torch.Generator(device="cuda").manual_seed(70)
    heads, kv, dim = 8, 2, 256
    scale = dim ** -0.5
    dtype = torch.float16 if name.startswith("gfx103") else torch.bfloat16
    q = torch.randn(1, heads, 96, dim, generator=g, device="cuda")
    k = torch.randn(1, kv, 160, dim, generator=g, device="cuda", dtype=dtype)
    v = torch.randn(1, kv, 160, dim, generator=g, device="cuda", dtype=dtype)
    got = prefill(q, k, v, scale, 8, force=True)
    assert got is not None
    ref = _spec(q, k, v, scale, 8)
    gap = (got - ref).abs().max().item()
    assert torch.allclose(got, ref, rtol=1e-3, atol=1e-3), gap
    for local in (0, 31, 95):
        one = causal(q[:, :, local:local + 1].contiguous(), k, v, scale, 8 + local)
        walk = (one - got[:, :, local:local + 1]).abs().max().item()
        assert torch.allclose(one, got[:, :, local:local + 1], rtol=1e-3, atol=1e-3), walk


def test_a_cache_prefix_uses_its_stride():
    g = torch.Generator(device="cuda")
    g.manual_seed(44)
    q = torch.randn(1, 4, 3, 16, generator=g, device="cuda")
    k = torch.randn(1, 2, 40, 16, generator=g, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 2, 40, 16, generator=g, device="cuda", dtype=torch.float16)
    scale = 16 ** -0.5
    got = causal(q, k[:, :, :17], v[:, :, :17], scale, 0)
    ref = _spec(q, k[:, :, :17].contiguous(), v[:, :, :17].contiguous(), scale, 0)
    assert torch.allclose(got, ref, rtol=1e-3, atol=1e-3)
