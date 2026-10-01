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
    # Decode scores one warp per key. The query sees the whole span.
    _case(1, 1, 32, 8, 2, 256, torch.bfloat16, 41, q_pos0=31)
    _case(8, 1, 32, 8, 2, 256, torch.float16, 42, q_pos0=31)


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
