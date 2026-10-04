"""Short-row norm, conv, and RoPE match the PyTorch formulas. Spanning affine codes match the bit reader."""

import pytest
import torch

if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
    pytest.skip("RDNA only", allow_module_level=True)

from tensorfold.rocm.checkpoint import _affine_quant  # noqa: E402
from tensorfold.rocm.qwen_math import _codes, _rms_torch, apply_rope, causal_conv, rms_norm  # noqa: E402


def _code(row, k, bits):
    bit = k * bits
    word, shift = divmod(bit, 32)
    low = int(row[word].item()) & 0xFFFFFFFF
    high = int(row[word + 1].item()) & 0xFFFFFFFF if shift + bits > 32 and word + 1 < row.numel() else 0
    value = low >> shift
    if shift + bits > 32:
        value |= high << ((32 - shift) & 31)
    return value & ((1 << bits) - 1)


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
def test_codes_match_the_bit_reader(bits):
    g = torch.Generator().manual_seed(bits + 20)
    words = torch.randint(-(2**31), 2**31, (4, 40), dtype=torch.int32, generator=g)
    got = _codes(words, bits, 96)
    for row in range(words.shape[0]):
        for k in range(96):
            assert int(got[row, k]) == _code(words[row], k, bits)


def test_affine_quant_accepts_the_mlx_widths():
    for bits in (2, 3, 4, 5, 6, 8):
        for group in (32, 64, 128):
            assert _affine_quant({"mode": "affine", "bits": bits, "group_size": group}) == (bits, group)
    with pytest.raises(ValueError):
        _affine_quant({"mode": "affine", "bits": 7, "group_size": 64})


def test_short_rms_matches_the_formula():
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(4, 8, 128, generator=g, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(128, generator=g, device="cuda")
    got = rms_norm(x, weight, 1e-6)
    ref = _rms_torch(x, weight, 1e-6)
    assert torch.allclose(got, ref, rtol=1e-4, atol=1e-4)
    bare = rms_norm(x, None, 1e-6)
    assert torch.allclose(bare, _rms_torch(x, None, 1e-6), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rms_rows_do_not_depend_on_the_row_count(dtype):
    x = torch.randn(1, 600, 16, 128, device="cuda").to(dtype)
    weight = torch.randn(128, device="cuda")
    whole = rms_norm(x, weight, 1e-6)
    for start, stop in ((0, 1), (0, 16), (16, 22), (22, 600)):
        assert torch.equal(rms_norm(x[:, start:stop].contiguous(), weight, 1e-6), whole[:, start:stop])


def test_decode_conv_matches_the_loop():
    g = torch.Generator(device="cuda").manual_seed(4)
    weight = torch.randn(32, 4, generator=g, device="cuda")
    state = torch.randn(2, 3, 32, generator=g, device="cuda")
    x = torch.randn(2, 1, 32, generator=g, device="cuda", dtype=torch.float16)
    # The host loop is the reference, so run it on a clone before the HIP path updates state.
    host_state = state.clone()
    window = torch.cat((host_state.float(), x.float()), dim=1)
    out = torch.zeros(2, 1, 32, device="cuda")
    for tap in range(4):
        out = out + window[:, tap:tap + 1] * weight[:, tap].view(1, 1, 32)
    ref = torch.nn.functional.silu(out)
    got, new_state = causal_conv(x, weight, state.clone())
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-5)
    assert torch.allclose(new_state, window[:, 1:].contiguous(), rtol=1e-5, atol=1e-5)


def test_decode_rope_matches_the_formula():
    g = torch.Generator(device="cuda").manual_seed(5)
    x = torch.randn(2, 4, 1, 64, generator=g, device="cuda", dtype=torch.bfloat16)
    got = apply_rope(x, 17, 10_000_000.0, 32)
    half = 16
    freq = 1.0 / (10_000_000.0 ** (torch.arange(half, device="cuda", dtype=torch.float32) / half))
    ang = 17 * freq
    cos, sin = ang.cos(), ang.sin()
    xf = x.float()
    x1, x2 = xf[..., :half], xf[..., half:32]
    rot = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos, xf[..., 32:]), dim=-1)
    assert got.dtype == x.dtype
    assert torch.allclose(got, rot.to(dtype=got.dtype), rtol=1e-4, atol=1e-4)


def test_rms_in_the_activation_dtype_matches_fp32_then_cast():
    """FP16 and BF16 rows give the bits of the fp32 kernel followed by one cast, which is what the forward ran."""

    from tensorfold.rocm.act import rms

    g = torch.Generator(device="cuda").manual_seed(9)
    weight = torch.randn(5120, generator=g, device="cuda")
    for dtype in (torch.float16, torch.bfloat16):
        x = torch.randn(3, 5120, generator=g, device="cuda").to(dtype)
        assert torch.equal(rms(x, weight, 1e-6), rms(x.float(), weight, 1e-6).to(dtype))
        assert torch.equal(rms(x, None, 1e-6), rms(x.float(), None, 1e-6).to(dtype))
