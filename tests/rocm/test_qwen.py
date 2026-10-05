"""The RDNA text forward on a tiny stack matches an fmaf reference that never calls the ROCm forward."""

import pytest
torch = pytest.importorskip("torch")

if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
    pytest.skip("RDNA only", allow_module_level=True)

from tensorfold.rocm.kernels import affine as affine_mod  # noqa: E402
from tensorfold.rocm.model.forward import greedy  # noqa: E402
from tensorfold.rocm.model.qwen import Engine, FullLayer, LinearLayer, TextModel  # noqa: E402
from tensorfold.rocm.model.qwen_math import Packed, Spec, affine_reference  # noqa: E402


def _pack(n, k, bits, group, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    codes = torch.randint(0, 1 << bits, (n, k), generator=g)
    words = torch.zeros((n, k * bits // 32), dtype=torch.int64)
    for col in range(k):
        value = (codes[:, col] & ((1 << bits) - 1)).to(torch.int64)
        bit = col * bits
        word, shift = divmod(bit, 32)
        words[:, word] |= value << shift
    scale = torch.rand((n, k // group), generator=g) * 0.05 + 0.02
    bias = torch.randn((n, k // group), generator=g) * 0.02
    return Packed(words.to(torch.int32), scale, bias, bits, group)


def _tiny(device):
    """Two layers, group 32, so every projection K is a whole number of groups. Weights stay packed."""

    bits, group = 8, 32
    spec = Spec(hidden=32, intermediate=64, n_layers=2, heads=4, kv_heads=2, head_dim=16, key_heads=2,
                value_heads=2, key_dim=16, value_dim=16, conv=4, vocab=48, eps=1e-6, rope_theta=10000.0,
                rotary_dim=8, full_every=2, bits=bits, group=group)
    g = torch.Generator()
    g.manual_seed(7)

    def vec(n):
        return torch.randn(n, generator=g).add(1)

    def packed(n, k, seed):
        item = _pack(n, k, bits, group, seed)
        return Packed(item.words.to(device), item.scale.to(device), item.bias.to(device), bits, group)

    linear = LinearLayer(
        vec(spec.hidden).to(device), vec(spec.hidden).to(device),
        packed(spec.key_width * 2 + spec.value_width, spec.hidden, 11),
        packed(spec.value_width, spec.hidden, 12),
        packed(spec.value_heads, spec.hidden, 13), packed(spec.value_heads, spec.hidden, 14),
        torch.randn(spec.key_width * 2 + spec.value_width, spec.conv, generator=g).mul(0.05).to(device),
        torch.randn(spec.value_heads, generator=g).to(device), torch.randn(spec.value_heads, generator=g).to(device),
        vec(spec.value_dim).to(device), packed(spec.hidden, spec.value_width, 15),
        packed(spec.intermediate, spec.hidden, 16), packed(spec.intermediate, spec.hidden, 17),
        packed(spec.hidden, spec.intermediate, 18))
    full = FullLayer(
        vec(spec.hidden).to(device), vec(spec.hidden).to(device),
        packed(spec.heads * spec.head_dim * 2, spec.hidden, 21),
        packed(spec.kv_heads * spec.head_dim, spec.hidden, 22),
        packed(spec.kv_heads * spec.head_dim, spec.hidden, 23),
        packed(spec.hidden, spec.heads * spec.head_dim, 24),
        vec(spec.head_dim).to(device), vec(spec.head_dim).to(device),
        packed(spec.intermediate, spec.hidden, 25), packed(spec.intermediate, spec.hidden, 26),
        packed(spec.hidden, spec.intermediate, 27))
    embed = packed(spec.vocab, spec.hidden, 3)
    return TextModel(spec, embed, [linear, full], vec(spec.hidden).to(device))


def _reference_linear(flat, packed):
    return affine_reference(flat, packed).to(flat.device)


def test_greedy_ids_match_the_reference_at_c1_and_c8():
    device = torch.device("cuda")
    model = _tiny(device)
    for packed in (model.embed, model.layers[0].qkv, model.layers[1].q, model.layers[1].o):
        assert packed.words.dtype == torch.int32 and packed.words.ndim == 2
        assert packed.words.shape[1] == packed.scale.shape[1] * packed.group * packed.bits // 32
    seen = []
    real = affine_mod.matmul

    def spy(x, words, scale, bias, **kwargs):
        assert words.dtype == torch.int32 and words.ndim == 2
        assert words.shape[1] == x.shape[1] * kwargs["bits"] // 32
        assert words.shape[1] != x.shape[1]
        assert not words.dtype.is_floating_point
        seen.append(x.shape[0])
        return real(x, words, scale, bias, **kwargs)

    affine_mod.matmul = spy
    try:
        engine = Engine(model, schedule="gemv", dtype=torch.bfloat16)
        one = [[1, 2, 3, 4]]
        eight = [[(row * 3 + index) % 40 + 1 for index in range(4)] for row in range(8)]
        got_one = engine.generate(one, 3)
        assert max(seen) == 4
        before = len(seen)
        ref_one = greedy(model, one, 3, _reference_linear, device, cache_dtype=torch.bfloat16)
        assert len(seen) == before
        assert got_one == ref_one and len(got_one[0]) == 3
        seen.clear()
        got_eight = engine.generate(eight, 3)
        assert max(seen) == 32
        before = len(seen)
        ref_eight = greedy(model, eight, 3, _reference_linear, device, cache_dtype=torch.bfloat16)
        assert len(seen) == before
        assert got_eight == ref_eight
        assert len(got_eight) == 8 and all(len(row) == 3 for row in got_eight)
        assert engine.projections > 0
    finally:
        affine_mod.matmul = real
