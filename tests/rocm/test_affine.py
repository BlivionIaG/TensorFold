"""Packed affine on RDNA: a row's bits do not depend on how many rows share the launch, and the weight is never expanded."""

import ctypes
import ctypes.util

import numpy as np
import pytest
import torch

if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
    pytest.skip("RDNA only", allow_module_level=True)

from tensorfold.rocm.affine import matmul, matmul_group, matmul_pair  # noqa: E402
from tensorfold.rocm.build import WMMA, gfx_name  # noqa: E402

ROWS = [1, 2, 7, 15, 16, 17, 31, 32]

_FMAF = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6").fmaf
_FMAF.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
_FMAF.restype = ctypes.c_float


def _fmaf(a, b, c):
    """float32 fused multiply-add. torch's addcmul is a separate multiply and add, so it is not this rounding."""

    a, b, c = np.broadcast_arrays(np.asarray(a, np.float32), np.asarray(b, np.float32), np.asarray(c, np.float32))
    fa = np.ascontiguousarray(a, np.float32).reshape(-1)
    fb = np.ascontiguousarray(b, np.float32).reshape(-1)
    fc = np.ascontiguousarray(c, np.float32).reshape(-1)
    out = np.empty(fa.shape, np.float32)
    fmaf = _FMAF
    for i in range(out.shape[0]):
        out[i] = fmaf(fa[i], fb[i], fc[i])
    return out.reshape(a.shape)


def _code(row, k, bits):
    bit = k * bits
    word, shift = divmod(bit, 32)
    low = int(row[word].item()) & 0xFFFFFFFF
    high = int(row[word + 1].item()) & 0xFFFFFFFF if shift + bits > 32 and word + 1 < row.numel() else 0
    value = low >> shift
    if shift + bits > 32:
        value |= high << ((32 - shift) & 31)
    return value & ((1 << bits) - 1)


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
        if shift + bits > 32:
            words[:, word + 1] |= value >> (32 - shift)
    scale = torch.rand((n, k // group), generator=g) * 0.2 + 0.02
    bias = torch.randn((n, k // group), generator=g) * 0.05
    return codes, words.to(torch.int32), scale, bias


def _reference(x, codes, scale, bias, group):
    """The GEMV formula: BF16 rounding of each code, then one product at a time, then the group scale and bias."""

    q = codes.to(torch.float32).to(torch.bfloat16).to(torch.float32).cpu().numpy()
    xf = x.float().cpu().numpy()
    sc = scale.float().cpu().numpy()
    bi = bias.float().cpu().numpy()
    acc = np.zeros((xf.shape[0], q.shape[0]), np.float32)
    for start in range(0, xf.shape[1], group):
        dot = np.zeros((xf.shape[0], q.shape[0]), np.float32)
        summed = np.zeros((xf.shape[0],), np.float32)
        for t in range(group):
            xv = xf[:, start + t]
            dot = _fmaf(xv[:, None], q[:, start + t][None, :], dot)
            summed = np.add(summed, xv, dtype=np.float32)
        g = start // group
        acc = _fmaf(dot, sc[:, g][None, :], acc)
        acc = _fmaf(summed[:, None], bi[:, g][None, :], acc)
    return torch.from_numpy(np.ascontiguousarray(acc))


def _skip_wmma(schedule):
    if schedule == "wmma" and gfx_name() not in WMMA:
        pytest.skip("this RDNA part has no WMMA; auto stays on the GEMV")


def test_visible_device_when_pinned():
    import os
    want = os.environ.get("EXPECT_BUS")
    if not want:
        return
    bus = getattr(torch.cuda.get_device_properties(0), "pci_bus_id", None)
    # hip properties expose the bus as an int on ROCm builds; skip the compare when this torch does not.
    if isinstance(bus, int):
        assert f"{bus:02x}" == want.lower(), f"visible bus {bus:02x}, expected {want}"


def test_pack_round_trip():
    codes, words, _, _ = _pack(4, 96, 3, 32, 4)
    for n in range(codes.shape[0]):
        for k in range(codes.shape[1]):
            assert _code(words[n], k, 3) == int(codes[n, k])


@pytest.mark.parametrize("bits,group,k", [(8, 64, 128), (4, 128, 256), (3, 32, 96), (5, 32, 128)])
def test_gemv_matches_the_formula(bits, group, k):
    codes, words, scale, bias = _pack(40, k, bits, group, 10 + bits)
    g = torch.Generator(device="cuda").manual_seed(10 + bits)
    x = torch.randn((8, k), generator=g, device="cuda", dtype=torch.bfloat16)
    got = matmul(x, words.cuda(), scale.cuda(), bias.cuda(), bits=bits, group=group, schedule="gemv", f32=True)
    assert torch.equal(got.cpu(), _reference(x, codes, scale, bias, group))


@pytest.mark.parametrize("schedule", ["auto", "gemv", "wmma"])
def test_rows_do_not_depend_on_row_count(schedule):
    _skip_wmma(schedule)
    _, words, scale, bias = _pack(48, 192, 8, 64, 3)
    words, scale, bias = words.cuda(), scale.cuda(), bias.cuda()
    g = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn((max(ROWS), 192), generator=g, device="cuda", dtype=torch.bfloat16)
    alone = torch.cat([matmul(x[r:r + 1], words, scale, bias, bits=8, group=64, schedule=schedule, f32=True)
                       for r in range(max(ROWS))])
    for m in ROWS:
        assert torch.equal(matmul(x[:m], words, scale, bias, bits=8, group=64, schedule=schedule, f32=True), alone[:m])
    perm = torch.randperm(max(ROWS), device="cuda")
    assert torch.equal(matmul(x[perm], words, scale, bias, bits=8, group=64, schedule=schedule, f32=True), alone[perm])
    if schedule == "auto" and gfx_name() in WMMA:
        wmma = torch.cat([matmul(x[r:r + 1], words, scale, bias, bits=8, group=64, schedule="wmma", f32=True)
                          for r in range(max(ROWS))])
        assert torch.equal(alone, wmma)


@pytest.mark.parametrize("schedule", ["auto", "gemv", "wmma"])
def test_word_spanning_rows_do_not_depend_on_row_count(schedule):
    """3-bit codes cross the 32-bit word. The same row in a wide launch matches the row alone."""

    _skip_wmma(schedule)
    _, words, scale, bias = _pack(32, 96, 3, 32, 9)
    words, scale, bias = words.cuda(), scale.cuda(), bias.cuda()
    x = torch.randn((17, 96), device="cuda", dtype=torch.bfloat16)
    alone = matmul(x[:1], words, scale, bias, bits=3, group=32, schedule=schedule, f32=True)
    assert torch.equal(matmul(x, words, scale, bias, bits=3, group=32, schedule=schedule, f32=True)[:1], alone)


def test_fp16_is_the_rdna2_schedule():
    """A WMMA part keeps one BF16 formula. RDNA2 FP16 auto matches its gemv and does not depend on M."""

    _, words, scale, bias = _pack(48, 192, 8, 64, 11)
    words, scale, bias = words.cuda(), scale.cuda(), bias.cuda()
    g = torch.Generator(device="cuda").manual_seed(11)
    x = torch.randn((max(ROWS), 192), generator=g, device="cuda", dtype=torch.float16)
    if gfx_name() in WMMA:
        with pytest.raises(ValueError, match="RDNA2"):
            matmul(x[:1], words, scale, bias, bits=8, group=64)
        return
    with pytest.raises(RuntimeError):
        matmul(x[:1], words, scale, bias, bits=8, group=64, schedule="wmma", f32=True)
    stored = matmul(x[:1], words, scale, bias, bits=8, group=64)
    assert stored.dtype == torch.float16
    alone = torch.cat([matmul(x[r:r + 1], words, scale, bias, bits=8, group=64, schedule="auto", f32=True)
                       for r in range(max(ROWS))])
    assert not torch.equal(alone, torch.zeros_like(alone))
    gemv = torch.cat([matmul(x[r:r + 1], words, scale, bias, bits=8, group=64, schedule="gemv", f32=True)
                      for r in range(max(ROWS))])
    assert torch.equal(alone, gemv)
    for m in ROWS:
        assert torch.equal(matmul(x[:m], words, scale, bias, bits=8, group=64, f32=True), alone[:m])
    perm = torch.randperm(max(ROWS), device="cuda")
    assert torch.equal(matmul(x[perm], words, scale, bias, bits=8, group=64, f32=True), alone[perm])


def test_pair_matches_two_wmma_launches():
    if gfx_name() not in WMMA:
        pytest.skip("the paired matmul is the WMMA schedule")
    _, words_a, scale_a, bias_a = _pack(64, 256, 8, 64, 21)
    _, words_b, scale_b, bias_b = _pack(64, 256, 8, 64, 22)
    tensors = [t.cuda() for t in (words_a, scale_a, bias_a, words_b, scale_b, bias_b)]
    words_a, scale_a, bias_a, words_b, scale_b, bias_b = tensors
    x = torch.randn(17, 256, device="cuda", dtype=torch.bfloat16)
    for rows in (1, 8, 17):
        got_a, got_b = matmul_pair(x[:rows], words_a, scale_a, bias_a, words_b, scale_b, bias_b, bits=8, group=64,
                                   f32=True)
        one_a = matmul(x[:rows], words_a, scale_a, bias_a, bits=8, group=64, schedule="wmma", f32=True)
        one_b = matmul(x[:rows], words_b, scale_b, bias_b, bits=8, group=64, schedule="wmma", f32=True)
        assert torch.equal(got_a, one_a)
        assert torch.equal(got_b, one_b)


@pytest.mark.parametrize("group,k", [(32, 128), (64, 256), (128, 256)])
def test_group_matches_solo_wmma(group, k):
    """Columns of different widths in one launch match the same columns launched alone."""

    if gfx_name() not in WMMA:
        pytest.skip("the grouped matmul is the WMMA schedule")
    packs = []
    for n, seed in ((40, 31), (16, 32), (64, 33)):
        packs.append(_pack(n, k, 8, group, seed))
    tensors = []
    for _, words, scale, bias in packs:
        tensors.append((words.cuda(), scale.cuda(), bias.cuda()))
    x = torch.randn(17, k, device="cuda", dtype=torch.bfloat16)
    for rows in (1, 8, 17):
        got = matmul_group(x[:rows], tensors, bits=8, group=group, f32=True)
        for (words, scale, bias), part in zip(tensors, got):
            solo = matmul(x[:rows], words, scale, bias, bits=8, group=group, schedule="wmma", f32=True)
            assert torch.equal(part, solo)


def test_wmma_partial_tile_matches_one_row():
    """The last 16-column tile is short. Its row still matches that row launched alone."""

    if gfx_name() not in WMMA:
        pytest.skip("partial-tile WMMA is the gfx11 schedule")
    _, words, scale, bias = _pack(40, 128, 8, 64, 5)
    words, scale, bias = words.cuda(), scale.cuda(), bias.cuda()
    x = torch.randn(17, 128, device="cuda", dtype=torch.bfloat16)
    alone = matmul(x[:1], words, scale, bias, bits=8, group=64, schedule="wmma", f32=True)
    wide = matmul(x, words, scale, bias, bits=8, group=64, schedule="wmma", f32=True)
    assert torch.equal(wide[:1], alone)


def test_fp16_word_spanning_rows_do_not_depend_on_row_count():
    if gfx_name() in WMMA:
        pytest.skip("FP16 activations are the RDNA2 schedule")
    _, words, scale, bias = _pack(32, 96, 3, 32, 13)
    words, scale, bias = words.cuda(), scale.cuda(), bias.cuda()
    x = torch.randn((17, 96), device="cuda", dtype=torch.float16)
    alone = matmul(x[:1], words, scale, bias, bits=3, group=32, f32=True)
    assert torch.equal(matmul(x, words, scale, bias, bits=3, group=32, f32=True)[:1], alone)
