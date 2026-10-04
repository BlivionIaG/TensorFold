"""W4A16 GPTQ on RDNA2: the packed int4 weight reproduces the fp16 dot's reference."""

import pytest
torch = pytest.importorskip("torch")

if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
    pytest.skip("RDNA only", allow_module_level=True)

from tensorfold.rocm.build import gfx_name  # noqa: E402
from tensorfold.rocm.qgemm import matmul, moe  # noqa: E402

try:
    _GFX = gfx_name()
except RuntimeError:  # gfx_name refuses a device outside TensorFold's RDNA schedule
    pytest.skip("not an RDNA device", allow_module_level=True)

if _GFX != "gfx1030":
    pytest.skip("the W4A16 GPTQ path is the gfx1030 fp16 dot", allow_module_level=True)


_PERM = (0, 2, 4, 6, 1, 3, 5, 7)  # exllama shuffle: nibble j of a word holds K offset _PERM[j]


def _pack(n, k, group, seed):
    """GPTQ packing: one qweight word is 8 shuffled K nibbles, one qzeros word 8 consecutive N nibbles."""

    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (k, n), generator=g)
    zeros = torch.randint(0, 16, (k // group, n), generator=g)
    qw = torch.zeros((k // 8, n), dtype=torch.int64)
    for i in range(8):
        qw |= (codes[_PERM[i]::8, :] & 0xF) << (4 * i)
    qz = torch.zeros((k // group, n // 8), dtype=torch.int64)
    for i in range(8):
        qz |= (zeros[:, i::8] & 0xF) << (4 * i)
    scales = (torch.rand((k // group, n), generator=g) * 0.2 + 0.02).to(torch.float16)
    return codes, zeros, qw.to(torch.int32), qz.to(torch.int32), scales


def _reference(x, codes, zeros, scales, zero_offset):
    """w[n, k] = (code - (zero + offset)) * scale, in the group that owns k, computed exactly in fp32.

    The kernel computes the same product through exllamav2's fp16 bit-trick, whose (q + 1024) and
    -(1024 + zero) terms cancel at the 1024 offset -- that cancellation costs about scale * 2**-11
    per weight element, which is the gap the comparison tolerance covers. This reference is exact,
    so a mismatch beyond it is a bug, and the decode/prefill agreement test pins the two kernels
    against each other with no such slack.
    """

    k, n = codes.shape
    groupsize = k // scales.shape[0]
    w = torch.empty((n, k), dtype=torch.float32)
    q = codes.t().float()
    z = zeros.float()
    for gi in range(scales.shape[0]):
        sl = slice(gi * groupsize, (gi + 1) * groupsize)
        w[:, sl] = (q[:, sl] - (z[gi].unsqueeze(1) + zero_offset)) * scales[gi].float().unsqueeze(1)
    return x.float().cpu() @ w.t()


@pytest.mark.parametrize("v2", [True, False])
@pytest.mark.parametrize("n,k,group", [(64, 128, 32), (128, 256, 64), (256, 512, 128), (1024, 1024, 128)])
@pytest.mark.parametrize("m", [1, 3, 8, 16, 65, 512])
def test_matches_reference(m, n, k, group, v2):
    """Every M-tile (decode and prefill) lands on the same dequantized product within fp16 rounding."""

    codes, zeros, qweight, qzeros, scales = _pack(n, k, group, seed=m * 101 + n + k)
    gen = torch.Generator().manual_seed(m * 7 + k)
    x = (torch.randn((m, k), generator=gen) * 0.5).to(torch.float16).cuda()
    out = matmul(x, qweight.cuda(), qzeros.cuda(), scales.cuda(), use_v2_format=v2,
                 prefill=m >= 16)
    ref = _reference(x.cpu(), codes, zeros, scales, 0 if v2 else 1)
    assert out.dtype == torch.float16 and out.shape == (m, n)
    torch.testing.assert_close(out.float().cpu(), ref, rtol=6e-2, atol=2.5)


@pytest.mark.parametrize("m", [1, 15, 16, 17, 64])
def test_decode_and_prefill_agree(m):
    """The two kernels are different tilings of one algorithm, so they must agree with each other."""

    n, k, group = 256, 256, 64
    _, _, qweight, qzeros, scales = _pack(n, k, group, seed=m + 5)
    gen = torch.Generator().manual_seed(m)
    x = (torch.randn((m, k), generator=gen) * 0.5).to(torch.float16).cuda()
    decode = matmul(x, qweight.cuda(), qzeros.cuda(), scales.cuda(), prefill=False)
    tiled = matmul(x, qweight.cuda(), qzeros.cuda(), scales.cuda(), prefill=True)
    torch.testing.assert_close(decode, tiled, rtol=2e-2, atol=2e-1)


def test_rejects_wrong_activation_dtype():
    """The RDNA2 path is fp16 only, so a bf16 activation is refused rather than silently wrong."""

    n, k, group = 64, 128, 32
    _, _, qweight, qzeros, scales = _pack(n, k, group, seed=3)
    x = torch.zeros((4, k), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError):
        matmul(x, qweight.cuda(), qzeros.cuda(), scales.cuda())


def _pack_experts(mats, experts, n, k, group, seed):
    """The dense packing, one expert at a time: qweight (mats, E, K / 8, N), qzeros (..., N / 8)."""

    gen = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (mats, experts, k, n), generator=gen)
    zeros = torch.randint(0, 16, (mats, experts, k // group, n), generator=gen)
    qw = torch.zeros((mats, experts, k // 8, n), dtype=torch.int64)
    for i in range(8):
        qw |= (codes[:, :, _PERM[i]::8, :] & 0xF) << (4 * i)
    qz = torch.zeros((mats, experts, k // group, n // 8), dtype=torch.int64)
    for i in range(8):
        qz |= (zeros[:, :, :, i::8] & 0xF) << (4 * i)
    scales = (torch.rand((mats, experts, k // group, n), generator=gen) * 0.2 + 0.02).to(torch.float16)
    return codes, zeros, qw.to(torch.int32), qz.to(torch.int32), scales


def _expert_weight(codes, zeros, scales, mat, expert, zero_offset):
    k = codes.shape[2]
    q = codes[mat, expert].t().float()
    z = zeros[mat, expert].float()
    sc = scales[mat, expert].float()
    groupsize = k // z.shape[0]
    w = torch.empty((q.shape[0], k))
    for gi in range(z.shape[0]):
        sl = slice(gi * groupsize, (gi + 1) * groupsize)
        w[:, sl] = (q[:, sl] - (z[gi].unsqueeze(1) + zero_offset)) * sc[gi].unsqueeze(1)
    return w


def _plan(picks, experts, tile):
    """TensorFold's plan: members sorted by expert, then one item per (expert, tile) run."""

    flat = picks.reshape(-1).to(torch.int32)
    members = torch.argsort(flat, stable=True).to(torch.int32)
    counts = torch.bincount(flat.to(torch.int64), minlength=experts)
    starts = torch.cumsum(counts, 0) - counts
    items = []
    for ex in range(experts):
        count = int(counts[ex])
        for off in range(0, count, tile):
            items.append((ex, int(starts[ex]) + off, min(count - off, tile)))
    return members, torch.tensor(items, dtype=torch.int32)


def _bf(v):
    return v.to(torch.bfloat16).float()


@pytest.mark.parametrize("epi", [0, 1, 2])
def test_moe_matches_reference(epi):
    """epi 0 checks the grouped GEMM against exact fp32; epi 1 and 2 check the epilogue against the
    kernel's own accumulators, so squaring or gating cannot turn a precision gap into a failure."""

    experts, k, n, group = 4, 256, 128, 64
    mats = 2 if epi == 2 else 1
    codes, zeros, qw, qz, scales = _pack_experts(mats, experts, n, k, group, seed=11 + epi)
    rows, slots, tile = 6, 3, 8
    picks = torch.randint(0, experts, (rows, slots), generator=torch.Generator().manual_seed(5))
    members, items = _plan(picks, experts, tile)
    x = (torch.randn(rows, k, generator=torch.Generator().manual_seed(21)) * 0.5).to(torch.bfloat16).cuda()
    pairs = (items.cuda(), members.cuda(), rows, slots)
    out = moe(x, qw.cuda(), qz.cuda(), scales.cuda(), *pairs, epi=epi)
    assert out.shape == (rows * slots, n)

    if epi == 0:
        w = [[_expert_weight(codes, zeros, scales, 0, ex, 0) for ex in range(experts)]][0]
        xf = x.float().cpu()
        ref = torch.empty(rows * slots, n)
        for pair in range(rows * slots):
            pid = int(members[pair])
            ref[pid] = xf[pid // slots] @ w[int(picks.reshape(-1)[pid])].t()
        torch.testing.assert_close(out.float().cpu(), ref, rtol=6e-2, atol=2.5)
        return

    def accumulators(mat: int) -> torch.Tensor:
        return moe(x, qw[mat:mat + 1].contiguous().cuda(), qz[mat:mat + 1].contiguous().cuda(),
                   scales[mat:mat + 1].contiguous().cuda(), *pairs, epi=0).cpu()

    if epi == 1:
        u = _bf(accumulators(0)).clamp(min=0.0)
        expect = _bf(u * u)
    else:
        gate, up = _bf(accumulators(0)), _bf(accumulators(1))
        expect = _bf(_bf(gate / (1.0 + torch.exp(-gate))) * up)
    torch.testing.assert_close(out.float().cpu(), expect, rtol=1e-3, atol=1e-3)
