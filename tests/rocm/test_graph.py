"""The captured decode step: device-position kernels equal the eager ones bit for bit, and so does the engine."""

import os

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("RDNA only", allow_module_level=True)

from tensorfold.rocm.build import gfx_name  # noqa: E402
from tensorfold.rocm.qwen import activation_dtype  # noqa: E402
from tensorfold.rocm.qwen_math import DevicePos, apply_rope  # noqa: E402

try:
    _ACT = activation_dtype(gfx_name())
except RuntimeError:
    pytest.skip("not an RDNA device", allow_module_level=True)


@pytest.mark.parametrize("pos", [0, 5, 127, 128, 299])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_attention_at_a_device_position_equals_the_sliced_cache(pos, dtype):
    """Keys past the position are never read, and the extra zero tiles leave every bit of the output alone."""

    from tensorfold.rocm.attention import causal, causal_at

    gen = torch.Generator().manual_seed(pos)
    heads, kv_heads, d, span = 8, 2, 128, 300
    q = torch.randn((1, heads, 1, d), generator=gen).cuda()
    k = torch.randn((1, kv_heads, span, d), generator=gen).to(dtype).cuda()
    v = torch.randn((1, kv_heads, span, d), generator=gen).to(dtype).cuda()
    at = DevicePos(q.device)
    at.set(pos)
    want = causal(q, k[:, :, :pos + 1], v[:, :, :pos + 1], d ** -0.5, pos)
    got = causal_at(q, k, v, d ** -0.5, at.i32)
    assert torch.equal(got, want)


@pytest.mark.parametrize("exact", [True, False])
@pytest.mark.parametrize("pos", [0, 17, 4095])
def test_rope_at_a_device_position_equals_the_integer_one(exact, pos):
    x = torch.randn((1, 8, 1, 256)).to(_ACT).cuda()
    at = DevicePos(x.device)
    at.set(pos)
    want = apply_rope(x, pos, 10_000_000.0, 64, exact=exact)
    got = apply_rope(x, 0, 10_000_000.0, 64, exact=exact, at=at)
    assert torch.equal(got, want)


def test_graph_decode_equals_eager_on_a_real_checkpoint():
    """Greedy ids of captured steps against eager ones, on a checkpoint with full and linear attention layers."""

    model_dir = os.environ.get("TENSORFOLD_GOLDEN_MODEL")
    if not model_dir:
        pytest.skip("set TENSORFOLD_GOLDEN_MODEL to a Qwen3.5 checkpoint")
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.rocm.engine import QwenEngine

    engine = QwenEngine.load(model_dir, keep=0, no_drafts=True)
    prompt = list(range(1000, 1037))
    runs = {}
    for graphs in (False, True):
        engine.graphs = graphs
        got = []
        engine.generate(prompt, 24, Sampling(seed=3, temperature=0.0), got.extend, stop_eos=False, draft=False)
        runs[graphs] = got
    assert len(runs[True]) == 24 and runs[True] == runs[False]


def test_the_opt_in_wmma_schedule_keeps_graph_and_eager_equal():
    """TENSORFOLD_ROCM_SCHEDULE=wmma runs WMMA at every row count, so its replayed steps equal its eager ones."""

    from tensorfold.rocm.build import WMMA

    model_dir = os.environ.get("TENSORFOLD_GOLDEN_MODEL")
    if not model_dir or gfx_name() not in WMMA:
        pytest.skip("set TENSORFOLD_GOLDEN_MODEL on a gfx11 / gfx12 part")
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.rocm.engine import QwenEngine

    engine = QwenEngine.load(model_dir, keep=0, no_drafts=True, schedule="wmma")
    runs = {}
    for graphs in (False, True):
        engine.graphs = graphs
        got = []
        engine.generate(list(range(1000, 1037)), 16, Sampling(seed=3, temperature=0.0), got.extend, stop_eos=False,
                        draft=False)
        runs[graphs] = got
    assert len(runs[True]) == 16 and runs[True] == runs[False]
