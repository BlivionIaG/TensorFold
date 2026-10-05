"""A lane round's window forward gives every row its serial decode step's bits, and a commit keeps the serial caches."""

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
    pytest.skip("RDNA only", allow_module_level=True)

from tensorfold.rocm.model.window import Window, commit, window_forward  # noqa: E402
from tensorfold.rocm.model.qwen import Engine  # noqa: E402
from tensorfold.rocm.serving.engine import QwenEngine, _grow, caches_equal, clone_caches  # noqa: E402
from tests.rocm.test_qwen import _tiny  # noqa: E402


def _engine():
    model = _tiny(torch.device("cuda"))
    engine = QwenEngine(model, Engine(model, schedule="gemv", dtype=torch.bfloat16), eos=(0,))
    engine.graphs = False
    return engine


def _copy(engine, caches, total):
    """A private copy of prefilled caches, its key/value buffers ``total`` slots long."""

    return _grow(clone_caches(caches), total, engine._dtype(), engine._device())


def _serial(engine, caches, tokens, pos):
    rows, states = [], []
    for i, t in enumerate(tokens):
        hidden, caches = engine._forward([t], caches, pos + i, decode=True)
        rows.append(hidden[0, -1].clone())
        states.append(clone_caches(caches))
    return rows, states


@pytest.mark.parametrize("keep", [1, 3, 5])
def test_a_window_is_its_serial_steps(keep):
    engine = _engine()
    prompt = [1 + (i * 7) % 47 for i in range(40)]
    _, caches = engine._span(prompt, None, 0, len(prompt) + 16)
    tokens = [5, 9, 13, 2, 7]
    total = len(prompt) + 16
    want, states = _serial(engine, _copy(engine, caches, total), tokens, len(prompt))
    window = Window(tokens, _copy(engine, caches, total), len(prompt))
    with torch.inference_mode():
        got = window_forward(engine.model, [window], engine.kernels.linear, engine._dtype())
    for i in range(len(tokens)):
        assert torch.equal(got[0, i], want[i]), i
    commit(engine.model, window, keep)
    assert caches_equal(window.caches, states[keep - 1])


def test_windows_of_several_streams_keep_their_own_bits():
    engine = _engine()
    first, second = [1 + (i * 5) % 40 for i in range(30)], [1 + (i * 11) % 40 for i in range(55)]
    _, a = engine._span(first, None, 0, len(first) + 16)
    _, b = engine._span(second, None, 0, len(second) + 16)
    want_a, _ = _serial(engine, _copy(engine, a, 64), [4, 8, 15], len(first))
    want_b, _ = _serial(engine, _copy(engine, b, 80), [16, 23], len(second))
    windows = [Window([4, 8, 15], _copy(engine, a, 64), len(first)), Window([16, 23], _copy(engine, b, 80), len(second))]
    with torch.inference_mode():
        got = window_forward(engine.model, windows, engine.kernels.linear, engine._dtype())
    for i, row in enumerate(want_a + want_b):
        assert torch.equal(got[0, i], row), i


def test_a_window_through_routed_experts_is_its_serial_steps(tmp_path):
    from tensorfold.rocm.model import qwen as qwen_mod
    from tests.rocm.test_moe import _checkpoint

    model = qwen_mod.load(_checkpoint(tmp_path, gptq=False), torch.device("cuda"))
    engine = QwenEngine(model, Engine(model, schedule="auto", dtype=qwen_mod.activation_dtype(_gfx())), eos=(0,))
    engine.graphs = False
    prompt = [1 + (i * 7) % 40 for i in range(33)]
    _, caches = engine._span(prompt, None, 0, len(prompt) + 16)
    tokens = [5, 9, 13, 2]
    want, states = _serial(engine, _copy(engine, caches, 64), tokens, len(prompt))
    window = Window(tokens, _copy(engine, caches, 64), len(prompt))
    with torch.inference_mode():
        got = window_forward(engine.model, [window], engine.kernels.linear, engine._dtype())
    for i in range(len(tokens)):
        assert torch.equal(got[0, i], want[i]), i
    commit(engine.model, window, 2)
    assert caches_equal(window.caches, states[1])


def _gfx() -> str:
    from tensorfold.rocm.kernels.build import gfx_name

    return gfx_name()
