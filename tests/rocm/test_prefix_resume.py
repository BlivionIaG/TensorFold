"""A prompt resumed from the token before its end matches a fresh prefill of the same ids."""

import pytest
torch = pytest.importorskip("torch")

if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
    pytest.skip("RDNA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.rocm.serving.engine import QwenEngine, caches_equal  # noqa: E402
from tensorfold.rocm.model.qwen import Engine  # noqa: E402
from tests.rocm.test_qwen import _tiny  # noqa: E402


def _engine():
    model = _tiny(torch.device("cuda"))
    return QwenEngine(model, Engine(model, schedule="gemv", dtype=torch.bfloat16), eos=(0,))


def test_resumed_prefill_matches_a_fresh_one():
    prompt = [3, 5, 7, 9, 11, 13]
    first = _engine()
    greedy = Sampling(seed=1, temperature=0)
    seen = []
    stats = first.generate(prompt, 2, greedy, lambda tokens: seen.extend(tokens) or False, stop_eos=False)
    assert stats["cached"] == 0
    assert len(seen) == 2
    extended = prompt + seen[:1]
    resumed = []
    again = first.generate(extended, 2, greedy, lambda tokens: resumed.extend(tokens) or False, stop_eos=False)
    assert again["cached"] == len(prompt) - 1
    fresh = _engine()
    other = []
    fresh.generate(extended, 2, greedy, lambda tokens: other.extend(tokens) or False, stop_eos=False)
    assert resumed == other
    held = first.cache.longest(extended)
    assert held is not None
    assert caches_equal(held[1], fresh.prefill_caches(extended[:len(held[0])]))


@pytest.mark.parametrize("cuts", [(16,), (16, 22), (1, 65, 137), (199,)])
def test_a_prompt_prefilled_in_pieces_matches_one_span(cuts):
    """Every prefill row's bits are its own: a resume from any cut repeats a fresh prefill."""

    model = _tiny(torch.device("cuda"))
    engine = QwenEngine(model, Engine(model, schedule="gemv", dtype=torch.bfloat16), eos=(0,))
    prompt = [1 + (index * 7) % 47 for index in range(200)]
    whole, fresh = engine._span(prompt, None, 0, len(prompt))
    caches, start = None, 0
    for stop in (*cuts, len(prompt)):
        hidden, caches = engine._span(prompt[start:stop], caches, start, len(prompt))
        start = stop
    assert torch.equal(hidden[0, -1], whole[0, -1])
    assert caches_equal(caches, fresh)


def test_a_message_start_matches_a_fresh_prefix_and_the_next_turn():
    prompt = [3, 5, 7, 9, 11, 13, 15, 17]
    first = _engine()
    first.points = lambda ids: [4] if len(ids) >= 8 else []
    greedy = Sampling(seed=1, temperature=0)
    seen = []
    first.generate(prompt, 1, greedy, lambda tokens: seen.extend(tokens) or False, stop_eos=False)
    held = first.cache.named(prompt, 4)
    assert held is not None and held[0] == prompt[:4]
    fresh_prefix = _engine()
    assert caches_equal(held[1], fresh_prefix.prefill_caches(prompt[:4]))
    extended = prompt + seen
    resumed = []
    again = first.generate(extended, 1, greedy, lambda tokens: resumed.extend(tokens) or False, stop_eos=False)
    assert again["cached"] == 4
    fresh = _engine()
    fresh.points = first.points
    other = []
    fresh.generate(extended, 1, greedy, lambda tokens: other.extend(tokens) or False, stop_eos=False)
    assert resumed == other


def test_each_request_stops_on_its_own_end_token():
    engine = _engine()
    engine.eos = (9,)
    script = iter([1, 9, 2, 2, 9])
    engine._sample = lambda *_args, **_kwargs: next(script)
    greedy = Sampling(seed=1, temperature=0)
    short, long = [], []
    engine.generate([3, 5, 7], 5, greedy, lambda tokens: short.extend(tokens) or False)
    engine.generate([3, 5, 7, 9, 11], 5, greedy, lambda tokens: long.extend(tokens) or False)
    assert short == [1, 9]
    assert long == [2, 2, 9]


def test_generate_accepts_the_server_grammar_argument():
    import inspect

    names = inspect.signature(QwenEngine.generate).parameters
    for name in ("draft", "stop_eos", "constraint", "background"):
        assert name in names
