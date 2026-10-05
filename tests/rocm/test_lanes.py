"""Lane rounds: requests decoding together each give their solo reply, drafted or serial, greedy or sampled."""

import threading

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
    pytest.skip("RDNA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tests.rocm.test_mtp import _engine, _served  # noqa: E402

PROMPTS = [[1, 2, 3, 4], [5, 6, 7, 8, 9, 10], [11, 3, 7], [2, 4, 6, 8, 10, 12, 14]]


def _alone(engine, prompt, sampling, draft=True):
    got = []
    engine.generate(prompt, 12, sampling, got.extend, stop_eos=False, draft=draft)
    return got


def _together(engine, samplings, drafts):
    out = [[] for _ in PROMPTS]
    errors = []

    def run(i):
        try:
            engine.generate(PROMPTS[i], 12, samplings[i], out[i].extend, stop_eos=False, draft=drafts[i])
        except Exception as exc:                      # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(PROMPTS))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    return out


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_streams_decoding_together_each_equal_their_solo_run(tmp_path, temperature):
    solo, model = _served(tmp_path)
    many = _engine(model, solo.kernels)
    many.streams, many.concurrent = 4, True
    samplings = [None if temperature == 0 else Sampling(seed=20 + i, temperature=temperature) for i in range(4)]
    drafts = [True, False, True, True]
    want = [_alone(solo, p, s, d) for p, s, d in zip(PROMPTS, samplings, drafts)]
    got = _together(many, samplings, drafts)
    solo.close()
    many.close()
    assert got == want


def test_a_drafted_reply_equals_its_serial_one_in_lane_rounds(tmp_path):
    engine, _ = _served(tmp_path)
    for i, prompt in enumerate(PROMPTS):
        sampling = Sampling(seed=40 + i, temperature=1.0)
        assert _alone(engine, prompt, sampling, True) == _alone(engine, prompt, sampling, False), i
    engine.close()
