"""Host checks for the serving-cell harness: the median line and the request prompts. No GPU."""

import pytest

pytest.importorskip("torch")

from tensorfold.rocm.bench import RATES, _prompts, median_row  # noqa: E402


def _row(rate: float) -> dict:
    row = {"prompt": 1024, "generated": 512, "concurrency": 8, "prefill_tokens": 8192, "decode_tokens": 4088}
    row.update({key: rate for key in RATES})
    return row


def test_the_median_line_takes_each_rate_over_the_runs():
    got = median_row([_row(3.0), _row(1.0), _row(2.0)])
    assert all(got[key] == 2.0 for key in RATES)
    assert (got["prompt"], got["generated"], got["concurrency"]) == (1024, 512, 8)


def test_requests_are_distinct_and_avoid_token_zero():
    prompts = _prompts(64, 2, 8, vocab=1000)
    assert len(prompts) == 8 and all(len(p) == 64 for p in prompts)
    assert len({tuple(p) for p in prompts}) == 8
    assert all(0 < token < 1000 for p in prompts for token in p)


def test_a_cell_needs_two_generated_tokens():
    with pytest.raises(ValueError):
        _prompts(64, 1, 1, vocab=1000)
